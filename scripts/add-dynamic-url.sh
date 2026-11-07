#!/usr/bin/env bash
# add-dynamic-url.sh — the EC2 instance keeps a free DuckDNS name pointed at
# its CURRENT public IP, updated at every boot + every 15 minutes.
#
# Run ONCE on the instance (e.g. from deploy-ec2.ps1 or ssh):
#     sudo bash add-dynamic-url.sh <domain-without-.duckdns.org> <duckdns-token>
#
# Result: http://<domain>.duckdns.org/docs works after every stop/start, $0.
# Get the free token at https://www.duckdns.org (login with Google/GitHub).
set -euo pipefail

DOMAIN="$1"
TOKEN="$2"

# --- AWS metadata: read this instance's own public IP (IMDSv2) --------------
imds_token=$(curl -sX PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $imds_token" \
  "http://169.254.169.254/latest/meta-data/public-ipv4")
echo "instance public IP: $PUBLIC_IP"
[ -n "$PUBLIC_IP" ] || { echo "could not read metadata (no public IPv4?)"; exit 1; }

# --- update the record right now so the URL starts working immediately -------
reply=$(curl -s "https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=${PUBLIC_IP}")
echo "duckdns response: $reply"
[ "$reply" = "OK" ] || { echo "duckdns rejected the update - check domain/token"; exit 1; }

# --- the updater the machine will keep running -------------------------------
cat > /usr/local/bin/duckdns-update.sh <<UPDATER
#!/usr/bin/env bash
T=\$(curl -sX PUT http://169.254.169.254/latest/api/token \
     -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
IP=\$(curl -s -H "X-aws-ec2-metadata-token: \$T" \
     http://169.254.169.254/latest/meta-data/public-ipv4)
[ -n "\$IP" ] || exit 0
curl -s "https://www.duckdns.org/update?domains=${DOMAIN}&token=${TOKEN}&ip=\$IP" \
  > /var/log/duckdns.log 2>&1
echo "\$(date -Is) duckdns: ${DOMAIN}.duckdns.org -> \${IP}" >> /var/log/duckdns.log
UPDATER
chmod +x /usr/local/bin/duckdns-update.sh

# --- systemd: run on every boot (IP change happens exactly then) -------------
cat > /etc/systemd/system/duckdns.service <<UNIT
[Unit]
Description=Point DuckDNS name at this instance's public IP (at boot)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/duckdns-update.sh

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/duckdns.timer <<UNIT
[Unit]
Description=Refresh the DuckDNS record periodically

[Timer]
OnBootSec=1min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now duckdns.service duckdns.timer
echo
echo "DONE. http://${DOMAIN}.duckdns.org/docs should resolve to ${PUBLIC_IP} within ~1 minute"
