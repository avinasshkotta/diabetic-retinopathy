# AWS free-tier deploy: EC2 t2.micro (1 vCPU / 1 GB / 30 GB, 750 hrs/mo free)
#
# ONE-TIME local prep:
#   docker build -f Dockerfile.cpu -t dr-api:cpu .
#   docker save dr-api:cpu -o dr-api-cpu.tar     # ~3 GB, scp'd once to the instance
#
# What this script does on the instance:
#   installs Docker, transfers the image, runs the container on port 80,
#   then health-checks it.  Result: http://<ip>/  (+ /docs Swagger UI)
#
# Prereqs: AWS CLI configured (aws configure) or launch the instance in the
# console; an EC2 key pair (.pem) saved locally; SG allowing 22 + 80 from 0.0.0.0/0.
#
param(
    [Parameter(Mandatory)][string]$PublicIp,
    [string]$Key = "$env:USERPROFILE\.ssh\dr-key.pem",
    [string]$User = "ec2-user"     # Amazon Linux 2023; use ubuntu for Ubuntu AMIs
)

Write-Host "==> waiting for SSH on $PublicIp ..."
$up = $false
for ($i = 0; $i -lt 30; $i++) {
    ssh -i $Key -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes `
        "$User@$PublicIp" "echo ok" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $up = $true; break }
    Start-Sleep -Seconds 5
}
if (-not $up) { throw "SSH never became reachable — check SG port 22 + key path" }

$cmds = @(
    'sudo dnf -y install docker > /dev/null 2>&1 || sudo yum -y install docker',
    'sudo systemctl enable --now docker',
    'sudo usermod -aG docker $USER'
)
foreach ($c in $cmds) {
    ssh -i $Key "$User@$PublicIp" $c
}

Write-Host "==> transferring image (~3 GB, one time, LAN/WAN dependent)..."
scp -i $Key dr-api-cpu.tar "${User}:`$HOME/"

Write-Host "==> loading + running container..."
ssh -i $Key "$User@$PublicIp" @"
docker load -i `$HOME/dr-api-cpu.tar
docker rm -f dr-api 2>/dev/null
docker run -d --restart unless-stopped --name dr-api -p 80:8080 dr-api:cpu
sleep 8
curl -fsS http://127.0.0.1:8080/health && echo
curl -fsS http://127.0.0.1:8080/ready && echo
"@

Write-Host ""
Write-Host "LIVE: http://$PublicIp/" -ForegroundColor Green
Write-Host "Docs: http://$PublicIp/docs   (Swagger UI)"
