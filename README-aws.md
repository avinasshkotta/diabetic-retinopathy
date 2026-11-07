# Going live on AWS free tier (Option 1: EC2 t2.micro)

Cost: **$0 for 12 months** (750 hrs/mo t2.micro + 30 GB EBS + static public IP
in free tier). This is the version to show your mentor — a real server with a
public URL in your own AWS account.

## 1. Launch the instance (console, ~3 min)

1. AWS Console → **EC2 → Launch instances**
2. Name: `dr-api`
3. AMI: **Amazon Linux 2023** (free-tier eligible, marked ✓)
4. Instance type: **t2.micro** (Free tier eligible)
5. Key pair: create new `dr-key` → **download the .pem** → save it to
   `%USERPROFILE%\.ssh\dr-key.pem`
6. Network settings → **Edit** → Add security group rule:
   - Type **HTTP**, Port **80**, Source **0.0.0.0/0**
   - (keep the default SSH rule for port 22, source = your IP)
7. Launch. Wait for **Running**. Note the **Public IPv4 address**.

> Free tier warning: tick "Stop" not "Terminate" when done each month's use,
> and stop the instance when not demoing — 750 hrs ≈ 31 days, you only get
> that if it runs 24/7 for one month. Stopping = $0 always (storage still free).

> **Permanent free URL:** every stop/start gives the instance a NEW public IP,
> so raw `http://<ip>/` links die on restart. The instance therefore updates a
> free dynamic-DNS record (`dr-grading.duckdns.org`) with its current IP at every
> boot + every 15 min — installed via `scripts/add-dynamic-url.sh` (systemd
> `duckdns.service` / `duckdns.timer`). Share **http://dr-grading.duckdns.org/docs**,
> never the raw IP. To rebuild this from scratch on a fresh instance:
>
> ```bash
> scp scripts/add-dynamic-url.sh ec2-user@<ip>:
> ssh ec2-user@<ip> "sudo bash add-dynamic-url.sh <domain> <duckdns-token>; rm add-dynamic-url.sh"
> ```
>
> Local helper: `scripts/live-url.ps1` prints the current URL on demand
> (needs one-time `winget install Amazon.AWSCLI` + `aws configure`, read-only key).

## 2. Build the CPU image locally (once)

```powershell
cd "C:\Users\avina\PycharmProjects\JupyterProject\diabetic-retinopathy"
docker build -f Dockerfile.cpu -t dr-api:cpu       # ~3 GB (CUDA-free)
docker save dr-api:cpu -o dr-api-cpu.tar
```

Why CPU image: t2.micro has no GPU and 1 GB RAM. `Dockerfile.cpu` pulls
`torch+cpu` (no 8 GB of CUDA libs) and defaults to the **single-seed model**
(`DR_SEEDS=42`, ~150 MB RAM) instead of the 3-model ensemble.

## 3. Deploy

```powershell
.\scripts\deploy-ec2.ps1 -PublicIp <YOUR_IP> -Key "$env:USERPROFILE\.ssh\dr-key.pem"
```

The script: waits for SSH → installs Docker → scp's the image → runs the
container on port 80 → curls `/health` and `/ready`.

## 4. Verify it's live

- `http://<YOUR_IP>/health` → `{"status":"ok"}`
- `http://<YOUR_IP>/ready` → `{"ready":true,...}`
- `http://<YOUR_IP>/docs` → **Swagger UI** — click `POST /predict`, upload a
  fundus image from `Diabetic-retinopathy dataset\train_images\`, Execute.
  This Swagger page is the demo to show your mentor.

## 5. Costs & hygiene

| item | free-tier |
|---|---|
| t2.micro 750 hrs/mo | $0 (12 mo) |
| 8 GB EBS gp3 | $0 (30 GB included) |
| public IPv4 (idle) | ~$0.005/hr — **stop the instance** when not using |
| data out | 100 GB/mo free |

- **Stop** the instance after demos (`aws ec2 stop-instances`).
- Never commit the `.pem`. Add to `.gitignore`.
- This serves a screening demo, not a diagnosis tool — keep the README
  disclaimer on anything you share publicly.

## 6. Optional: keep the box updated via CI

Later (when the repo is on GitHub), the `deploy-ec2` job can ssh in and
`docker pull`/reload on push — same gate structure as the ECS workflow,
just pointed at this instance. The ECS workflow in `.github/workflows/ci.yml`
stays as the "this is how it scales on real infra" artifact for interviews.
