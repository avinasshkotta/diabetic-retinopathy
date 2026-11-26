# Legacy: AWS deployment (retired)

The service ran on an AWS EC2 `t2.micro` with a DuckDNS name that the instance
re-pointed at its own public IP on every boot. It has been replaced by **Azure
Container Apps**, deployed by GitHub Actions (see the repository README and
`.github/workflows/ci.yml`).

Kept here for history and for interview questions about the migration — not as
instructions. The files reference paths that no longer exist in the repo root.

| File | What it was |
|---|---|
| `README-aws.md` | EC2 launch, security group, Docker build, DuckDNS setup |
| `deploy-ec2.ps1` | SSH from Windows → build image on the instance → health check |
| `add-dynamic-url.sh` | systemd service + timer that updated the DuckDNS A record |
| `live-url.ps1` | printed the current public URL without the AWS console |
| `deploy.ps1` | ECS service update (staging/prod) used by the old CI |
| `ecs-task-definition.json` | ECS Fargate task definition template |

## Why it moved

- EC2 needs patching, key management and a public IP; Container Apps manages all three.
- A stopped instance means a dead URL. Scale-to-zero on Container Apps is $0 **and**
  the URL keeps working.
- One CI/CD pattern now covers all three projects: build → GHCR → `az containerapp update`,
  authenticated with GitHub OIDC (no stored Azure secret).
