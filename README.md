# diabetic-retinopathy — MLOps pipeline

[![ci-cd](https://github.com/avinasshkotta/diabetic-retinopathy/actions/workflows/ci.yml/badge.svg)](https://github.com/avinasshkotta/diabetic-retinopathy/actions/workflows/ci.yml)

**Live API:** http://dr-grading.duckdns.org/ · Swagger UI: http://dr-grading.duckdns.org/docs

> The URL is a free dynamic-DNS record the instance updates itself (with its
> current public IP) at every boot and every 15 min (`scripts/add-dynamic-url.sh`
> → systemd `duckdns.service`/`duckdns.timer`). It survives stop/starts, where a
> raw public IP would die. `$0` — no Elastic IP needed.
> AWS free-tier t2/t3 micro, Docker, CPU image — see `README-aws.md`.

📖 **New here? Read [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)** — the whole
process start to finish: architecture, every stage, the two core flows
(request + retrain), a 2-minute demo script, and what each file does.

Production-oriented repo for the DR grading model (EfficientNet-B0, 3-seed
ensemble, APTOS-2019).  Research notebook lives in the original PyCharm
folder; this repo is the pipeline.

## Layout

```
data/raw/     -> junction to the 8 GB dataset (never committed)
data/gold/    -> frozen split manifests (train/val/test CSVs, generated ONCE)
models/       -> .pt checkpoints + dr_metadata.json (deployment contract)
src/dr_pipeline/
    config.py     single source of truth: paths, hyperparameters, eval gate
    data.py       raw -> gold layers, split manifests, Dataset/DataLoader
    train.py      CLI training (one seed / ensemble / smoke)
    evaluate.py   test-set metrics + eval_gate() promotion decision
    predict.py    single-image inference core (wraps the FastAPI Step 3)
tests/        fast structural tests (leakage, stratification, gate logic)
reports/      generated metrics (gitignored)
```

## Commands

```bash
export PYTHONPATH=src            # Windows: $env:PYTHONPATH="src"

python -m dr_pipeline.data make-split      # once — freezes the split
python -m dr_pipeline.data inspect
python -m dr_pipeline.train --all-seeds    # 3-seed ensemble (~10 min GPU)
python -m dr_pipeline.train --seed 42 --smoke   # CI plumbing check
python -m dr_pipeline.evaluate --ensemble --gate  # promotion decision
python -m dr_pipeline.predict path/to/fundus.png
pytest tests -q
```

## Eval gate (CI blocks promotion below these)

| metric | required | measured |
|---|---|---|
| ensemble AUC | ≥ 0.86 | 0.8821 |
| referable AUC | ≥ 0.90 | 0.9627 |
| sensitivity (t=0.3537) | ≥ 0.80 | 0.953 |
| specificity (t=0.3537) | ≥ 0.70 | 0.864 |

## Roadmap (mentor's diagram)

- [x] Feature engineering → gold layer (frozen split manifests)
- [x] MLflow experiment tracking (train/eval runs, gate tags)
- [x] MLflow model registry (`dr-effnetb0`, gated `production` alias)
- [x] FastAPI endpoint (`/health /ready /model /predict`)
- [x] Docker container (multi-stage, non-root, healthcheck)
- [x] GitHub Actions CI/CD (test → eval-gate → image → staging → manual prod)
- [x] AWS ECS (task definition + `scripts/deploy.ps1`; canary via service rollout)
- [x] Evidently drift monitor + HITL gate (`monitor.py`, abstain routing)
- [x] Retrain trigger (`retrain.py`: drift / reviewed-labels / schedule)

### Local services

```bash
# mlflow (port 5001 — Windows reserves 5000)
py -m mlflow server --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root mlartifacts --host 127.0.0.1 --port 5001
# API
$env:PYTHONPATH="src"; py -m uvicorn dr_pipeline.serving:app --port 8080
# registry-backed serving
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
$env:MODEL_URI="models:/dr-effnetb0@production"
```
