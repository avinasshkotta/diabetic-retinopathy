# diabetic-retinopathy — MLOps pipeline

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

- [x] Feature engineering → gold layer
- [ ] Step 2: MLflow experiment tracking + model registry
- [ ] Step 3: FastAPI endpoint
- [ ] Step 4: Docker container
- [ ] Step 5: GitHub Actions CI/CD
- [ ] Step 6: AWS ECS / Kubernetes
- [ ] Step 7: Evidently drift monitor + HITL gate
- [ ] Step 8: retrain trigger
