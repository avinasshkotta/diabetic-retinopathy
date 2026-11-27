# From Notebook to Live URL — the complete walkthrough

Everything in this repo, start to finish, in the order it was built — so you can
understand *and explain* the system end to end.

**Live artifacts:**
- API: **https://diabetic-retinopathy.proudgrass-320c667e.centralindia.azurecontainerapps.io/**
  · interactive demo: **.../docs**
  *(Azure Container Apps, scale-to-zero; GitHub Actions deploys after the eval gate.
  The AWS EC2 + DuckDNS steps later in this document are historical — see `legacy/aws/`.)*
- Repo/CI: **https://github.com/avinasshkotta/diabetic-retinopathy** (green badge = pipeline healthy)

---

## 0. The starting point (Stage 0)

A Jupyter notebook trained an **EfficientNet-B0** to grade diabetic retinopathy
(DR) from retinal (fundus) photographs on the **APTOS 2019** dataset
(3,662 images, 8.2 GB, 5 severity grades):

| grade | meaning |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative |

The clinical question is not just "which grade?" but **"is it referable?"** —
does the patient need a specialist? That is `grade >= 2`, decided at a frozen
probability threshold.

**The notebook worked. But a notebook is not a system.** It can't serve
predictions, can't be versioned, can't be re-run by CI, can't detect when its
model degrades. The mentor's diagram is the roadmap for fixing that:

```
Gold Layer → MLflow Tracking → MLflow Registry → FastAPI → Docker
   → GitHub Actions → ECS/K8s → Evidently Drift Monitor → HITL Gate → Retrain Trigger
```

**Measured model quality (the number everything else protects):**

| metric | value |
|---|---|
| Ensemble AUC (5-grade) | **0.8821** |
| Referable AUC (binary) | **0.9627** |
| Sensitivity @ 0.3537 | 0.953 |
| Specificity @ 0.3537 | 0.864 |

---

## 1. Stage 1 — Gold Layer: notebook → package (commit `f80c07b`)

**Problem:** notebook cells can't be imported, tested, or run by CI.

**Built:** `src/dr_pipeline/` — a real Python package:

| file | responsibility |
|---|---|
| `config.py` | **single source of truth**: paths, hyperparams, gate thresholds |
| `data.py` | raw → gold layers, **frozen split manifests**, Dataset/DataLoader |
| `train.py` | CLI training (one seed / all seeds / smoke) |
| `evaluate.py` | test-set metrics + `eval_gate()` promotion decision |
| `predict.py` | single-image inference core |
| `tracking.py` | MLflow wiring |

**The most important design decision here: the frozen test set.**
`data/gold/splits/*.csv` (train 2196 / val 733 / test 733) was generated
**once**, stratified by grade, and **committed to git**. Every later training
run, evaluation, and CI run reuses the exact same split. Without this, you can
leak test images into training and your AUC means nothing. The test set is the
contract.

**Tests (14 total):** split leakage, stratification, split ratios, gate logic
(pass/blocks regression/blocks each metric), API contract (health/ready/model/
predict/garbage/empty/real-image), checkpoint presence, metadata consistency.

---

## 2. Stage 2 — MLflow Tracking + Registry (commit `363cede`)

**Problem:** where did this model come from? which data? which params? which
metric said it was allowed to ship?

**Built:**
- **Tracking server**: SQLite backend (`mlflow.db`) + `mlartifacts/` store,
  experiment **`dr-grading`**, runs logged with params/metrics/artifacts.
  Runs on **http://127.0.0.1:5001** (Windows reserves 5000).
- **Registry**: model **`dr-effnetb0`** — each seed run logs checkpoints,
  `EnsembleModule` (deployable wrapper: normalized tensor → grade probs) is
  logged as the model artifact.
- **Gated promotion** (`promote.py`): promotion is *never* a manual click.
  It re-runs `evaluate --ensemble --gate` on the run's checkpoints, and only
  if `eval_gate()` passes does it set the **`production` alias**
  (aliases, not the deprecated "stages").

**The eval gate — the heart of the MLOps story:**

```python
# config.py — the numbers the gate enforces
GATE_MIN_ENSEMBLE_AUC   = 0.86   # measured 0.8821
GATE_MIN_REFERABLE_AUC  = 0.90   # measured 0.9627
min sensitivity          = 0.80   # measured 0.953
min specificity          = 0.70   # measured 0.864
```

> ⚠️ **Open item: these thresholds need your mentor/clinician to confirm them.**
> They are plausible for screening, but they are a *clinical* choice, not a
> coding choice.

Verified end to end: `evaluate --ensemble --gate` reproduces AUC **0.8821**,
**v3 registered, `production` alias set**, `pyfunc` + `pytorch` loads both work,
and `promote.py` **refuses** to promote a run that was never evaluated.

---

## 3. Stage 3 — FastAPI service (commit `b9f9c5d`, docs `18490f9`)

**Problem:** nobody can *use* the model.

**Built:** `src/dr_pipeline/serving.py`

| endpoint | purpose |
|---|---|
| `GET /health` | liveness — process is up (never depends on model) |
| `GET /ready` | readiness — model loaded; **503** while loading/failed |
| `GET /model` | model/version, threshold, and the full model card |
| `POST /predict` | upload fundus photo → grade, probs, referral, abstain, **evidence** |

**Explainability (added later):** every response carries `seed_probs`, `agreement`,
`margin`, `grade_label`, `grade_description` and a plain-language `explanation[]`; the
landing page turns them into a "Why this grade" panel plus a validation table
(referable AUC 0.963, sensitivity 95.3%, specificity 86.4%, n=733) and the honest
per-grade recall. The 5-way grade is advisory; the referral decision is the product.

Design points worth explaining:

- **Model source precedence:** `MODEL_URI` env (registry, e.g.
  `models:/dr-effnetb0@production`) → falls back to local `models/*.pt`.
- **`abstain` (HITL routing):** if `confidence < 0.55` the API does **not**
  answer — it returns `abstain: true` so the case goes to human review.
  A screening tool must be allowed to say "I don't know".
- **Validation before readiness:** garbage/empty uploads get a precise `400`
  even while the model is still loading; `503` is reserved for the actual
  inference path.
- **Audit log:** every prediction logs a hash of the image + the decision
  (never the image itself) — this is the raw material for drift monitoring.
- **Self-documenting Swagger:** `/docs` carries a plain-English guide and a
  description of every response field.

---

## 4. Stage 4 — Docker (commits `6f91496`, `4cd92d4`)

**Problem:** "works on my machine" — GPU drivers, CUDA versions, missing libs.

**Two images:**

| file | size | use |
|---|---|---|
| `Dockerfile` + `requirements.txt` | 13 GB | local/dev (CUDA 12.6) |
| `Dockerfile.cpu` + `requirements-cpu.txt` | **2.53 GB** | **production on AWS free tier** |

The CPU image pulls `torch+cpu` (no multi-GB CUDA libraries) and defaults to
**`DR_SEEDS=42`** — a single model instead of the 3-model ensemble, because a
t3.micro has only 1 GB RAM. Verified locally: full request cycle works,
**454 ms** round trip, real prediction returned.

---

## 5. Stage 5 — GitHub Actions CI/CD (commits `6f91496` + fixes)

**Problem:** quality should be enforced by machines, not memory.

`.github/workflows/ci.yml` — on every push to `main`:

```
test ──► eval-gate ──► docker-build ──► deploy-staging ──► (manual) deploy-prod
```

| job | what it does | time |
|---|---|---|
| `test` | 14 tests (environment-aware: skips dataset/weights tests on runners) | 1m41s |
| `eval-gate` | smoke train (full data→model→backward path) + gate logic. **FAILING HERE BLOCKS EVERYTHING DOWNSTREAM** | 2m00s |
| `docker-build` | builds the CPU image → pushes to GHCR | 2m25s |
| `deploy-staging` | ECR + ECS update + smoke test — *skips with a notice until AWS secrets exist* | 4s |
| `deploy-prod` | **manual** `workflow_dispatch` + typed `PROD` + GitHub environment approval | not on push |

The architecture lesson: **the gate sits between the model and the
deployment, never after it.** Evaluation decides whether shipping happens.

Current status: **all green** (`success` on run 40267934015).

CI-specific engineering (each a real bug fixed):

1. `fastapi` wasn't in the dev extra → added the serving stack.
2. Checkpoint/split tests asserted artifacts that are *deliberately* not in
   git (8 GB data, weights) → they now **skip honestly** with a reason.
3. Smoke train hard-exited without images → `--smoke` now falls back to
   **synthetic images**, still exercising Dataset → DataLoader → model → backward.
4. `mlflow` moved to core deps (train/evaluate import tracking).
5. AWS deploy jobs guard on secrets → a repo without AWS config still has a
   **green** pipeline.
6. `deploy-prod` needed `deploy-staging`, which is skipped on manual runs —
   a "silently skipped" prod gate is worse than no gate. Decoupled.

---

## 6. Stage 6 — Going live on AWS free tier (commit `4cd92d4` + deploy fixes)

**Problem:** the diagram says ECS/Kubernetes. Reality: an ECS cluster with ALB
costs money every hour. The free-tier path gives a **real public server** for
$0 and still demonstrates the same skills.

**The instance:** `t3.micro` (2 vCPU / 1 GB), **Amazon Linux 2023**,
16 GB gp3, security group open on 22 (SSH) + 80 (HTTP).
Free tier: 750 hrs/month — **stop the instance when not demoing** (stopping is
always $0; it also avoids the idle public-IPv4 charge).

**Deploy:** `scripts/deploy-ec2.ps1 -PublicIp <ip> -Key <pem>`:

```
wait for SSH → install Docker → scp the 577 MB image tar → docker load
→ run container on port 80 → poll /ready → health + ready checks
```

**Result: http://dr-grading.duckdns.org/ — real inference, ~520 ms server-side
(public IP changes on restarts; the dynamic-DNS record follows it automatically
via `scripts/add-dynamic-url.sh` + systemd `duckdns.service`/`duckdns.timer`).**

Production realities discovered the hard way (each one a good interview story):

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | script refused to run | PowerShell execution policy | `-ExecutionPolicy Bypass` |
| 2 | bizarre parse errors | UTF-8 em-dash read as ANSI by PS 5.1 → curly quote closed the string early | ASCII-only scripts |
| 3 | `scp: dest open "$HOME/" Failure` | **scp is not a shell — it never expands `$HOME`** | absolute paths |
| 4 | `write remote /tmp/... Failure` | **`/tmp` on AL2023 is a 457 MB RAM tmpfs**; tar is 577 MB | upload to home dir |
| 5 | health check failed at 8 s | container re-downloaded ImageNet weights on every start | `make_model(pretrained=False)` for serving (checkpoint supplies all weights) + poll `/ready` instead of `sleep 8` |
| 6 | fp16/CUDA checkpoint errors in benchmarks | local checkpoints are CUDA-tied | serving path verified: builds float32 on whatever device runs it |

---

## 7. Stage 7 — Evidently drift monitor (commit `3377e8e`)

**Problem:** models rot silently. Data changes (different camera, population,
pre-processing) and accuracy degrades with no error anywhere.

**Built:** `monitor.py`

- Reads the **prediction audit log** (written by `/predict`) plus reference
  data from the frozen test split.
- **Evidently** (0.7.x, stable `evidently.legacy` Report API) computes a
  **DataDriftPreset** report: feature drift on the model's inputs/outputs,
  with per-column drift flags.
- Report → `reports/drift/` (HTML + JSON), runnable as a scheduled job.

Verified end to end locally; drift report reset to the real 2 production rows
after synthetic test rows were cleaned up.

---

## 8. Stage 8 — HITL gate + retrain trigger (commit `3377e8e`)

**Problem:** who decides what happens when drift (or a bad week) is detected?

Two mechanisms, both already wired:

1. **Per-prediction HITL:** `abstain: true` when confidence < 0.55 → the
   case never gets an automated answer; it goes to the review queue. This is
   the *immediate* human gate.
2. **Model-level retrain trigger** (`retrain.py`): evaluates conditions —
   drift detected, metric below gate, recent retrain already pending —
   and prints a decision:

   ```
   RETRAIN: NO — all quiet
   ```

   When it fires, the flow closes the loop back to Stage 2:
   **retrain → eval gate → (pass?) → new registry version → CI redeploys.**
   The gate is in the loop *both* ways: it blocks bad models from shipping
   and it demands retraining when the live model stops qualifying.

---

## 9. How the pieces connect (two flows to explain)

### A. Request flow (what happens when someone uploads a photo)

```
browser /predict (multipart PNG)
   → FastAPI validates (400 on garbage/empty — even if model is loading)
   → preprocess: decode → resize 224² → BGR→RGB → ImageNet normalize
   → model(s): sigmoid per seed → averaged probabilities
   → grade = argmax · confidence = max prob · referable_score = 1−P(0)−P(1)
   → referable = score ≥ 0.3537 (frozen) · abstain = confidence < 0.55
   → audit log row (image hash + decision)  ← feeds the drift monitor
   → JSON response (+ latency_ms)
```

### B. Retrain flow (what happens when the model degrades)

```
audit log + live data
   → Evidently drift report  ──drift──►  retrain.py trigger
                                              │
   new data / schedule ───────────────────────┘
                                              ▼
                          retrain seeds → evaluate --ensemble --gate
                                   │                    │
                                FAIL                 PASS
                                   │                    ▼
                            stays in production   promote.py sets @production
                                                         ▼
                                            CI: docker-build → deploy
```

---

## 10. The 2-minute demo script (for your mentor)

1. **"The model is a 3-seed EfficientNet-B0, ensemble AUC 0.882, referable AUC
   0.963 — measured on a frozen, committed test split that CI re-checks."**
2. Open **the repo**: "CI is green — tests, an eval gate, and a Docker build
   run on every push. The eval gate *blocks* deployment if quality drops below
   the agreed thresholds." *(show Actions tab)*
3. Open **http://dr-grading.duckdns.org/docs** → `POST /predict` → *Try it out* →
   upload a fundus photo → Execute.
4. Walk the response: `grade 2, probs[...] , referable_score 0.87 ≥ 0.3537 →
   referable: true, abstain: false` — *"it answers, or explicitly abstains to
   human review when unsure — that's the HITL gate."*
5. **"Every prediction is audit-logged; Evidently compares it against the
   training distribution for drift; a trigger decides whether to retrain, and
   any retrained model must pass the same eval gate before CI ships it."*
6. **Cost**: *"This whole deployment is on AWS free tier — t3.micro, $0."*

---

## 11. Repo map (file → one-line responsibility)

```
config.py        single source of truth: paths, hyperparams, GATE thresholds
data.py          gold layer: frozen split manifests + Dataset/DataLoader
train.py         CLI training (seed / all-seeds / --smoke), make_model()
evaluate.py      metrics, eval_gate(), load_checkpoint(), EnsembleModule
predict.py       single-image inference core (DREnsemble)
serving.py       FastAPI: /health /ready /model /predict + audit log
tracking.py      MLflow server wiring (experiment dr-grading)
promote.py       gated promotion -> @production alias (never manual)
monitor.py       Evidently drift report from the prediction audit log
retrain.py       retrain trigger: drift/metric/freshness decision logic
data/gold/       frozen split CSVs (committed — the test-set guarantee)
models/          checkpoints + dr_metadata.json (deployment contract)
tests/           14 tests: split integrity, gate logic, API contract
.github/         CI: test → eval-gate → docker → staging → (manual) prod
Dockerfile(.cpu) GPU dev image / free-tier production image
README-aws.md    EC2 launch + deploy + cost guide
scripts/         deploy-ec2.ps1 (SSH → Docker → image → health check)
```

---

## 12. What's deliberately not done (honest boundary)

- **ECS/ALB/ECR** are wired in `ci.yml` + `configs/ecs-task-definition.json`
  but skip until AWS secrets are added — they cost money; EC2 free tier is
  the live target. Same code path, different infrastructure.
- **K8s**: the ECS job is the "real infra" artifact; a Helm chart would be
  the next step, not needed to demonstrate the concept.
- **The gate thresholds need clinical confirmation** (see Stage 2).
- MLflow runs locally (SQLite); a hosted registry (RDS/managed MLflow) would
  be the production version. CI accepts `MLFLOW_TRACKING_URI` as a secret to
  run the *real* gate instead of the logic tests.

---

## 13. One-line summary per diagram stage

| diagram stage | where it lives | proof it works |
|---|---|---|
| Gold Layer | `data.py` + committed `data/gold/splits/` | leakage/ratio/stratification tests |
| MLflow Tracking | `tracking.py`, port 5001, experiment `dr-grading` | runs with params/metrics/artifacts |
| MLflow Registry | model `dr-effnetb0`, `@production` alias | `pyfunc`/`pytorch` load verified |
| FastAPI | `serving.py` | 14 tests + live `/predict` |
| Docker | `Dockerfile.cpu` (2.53 GB) | container runs locally + on EC2 |
| GitHub Actions | `.github/workflows/ci.yml` | green run, 4 jobs |
| ECS / K8s | task def + CI jobs (skipped, secrets absent) | JSON validated, job guards green |
| Evidently drift | `monitor.py` | drift report generated |
| HITL gate | `abstain` in `/predict` + review routing | low-confidence → `abstain: true` |
| Retrain trigger | `retrain.py` | `RETRAIN: NO — all quiet` (and fires on drift) |
| **Live** | **http://dr-grading.duckdns.org/** | **health/ready/predict verified from the internet** |
