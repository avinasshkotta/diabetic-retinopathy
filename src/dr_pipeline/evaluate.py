"""Evaluation + promotion gate (extracted from notebook cells 26/29).

    python -m dr_pipeline.evaluate --checkpoint models/dr_efficientnetb0_seed42.pt
    python -m dr_pipeline.evaluate --ensemble           # all 3 seeds
    python -m dr_pipeline.evaluate --ensemble --gate    # CI mode: exit 1 on fail

Outputs reports/evaluation.json with all metrics.  --gate turns the measured
numbers into a pass/fail decision against the thresholds in config.GATE_*,
which is what GitHub Actions will call before a model may be promoted.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve)

from . import config as C
from .data import load_split, make_loader
from .train import DEVICE, auc_of, predict, make_model


def load_checkpoint(path) -> torch.nn.Module:
    m = make_model()
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    m.eval()
    return m


class EnsembleModule(torch.nn.Module):
    """Deployable wrapper: normalised NCHW tensor -> grade probabilities.

    Averages the *sigmoid* outputs of the seed models (identical maths to
    evaluate()'s ensemble path).  Preprocessing (decode/resize/normalise)
    happens outside — in predict.preprocess / the FastAPI service.
    Served as the 'model' artifact that promote registers.
    """

    def __init__(self, models: list[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [torch.sigmoid(m(x)) for m in self.models]).mean(dim=0)


def evaluate(models: list[torch.nn.Module]) -> dict:
    """Full test-set evaluation of one model or an ensemble."""
    test_df = load_split("test")
    val_df = load_split("val")
    labels = test_df["grade"].to_numpy()
    test_loader = make_loader(test_df)

    per_model = [predict(m, test_loader) for m in models]
    aucs = [auc_of(p, labels) for p in per_model]
    probs = np.mean(per_model, axis=0)
    y_pred = probs.argmax(1)

    # referable-DR operating point (frozen threshold from the contract)
    ref_true = (labels >= C.REFERABLE_MIN_GRADE).astype(int)
    ref_score = 1 - probs[:, 0] - probs[:, 1]        # P(grade >= 2)
    pred = (ref_score >= C.THRESHOLD_REFERABLE).astype(int)
    sens = float((pred[ref_true == 1] == 1).mean())
    spec = float((pred[ref_true == 0] == 0).mean())

    # threshold is *tuned on validation*, never on test (notebook rule)
    val_labels = val_df["grade"].to_numpy()
    val_probs = np.mean([predict(m, make_loader(val_df)) for m in models], axis=0)
    val_ref = (val_labels >= C.REFERABLE_MIN_GRADE).astype(int)
    val_score = 1 - val_probs[:, 0] - val_probs[:, 1]
    fpr, tpr, ths = roc_curve(val_ref, val_score)
    bi = int(np.argmax(tpr - fpr))

    return {
        "n_test": int(len(test_df)),
        "single_model_aucs": [float(a) for a in aucs],
        "ensemble_auc": float(auc_of(probs, labels)),
        "referable_auc": float(roc_auc_score(ref_true, ref_score)),
        "threshold": C.THRESHOLD_REFERABLE,
        "sensitivity": sens,
        "specificity": spec,
        "val_tuned_threshold": float(ths[bi]),
        "val_youden_j": float(tpr[bi] - fpr[bi]),
        "classification_report": classification_report(
            labels, y_pred, output_dict=True, zero_division=0),
        "per_grade_auc": {
            str(g): float(roc_auc_score(np.eye(C.N_GRADES)[labels][:, g],
                                        probs[:, g]))
            for g in range(C.N_GRADES)
        },
    }


def eval_gate(m: dict) -> tuple[bool, list[str]]:
    """The promotion decision.  Called by CI --gate; also importable by tests."""
    checks = [
        ("ensemble_auc", m["ensemble_auc"], C.GATE_MIN_ENSEMBLE_AUC),
        ("referable_auc", m["referable_auc"], C.GATE_MIN_REFERABLE_AUC),
        ("sensitivity", m["sensitivity"], C.GATE_MIN_SENSITIVITY),
        ("specificity", m["specificity"], C.GATE_MIN_SPECIFICITY),
    ]
    failures = [f"{name} {val:.4f} < required {req:.4f}"
                for name, val, req in checks if val < req]
    return (not failures), failures


def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.evaluate")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="path to a single .pt file")
    g.add_argument("--ensemble", action="store_true",
                   help="average all 3 registered seed checkpoints")
    ap.add_argument("--gate", action="store_true",
                    help="exit 1 if eval_gate fails (CI mode)")
    ap.add_argument("--no-track", action="store_true",
                    help="skip MLflow logging (offline use only)")
    args = ap.parse_args()

    if args.ensemble:
        paths = [C.MODELS_DIR / f"dr_efficientnetb0_seed{s}.pt"
                 for s in C.SEEDS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            sys.exit(f"missing checkpoints: {missing}")
        models = [load_checkpoint(p) for p in paths]
    else:
        models = [load_checkpoint(args.checkpoint)]

    metrics = evaluate(models)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.REPORTS_DIR / "evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"ensemble AUC    : {metrics['ensemble_auc']:.4f}")
    print(f"referable AUC   : {metrics['referable_auc']:.4f}")
    print(f"sensitivity     : {metrics['sensitivity']:.3f} "
          f"(t={metrics['threshold']})")
    print(f"specificity     : {metrics['specificity']:.3f}")
    print(f"report          : {C.REPORTS_DIR / 'evaluation.json'}")

    ok, failures = eval_gate(metrics)

    # --- MLflow: record the evaluation + gate decision on the run(s) ----
    if not args.no_track:
        import mlflow
        from . import tracking
        tracking.configure()
        if not tracking.server_reachable():
            sys.exit(f"MLflow server unreachable at {tracking.DEFAULT_URI} — "
                     f"start it or pass --no-track")
        run_name = "ensemble-eval" if args.ensemble else "eval"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tag("kind", "evaluation")
            mlflow.set_tags({
                "checkpoints": "ensemble" if args.ensemble else args.checkpoint,
                "gate": "PASS" if ok else "FAIL",
                "gate_failures": "; ".join(failures),
                "n_models": len(models),
            })
            flat = {k: float(v) for k, v in metrics.items()
                    if isinstance(v, (int, float))}
            flat.update({f"auc_grade_{g}": v
                         for g, v in metrics["per_grade_auc"].items()})
            mlflow.log_metrics(flat)
            mlflow.log_artifact(str(C.REPORTS_DIR / "evaluation.json"))
            # raw checkpoints kept for traceability ...
            if args.ensemble:
                for s in C.SEEDS:
                    mlflow.log_artifact(
                        str(C.MODELS_DIR / f"dr_efficientnetb0_seed{s}.pt"),
                        artifact_path="checkpoints")
                # ... but the registry source is a *logged model* so that
                # mlflow.pyfunc/pytorch.load_model works on 'production'.
                import mlflow.pytorch
                import numpy as np
                module = EnsembleModule(models).cpu().eval()
                example = np.random.randn(1, 3, *C.IMG_SIZE).astype("float32")
                mlflow.pytorch.log_model(
                    module, name="model",
                    input_example=example,
                    # pt2 (torch.export) breaks CPU->CUDA loading; classic
                    # pickle serialization loads on any device via map_location
                    serialization_format="pickle",
                )
            print(f"mlflow run      : {run.info.run_id}")

    if args.gate:
        if ok:
            print("GATE: PASS — model may be promoted")
        else:
            print("GATE: FAIL")
            for f_ in failures:
                print(f"  - {f_}")
            sys.exit(1)


if __name__ == "__main__":
    main()
