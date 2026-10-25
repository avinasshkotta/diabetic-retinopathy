"""Prediction logger + Evidently drift monitor — Step 7.

    python -m dr_pipeline.monitor log-predictions   # (called by serving)
    python -m dr_pipeline.monitor drift-report       # daily job
    python -m dr_pipeline.monitor status            # latest drift verdict

Drift check: compare recent *reference* (gold train) image stats vs. the
last window of production predictions in data/gold/predictions.csv.
Raises alert -> HITL queue + retrain trigger (see monitor gate output).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from . import config as C

PRED_LOG = C.DATA_GOLD / "predictions.csv"   # production audit log
DRIFT_REPORT = C.REPORTS_DIR / "drift_report.json"

# thresholds (mentor diagram: Evidently -> HITL Gate)
DRIFT_MAX_SHARE = 0.30        # max shifted-feature share before alert
DRIFT_MIN_CONF_MEAN = 0.60    # alert if mean confidence drops below


def log_prediction(record: dict) -> None:
    """Append one prediction row (image hash only — no PHI images)."""
    C.DATA_GOLD.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    df = pd.DataFrame([row])
    header = not PRED_LOG.exists()
    df.to_csv(PRED_LOG, mode="a", header=header, index=False)


def load_predictions(days: int = 7) -> pd.DataFrame:
    if not PRED_LOG.exists():
        sys.exit(f"{PRED_LOG} not found — no production traffic logged yet")
    df = pd.read_csv(PRED_LOG, parse_dates=["ts"])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return df[df["ts"] >= cutoff]


def drift_report(days: int = 7) -> dict:
    """Evidently dataset-drift + confidence check on the recent window.

    Uses evidently.legacy (the stable Report API in 0.7.x; the new
    presets API does not yet expose a results payload).
    """
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.report import Report

    recent = load_predictions(days)
    if len(recent) < 50:
        sys.exit(f"only {len(recent)} predictions in window — need >= 50")

    # reference = gold train grade distribution; current = predicted grades
    ref = pd.DataFrame({"grade": load_split_train()["grade"]})
    cur = pd.DataFrame({"grade": recent["predicted_grade"]})

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    share = _share_drifted(report)
    conf_mean = float(recent["confidence"].mean())
    abstain_rate = float(recent["abstain"].mean())

    alert = (share > DRIFT_MAX_SHARE or
             conf_mean < DRIFT_MIN_CONF_MEAN or
             abstain_rate > 0.20)

    report_obj = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "n_predictions": int(len(recent)),
        "share_drifted": float(share),
        "mean_confidence": conf_mean,
        "abstain_rate": abstain_rate,
        "alert": bool(alert),
        "action": ("HITL_QUEUE + RETRAIN_TRIGGER" if alert
                   else "none — keep serving"),
    }
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT.write_text(json.dumps(report_obj, indent=2))
    return report_obj


def _share_drifted(report) -> float:
    """Pull share_of_drifted_columns out of the legacy Report payload,
    walking the dict (nesting has shifted across evidently versions)."""
    payload = report.as_dict()
    found: list[float] = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "share_of_drifted_columns" and isinstance(v, (int, float)):
                    found.append(float(v))
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    if not found:
        raise RuntimeError("could not locate share_of_drifted_columns in "
                           "Evidently report payload")
    return found[0]


def load_split_train() -> pd.DataFrame:
    from .data import load_split
    return load_split("train")


def status() -> None:
    if not DRIFT_REPORT.exists():
        print("no drift report yet — run `python -m dr_pipeline.monitor drift-report`")
        return
    r = json.loads(DRIFT_REPORT.read_text())
    flag = "ALERT" if r["alert"] else "OK"
    print(f"[{flag}] window={r['window_days']}d n={r['n_predictions']} "
          f"drift={r['share_drifted']:.2f} conf={r['mean_confidence']:.3f} "
          f"abstain={r['abstain_rate']:.2f} -> {r['action']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.monitor")
    ap.add_argument("cmd", choices=["drift-report", "status", "log-predictions"])
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    if args.cmd == "drift-report":
        r = drift_report(args.days)
        print(json.dumps(r, indent=2))
        # non-zero exit when alerting -> CI/cron notices and opens HITL work
        sys.exit(2 if r["alert"] else 0)
    elif args.cmd == "status":
        status()
    else:
        rec = json.loads(sys.stdin.read())      # piped from serving
        log_prediction(rec)
        print("logged")


if __name__ == "__main__":
    main()
