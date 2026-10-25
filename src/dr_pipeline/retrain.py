"""Retrain trigger — closes the loop (mentor diagram: HITL -> Retrain).

    python -m dr_pipeline.retrain --check       # should we retrain?
    python -m dr_pipeline.retrain --run          # yes -> train + eval + promote

Triggers (any one fires):
  1. drift alert from monitor (drift_report.json.alert == true)
  2. >= N new HITL-reviewed labels in data/gold/reviewed.csv
  3. scheduled: last successful training run older than --max-age-days

--run executes the standard path: all-seeds train -> ensemble eval --gate
-> promote --latest-eval.  A gate FAIL stops the chain (no promotion), and
the previous production version keeps serving — that's the rollback story.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config as C

REVIEWED = C.DATA_GOLD / "reviewed.csv"      # HITL-reviewed labels
NEW_LABEL_MIN = 100                           # trigger 2 threshold
DRIFT_ALERT_FILE = C.REPORTS_DIR / "drift_report.json"


def _age_of_last_training() -> timedelta | None:
    try:
        from .tracking import configure, server_reachable
        import mlflow
        configure()
        if not server_reachable():
            return None
        exp = mlflow.get_experiment_by_name("dr-grading")
        if exp is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.kind = 'training' and attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"], max_results=1)
        if runs.empty:
            return None
        # start_time comes back as ISO string (not epoch ms)
        import pandas as pd
        started = pd.to_datetime(runs.iloc[0]["start_time"], utc=True)
        return datetime.now(timezone.utc) - started.to_pydatetime()
    except Exception:
        return None


def check(max_age_days: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    # trigger 1: drift alert
    if DRIFT_ALERT_FILE.exists():
        r = json.loads(DRIFT_ALERT_FILE.read_text())
        if r.get("alert"):
            reasons.append(f"drift alert ({r.get('action')})")

    # trigger 2: accumulated HITL-reviewed labels
    if REVIEWED.exists():
        import pandas as pd
        n = len(pd.read_csv(REVIEWED))
        if n >= NEW_LABEL_MIN:
            reasons.append(f"{n} reviewed labels (>= {NEW_LABEL_MIN})")

    # trigger 3: schedule
    age = _age_of_last_training()
    if age is None or age > timedelta(days=max_age_days):
        reasons.append(
            f"last training {'never ran' if age is None else f'{age.days}d ago'}"
            f" (> {max_age_days}d)")

    return bool(reasons), reasons


def run() -> None:
    """The full chain; each step must succeed before the next begins."""
    env = {"PYTHONPATH": "src", "PYTHONUTF8": "1", **dict(**__import__("os").environ)}
    steps = [
        ["py", "-m", "dr_pipeline.train", "--all-seeds"],
        ["py", "-m", "dr_pipeline.evaluate", "--ensemble", "--gate"],
        ["py", "-m", "dr_pipeline.promote", "--latest-eval"],
    ]
    for cmd in steps:
        print(f"==> {' '.join(cmd)}")
        r = subprocess.run(cmd, env=env, cwd=C.REPO_ROOT)
        if r.returncode != 0:
            sys.exit(f"chain stopped at: {' '.join(cmd)} "
                     f"(exit {r.returncode}) — production untouched")
    print("retrain chain complete: production alias updated")


def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.retrain")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--max-age-days", type=int, default=30)
    args = ap.parse_args()

    if args.check:
        fire, reasons = check(args.max_age_days)
        if fire:
            print("RETRAIN: YES")
            for r_ in reasons:
                print(f"  - {r_}")
            sys.exit(0)      # 0 = trigger fired (cron-friendly)
        print("RETRAIN: NO — all quiet")
        sys.exit(1)          # 1 = nothing to do
    run()


if __name__ == "__main__":
    main()
