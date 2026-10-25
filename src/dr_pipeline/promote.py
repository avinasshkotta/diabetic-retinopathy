"""Model registry promotion — the only supported path to production.

    python -m dr_pipeline.promote --list
    python -m dr_pipeline.promote --run <run_id>     # evaluate + register + promote
    python -m dr_pipeline.promote --latest-eval      # promote newest eval run

Policy (enforced here, not by convention):
  1. The run must be an *evaluation* run carrying all gate metrics.
  2. eval_gate() must pass.
  3. Only then is a new model version registered and moved to the
     'production' alias; otherwise exit 1 and the version never exists.
  4. Previous production version keeps its own alias history — MLflow
     aliases are pointers, so rolling back = re-pointing the alias.

Requires the MLflow server (see tracking.py).
"""

from __future__ import annotations

import argparse
import sys

from . import config as C
from .evaluate import eval_gate
from .tracking import (EXPERIMENT, GATE_METRICS, REGISTRY_MODEL_NAME,
                        configure, server_reachable)


def _gate_metrics_of(run) -> dict:
    return {k: run.data.metrics[k] for k in GATE_METRICS
            if k in run.data.metrics}


def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.promote")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="MLflow run id to promote")
    g.add_argument("--latest-eval", action="store_true",
                   help="promote the most recent evaluation run")
    g.add_argument("--list", action="store_true",
                   help="list registered versions + aliases")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the policy without registering anything")
    args = ap.parse_args()

    configure()
    if not server_reachable():
        sys.exit(f"MLflow server unreachable — start it (see tracking.py)")

    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()

    # ---------------------------------------------------------------- list
    if args.list:
        try:
            versions = client.search_model_versions(
                f"name='{REGISTRY_MODEL_NAME}'")
        except Exception as e:  # registry empty / not created yet
            print(f"no model '{REGISTRY_MODEL_NAME}' in registry yet: {e}")
            return
        if not versions:
            print(f"'{REGISTRY_MODEL_NAME}': no versions registered")
            return
        print(f"'{REGISTRY_MODEL_NAME}':")
        for v in sorted(versions, key=lambda x: int(x.version)):
            # search_model_versions does not populate .aliases in MLflow 3
            full = client.get_model_version(REGISTRY_MODEL_NAME, v.version)
            aliases = ", ".join(full.aliases) or "-"
            run = client.get_run(v.run_id)
            gate = run.data.tags.get("gate", "?")
            auc = run.data.metrics.get("ensemble_auc", float("nan"))
            print(f"  v{v.version}  gate={gate}  auc={auc:.4f}  "
                  f"aliases=[{aliases}]")
        return

    # ------------------------------------------------------------- select
    if args.latest_eval:
        runs = mlflow.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT).experiment_id],
            filter_string="tags.kind = 'evaluation'",
            order_by=["attributes.start_time DESC"], max_results=1)
        if runs.empty:
            sys.exit("no evaluation runs found — run dr_pipeline.evaluate first")
        run_id = runs.iloc[0]["run_id"]
    else:
        run_id = args.run

    run = mlflow.get_run(run_id)
    kind = run.data.tags.get("kind")
    if kind != "evaluation":
        sys.exit(f"run {run_id} is tagged kind={kind!r}, not 'evaluation' — "
                 f"only evaluation runs may be promoted")

    missing = [k for k in GATE_METRICS if k not in run.data.metrics]
    if missing:
        sys.exit(f"run {run_id} lacks gate metrics {missing} — "
                 f"re-run dr_pipeline.evaluate against it")

    metrics = _gate_metrics_of(run)
    ok, failures = eval_gate(metrics)

    print(f"run {run_id}  ({run.data.tags.get('checkpoints', '?')})")
    for k, v in metrics.items():
        print(f"  {k:16s} {v:.4f}")
    if not ok:
        print("GATE: FAIL — promotion refused")
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)
    print("GATE: PASS")

    if args.dry_run:
        print("dry-run: nothing registered")
        return

    # ---------------------------------------------------------- register
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, REGISTRY_MODEL_NAME)
    version = result.version
    print(f"registered {REGISTRY_MODEL_NAME} v{version}")

    # policy tags on the version + move the 'production' alias
    client.set_model_version_tag(REGISTRY_MODEL_NAME, version,
                                 "gate", "PASS")
    client.set_model_version_tag(REGISTRY_MODEL_NAME, version,
                                 "source_run", run_id)
    client.set_registered_model_alias(REGISTRY_MODEL_NAME, "production",
                                      version)
    print(f"alias 'production' -> v{version}")

    # what the serving step will load
    print(f"serve with: models:/dr-effnetb0@production")


if __name__ == "__main__":
    main()
