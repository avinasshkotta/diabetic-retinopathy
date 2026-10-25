"""MLflow wiring: one place that knows the tracking URI, experiment name and
registry model name.

Requires a tracking server with a SQL backend (the model registry does not
work on the plain file store):

    mlflow server --backend-store-uri sqlite:///mlflow.db \
                  --default-artifact-root mlartifacts \
                  --host 127.0.0.1 --port 5001

Override with MLFLOW_TRACKING_URI (CI/staging/prod all point here).
"""

from __future__ import annotations

import os

import mlflow

DEFAULT_URI = "http://127.0.0.1:5001"   # 5000 is reserved by Windows 1809+
EXPERIMENT = "dr-grading"
REGISTRY_MODEL_NAME = "dr-effnetb0"   # the registered model in the registry

# metrics a run must carry before it is allowed into the registry
GATE_METRICS = ("ensemble_auc", "referable_auc", "sensitivity", "specificity")


def configure() -> str:
    """Point MLflow at the server and select the experiment. Returns the URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT)
    return uri


def server_reachable() -> bool:
    try:
        mlflow.search_experiments(max_results=1)
        return True
    except Exception:
        return False
