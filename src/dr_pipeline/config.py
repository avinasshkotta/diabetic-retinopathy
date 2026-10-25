"""Single source of truth for the DR pipeline.

Every hyperparameter that used to live in a notebook cell now lives here.
Training, evaluation and (later) the FastAPI service all import from this
module, so a value can never silently disagree between components.

The measured results of a training run are written to models/dr_metadata.json
(that file is an *output* / deployment contract — this file is the *input*).
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (repo-root relative; override data location with DR_DATA_RAW)
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = Path(os.environ.get("DR_DATA_RAW", REPO_ROOT / "data" / "raw"))
DATA_GOLD = REPO_ROOT / "data" / "gold"
SPLIT_DIR = DATA_GOLD / "splits"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

IMG_DIR = DATA_RAW / "train_images"

def _find_labels() -> Path:
    """The APTOS folder ships 'train-dataset.csv'; Kaggle's is 'train.csv'.
    Both are byte-identical — accept either."""
    for name in ("train.csv", "train-dataset.csv"):
        p = DATA_RAW / name
        if p.exists():
            return p
    return DATA_RAW / "train.csv"

TRAIN_CSV = _find_labels()
METADATA_FILE = MODELS_DIR / "dr_metadata.json"  # deployment contract

# --------------------------------------------------------------------------
# Model / training profile (matches the measured PyTorch GPU run)
# --------------------------------------------------------------------------
IMG_SIZE: tuple[int, int] = (224, 224)
BATCH = 32
EPOCHS_F1 = 8          # phase 1: frozen backbone
EPOCHS_F2 = 4          # phase 2: fine-tune top blocks
LR1, LR2 = 1e-3, 1e-5
WEIGHT_DECAY = 1e-4
PATIENCE_F1, PATIENCE_F2 = 3, 2
SEEDS: tuple[int, ...] = (42, 123, 777)

# --------------------------------------------------------------------------
# Clinical / task constants
# --------------------------------------------------------------------------
N_GRADES = 5
REFERABLE_MIN_GRADE = 2          # grades 2-4 = referable DR
THRESHOLD_REFERABLE = 0.3537     # Youden's J tuned on validation, frozen

# ImageNet normalisation (must match training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BEN_GRAHAM = False               # notebook default: off

# --------------------------------------------------------------------------
# Eval gate — CI blocks promotion when a run falls below these.
# Measured reference: ensemble AUC 0.8821, referable AUC ~0.93.
# Thresholds are set slightly under measured values to absorb seed noise
# but well above "model is broken" territory.  Change via PR, never locally.
# --------------------------------------------------------------------------
GATE_MIN_ENSEMBLE_AUC = 0.86
GATE_MIN_REFERABLE_AUC = 0.90
GATE_MIN_SENSITIVITY = 0.80      # at the frozen referral threshold
GATE_MIN_SPECIFICITY = 0.70
