"""Tests that run in seconds and gate CI.

Heavy tests (full training) live behind the smoke flag in train.py;
these are the fast structural guarantees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dr_pipeline import config as C  # noqa: E402
from dr_pipeline.data import load_split  # noqa: E402


# --------------------------------------------------------------------------
# Split integrity — the frozen-test-set guarantee
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def splits():
    if not (C.SPLIT_DIR / "test.csv").exists():
        pytest.skip("gold splits not generated yet")
    return {s: load_split(s) for s in ("train", "val", "test")}


def test_no_leakage(splits):
    ids = {s: set(df.id_code) for s, df in splits.items()}
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert not ids["train"] & ids["val"]


def test_split_sizes(splits):
    n = sum(len(df) for df in splits.values())
    assert abs(len(splits["train"]) / n - 0.60) < 0.01
    assert abs(len(splits["val"]) / n - 0.20) < 0.01
    assert abs(len(splits["test"]) / n - 0.20) < 0.01


def test_stratification_preserves_grades(splits):
    """Every split must contain all 5 grades or AUC is undefined."""
    for name, df in splits.items():
        assert sorted(df.grade.unique()) == [0, 1, 2, 3, 4], name


# --------------------------------------------------------------------------
# Eval gate logic (no model needed)
# --------------------------------------------------------------------------
def _fake_metrics(**over):
    m = {"ensemble_auc": 0.88, "referable_auc": 0.93,
         "sensitivity": 0.85, "specificity": 0.75}
    m.update(over)
    return m


def test_gate_passes_on_measured_values():
    from dr_pipeline.evaluate import eval_gate
    ok, failures = eval_gate(_fake_metrics())
    assert ok, failures


def test_gate_blocks_regression():
    from dr_pipeline.evaluate import eval_gate
    ok, failures = eval_gate(_fake_metrics(ensemble_auc=0.50))
    assert not ok
    assert any("ensemble_auc" in f for f in failures)


def test_gate_blocks_each_metric():
    from dr_pipeline.evaluate import eval_gate
    for key in ("ensemble_auc", "referable_auc", "sensitivity", "specificity"):
        ok, _ = eval_gate(_fake_metrics(**{key: 0.0}))
        assert not ok, f"gate must fail when {key} collapses"


# --------------------------------------------------------------------------
# Deployment contract consistency
# --------------------------------------------------------------------------
def test_metadata_contract_present():
    assert C.METADATA_FILE.exists(), "models/dr_metadata.json is the contract"
    import json
    meta = json.loads(C.METADATA_FILE.read_text())
    assert meta["framework"] == "pytorch"
    assert meta["seeds"] == list(C.SEEDS)
    assert abs(meta["threshold_referable"] - C.THRESHOLD_REFERABLE) < 1e-3
    assert tuple(meta["img_size"]) == C.IMG_SIZE


def test_checkpoints_exist_for_ensemble():
    # weights are gitignored by design — CI (fresh clone) has none; skip there
    if not all((C.MODELS_DIR / f"dr_efficientnetb0_seed{s}.pt").exists()
               for s in C.SEEDS):
        import pytest
        pytest.skip("checkpoints not present (training artifact, not in git)")
    for s in C.SEEDS:
        assert (C.MODELS_DIR / f"dr_efficientnetb0_seed{s}.pt").exists()
