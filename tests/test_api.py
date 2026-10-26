"""API contract tests (fast — use local checkpoints, no MLflow needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from dr_pipeline import serving
    with TestClient(serving.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_model(client):
    r = client.get("/ready")
    if r.status_code == 503:
        # CI has no weights (gitignored) — readiness must still report honestly
        pytest.skip(f"model not loaded here: {r.json().get('detail')}")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_model_info_has_threshold(client):
    body = client.get("/model").json()
    assert body["threshold"] == pytest.approx(0.3537, abs=1e-3)


def _fundus_bytes() -> bytes:
    """A real APTOS image for the integration path."""
    import glob
    hits = glob.glob(str(
        Path(__file__).resolve().parents[1] / "data" / "raw"
        / "train_images" / "*.png"))
    if not hits:
        pytest.skip("dataset not linked")
    return Path(hits[0]).read_bytes()


def test_predict_real_image(client):
    from dr_pipeline import config as C
    r = client.post("/predict", files={"file": ("x.png", _fundus_bytes(),
                                                 "image/png")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert len(b["probs"]) == C.N_GRADES
    assert 0 <= b["grade"] <= 4
    assert isinstance(b["referable"], bool)
    assert isinstance(b["abstain"], bool)
    assert b["latency_ms"] > 0


def test_predict_rejects_garbage(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image",
                                                 "image/png")})
    assert r.status_code == 400


def test_predict_rejects_empty(client):
    r = client.post("/predict", files={"file": ("x.png", b"",
                                                 "image/png")})
    assert r.status_code == 400
