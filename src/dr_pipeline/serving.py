"""FastAPI inference service — Step 3 of the MLOps pipeline.

    uvicorn dr_pipeline.serving:app --port 8080

Endpoints:
    GET  /health   -> liveness (process is up)
    GET  /ready    -> readiness (model loaded)
    GET  /model    -> which registry version is serving
    POST /predict  -> grade + referable flag + abstain (HITL routing)

Model source (in order of precedence):
    MODEL_URI env var (e.g. models:/dr-effnetb0@production)
    -> falls back to local models/*.pt checkpoints (dev / no MLflow server)
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from . import config as C

app = FastAPI(title="dr-grading", version="1.0.0")

# --------------------------------------------------------------------------
# model loading (once, at startup)
# --------------------------------------------------------------------------
_state = {"model": None, "source": None, "loaded_at": None}


class RegistryEnsemble:
    """Wraps the MLflow-logged EnsembleModule (normalised tensor -> probs)."""

    def __init__(self, model):
        self.model = model

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(x).float()
            if torch.cuda.is_available():
                t = t.cuda()
                self.model = self.model.cuda()
            out = self.model(t)
        return out.cpu().numpy()


class LocalEnsemble:
    """Wraps the local .pt seed checkpoints (dev fallback, no server needed)."""

    def __init__(self):
        from .predict import DREnsemble
        self.inner = DREnsemble()

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        import torch
        # (N,5): batch of normalised tensors -> ensemble probabilities
        return np.stack(
            [self.inner.predict_proba(torch.from_numpy(xi[None]))
             for xi in x])


def _load_model() -> None:
    uri = os.environ.get("MODEL_URI")
    if uri:
        import mlflow.pyfunc
        _state["model"] = RegistryEnsemble(mlflow.pyfunc.load_model(uri))
        _state["source"] = uri
    else:
        _state["model"] = LocalEnsemble()
        _state["source"] = "local checkpoints (set MODEL_URI for registry)"


@app.on_event("startup")
def startup() -> None:
    try:
        _load_model()
        _state["loaded_at"] = time.time()
    except Exception as e:  # /ready will report, /health still responds
        _state["model"] = None
        _state["source"] = f"FAILED: {e}"


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if _state["model"] is None:
        raise HTTPException(503, detail=f"model not loaded: {_state['source']}")
    return {"ready": True, "model": _state["source"]}


@app.get("/model")
def model_info() -> dict:
    return {"source": _state["source"], "loaded_at": _state["loaded_at"],
            "threshold": C.THRESHOLD_REFERABLE}


class PredictResponse(BaseModel):
    grade: int
    confidence: float
    probs: list[float]
    referable: bool
    referable_score: float
    threshold: float
    abstain: bool
    model: str
    latency_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if _state["model"] is None:
        raise HTTPException(503, "model not loaded")

    t0 = time.time()
    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "image too large (>20MB)")
    if not raw:
        raise HTTPException(400, "empty upload")

    # decode -> model input (same preprocessing as training)
    import cv2
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(400, "not a decodable image")
    img = cv2.resize(img, C.IMG_SIZE)
    if img.ndim != 3 or img.shape[2] != 3:
        raise HTTPException(400, "not a color image")
    x = np.ascontiguousarray(img[:, :, ::-1]).transpose(2, 0, 1)[None]
    x = x.astype(np.float32) / 255.0
    mean = np.array(C.IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
    std = np.array(C.IMAGENET_STD, dtype=np.float32)[None, :, None, None]
    x = (x - mean) / std

    try:
        probs = _state["model"].predict_probs(x)[0]
    except Exception as e:
        raise HTTPException(500, f"inference failed: {e}")

    grade = int(probs.argmax())
    confidence = float(probs.max())
    ref_score = float(1 - probs[0] - probs[1])

    return PredictResponse(
        grade=grade,
        confidence=round(confidence, 4),
        probs=[round(float(p), 4) for p in probs],
        referable=ref_score >= C.THRESHOLD_REFERABLE,
        referable_score=round(ref_score, 4),
        threshold=C.THRESHOLD_REFERABLE,
        # low confidence -> service asks for human review instead of answering
        abstain=confidence < 0.55,
        model=_state["source"],
        latency_ms=round((time.time() - t0) * 1000, 1),
    )
