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
from pydantic import BaseModel, Field

from . import config as C

app = FastAPI(
    title="Diabetic Retinopathy Grading API",
    version="1.0.0",
    description=(
        "Screens a retinal (fundus) photograph for diabetic retinopathy (DR).\n\n"
        "**How to use:** expand `POST /predict` → `Try it out` → `Choose file` "
        "(any fundus photo) → `Execute`.\n\n"
        "**Response fields:**\n"
        "- `grade` — predicted severity: 0 = no DR, 1 = mild, 2 = moderate, "
        "3 = severe, 4 = proliferative\n"
        "- `probs` — model probability for each grade [0..4], sums to 1\n"
        "- `confidence` — the highest of those probabilities (model certainty)\n"
        "- `referable_score` — P(grade >= 2): chance the case needs a specialist\n"
        "- `referable` — clinical decision: `referable_score >= threshold` "
        "→ refer to an ophthalmologist\n"
        "- `threshold` — frozen referral cutoff (0.3537), enforced by the eval gate\n"
        "- `abstain` — `true` when confidence < 0.55: the model refuses to answer "
        "and routes the case to human review (HITL)\n"
        "- `model` — which model/registry version answered\n"
        "- `latency_ms` — server-side inference time\n\n"
        "Screening aid only — not a diagnosis."
    ),
)

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
    """Wraps the local .pt seed checkpoints (dev fallback, no server needed).

    DR_SEEDS env var selects which seeds to load (comma-separated).
    Free-tier instances have 1 GB RAM -> default is single-seed there;
    locally/defaults follow config.SEEDS (full 3-model ensemble).
    """

    def __init__(self):
        import os
        from .predict import DREnsemble
        seeds_env = os.environ.get("DR_SEEDS", "")
        seeds = (tuple(int(s) for s in seeds_env.split(",")) if seeds_env
                 else C.SEEDS)
        self.inner = DREnsemble(seeds=seeds)

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
    """Liveness: the process is up. Never depends on the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Readiness: the model is loaded and can serve. 503 while loading or if loading failed."""
    if _state["model"] is None:
        raise HTTPException(503, detail=f"model not loaded: {_state['source']}")
    return {"ready": True, "model": _state["source"]}


@app.get("/model")
def model_info() -> dict:
    """Which model/source is serving and which referral threshold it uses."""
    return {"source": _state["source"], "loaded_at": _state["loaded_at"],
            "threshold": C.THRESHOLD_REFERABLE}


class PredictResponse(BaseModel):
    grade: int = Field(description="Predicted DR severity: 0=none, 1=mild, "
                                   "2=moderate, 3=severe, 4=proliferative")
    confidence: float = Field(description="Highest class probability — how "
                                          "certain the model is (0..1)")
    probs: list[float] = Field(description="Probability per grade [0,1,2,3,4], "
                                           "sums to 1")
    referable: bool = Field(description="Clinical decision: needs specialist "
                                        "referral (referable_score >= threshold)")
    referable_score: float = Field(description="P(grade >= 2) = 1 - P(0) - P(1); "
                                               "the chance of referable disease")
    threshold: float = Field(description="Frozen referral threshold (0.3537) "
                                         "enforced by the eval gate")
    abstain: bool = Field(description="true = confidence < 0.55, routed to human "
                                      "review (HITL) instead of an answer")
    model: str = Field(description="Which model / registry version answered")
    latency_ms: float = Field(description="Server-side inference time")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Grade a fundus photograph (multipart file upload) -> DR severity,
    referral decision, and abstain/HITL flag."""
    # validate the upload FIRST (cheap, model-independent) so clients get a
    # precise 4xx even while the model is still loading; 503 only for the
    # actual inference path.
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

    if _state["model"] is None:
        raise HTTPException(503, "model not loaded")

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
    abstain = confidence < 0.55

    # audit + drift log: image hash only, never the image itself
    try:
        import hashlib
        from .monitor import log_prediction
        log_prediction({
            "image_sha256": hashlib.sha256(raw).hexdigest(),
            "predicted_grade": grade,
            "confidence": round(confidence, 4),
            "referable_score": round(ref_score, 4),
            "abstain": bool(abstain),
        })
    except Exception:
        pass  # monitoring must never break inference

    return PredictResponse(
        grade=grade,
        confidence=round(confidence, 4),
        probs=[round(float(p), 4) for p in probs],
        referable=ref_score >= C.THRESHOLD_REFERABLE,
        referable_score=round(ref_score, 4),
        threshold=C.THRESHOLD_REFERABLE,
        # low confidence -> service asks for human review instead of answering
        abstain=abstain,
        model=_state["source"],
        latency_ms=round((time.time() - t0) * 1000, 1),
    )
