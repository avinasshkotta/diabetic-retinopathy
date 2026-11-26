"""Single-image inference — the core the FastAPI service will wrap.

    python -m dr_pipeline.predict path/to/fundus.png

Returns grade probabilities, the arg-max grade, and the clinical decision
(referable / not) at the frozen threshold.  Also exposes `abstain` so the
service can route low-confidence images to human review instead of answering.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from . import config as C
from .data import IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_checkpoint
from .train import DEVICE, make_model  # noqa: F401  (re-export for callers)


def preprocess(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if C.BEN_GRAHAM:
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10), -4, 128)
    img = cv2.resize(img, C.IMG_SIZE)
    x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1]))
    x = x.permute(2, 0, 1).float().div_(255.0)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)


class DREnsemble:
    """Loads the 3 seed checkpoints once; predicts many times."""

    def __init__(self, seeds: tuple[int, ...] = C.SEEDS,
                 confidence_abstain: float = 0.55):
        paths = [C.MODELS_DIR / f"dr_efficientnetb0_seed{s}.pt" for s in seeds]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing checkpoints: {missing}")
        self.models = [load_checkpoint(p) for p in paths]
        self.confidence_abstain = confidence_abstain

    @torch.no_grad()
    def predict_per_seed(self, x: torch.Tensor) -> list[np.ndarray]:
        """One probability vector per seed checkpoint (for agreement reporting)."""
        out = []
        for m in self.models:
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                out.append(torch.sigmoid(m(x.to(DEVICE))).float().cpu().numpy()[0])
        return out

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return np.mean(self.predict_per_seed(x), axis=0)

    def predict(self, path: str) -> dict:
        probs = self.predict_proba(preprocess(path))
        grade = int(probs.argmax())
        confidence = float(probs.max())
        referable_score = float(1 - probs[0] - probs[1])
        return {
            "grade": grade,
            "confidence": confidence,
            "probs": [round(float(p), 4) for p in probs],
            "referable": referable_score >= C.THRESHOLD_REFERABLE,
            "referable_score": round(referable_score, 4),
            "threshold": C.THRESHOLD_REFERABLE,
            # low confidence -> HITL queue instead of an answer
            "abstain": confidence < self.confidence_abstain,
        }


def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.predict")
    ap.add_argument("image")
    ap.add_argument("--single", action="store_true",
                    help="use seed[0] only (1x latency, ~0.005 AUC less)")
    args = ap.parse_args()

    models = tuple(C.SEEDS[:1]) if args.single else C.SEEDS
    result = DREnsemble(seeds=models).predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
