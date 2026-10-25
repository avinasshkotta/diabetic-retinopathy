"""Training entry point (extracted from notebook cells 23/46).

    python -m dr_pipeline.train --seed 42
    python -m dr_pipeline.train --all-seeds          # the 3-seed ensemble
    python -m dr_pipeline.train --seed 42 --smoke    # 60-image CI sanity run

Saves state_dict to models/dr_efficientnetb0_seed<seed>.pt and a metrics
sidecar to reports/seed<seed>.json (metrics feed MLflow in Step 2).
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import models

from . import config as C
from .data import load_split, make_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"

if AMP:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_model() -> nn.Module:
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, C.N_GRADES)
    return m.to(DEVICE)


def set_phase(model: nn.Module, phase: int):
    """Phase 1: backbone frozen.  Phase 2: fine-tune last 3 blocks only."""
    for p in model.features.parameters():
        p.requires_grad_(phase == 2)
    if phase == 2:
        for blk in list(model.features.children())[:-3]:
            for p in blk.parameters():
                p.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def predict(model: nn.Module, loader) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
            out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)


def auc_of(probs: np.ndarray, labels: np.ndarray) -> float:
    return roc_auc_score(np.eye(C.N_GRADES)[labels], probs,
                         average="macro", multi_class="ovr")


def train_phase(model, loader, val_loader, epochs, lr, patience,
                val_labels, class_weights=None):
    params = set_phase(model, 1 if lr == C.LR1 else 2)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=C.WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
        va = auc_of(predict(model, val_loader), val_labels)
        print(f"    epoch {ep + 1}/{epochs}  loss {tot / len(loader):.4f}  "
              f"val AUC {va:.4f}")
        if va > best:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("    early stop")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_one_seed(seed: int, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   epochs_f1: int, epochs_f2: int, limit: int | None = None,
                   save: bool = True) -> tuple[nn.Module, dict]:
    set_seed(seed)
    val_loader = make_loader(val_df)
    val_labels = val_df["grade"].to_numpy()
    train_loader = make_loader(train_df, train_aug=True, limit=limit)

    model = make_model()
    print(f"  [seed {seed}] phase 1 - frozen backbone @ {C.LR1}")
    train_phase(model, train_loader, val_loader, epochs_f1, C.LR1,
                C.PATIENCE_F1, val_labels)
    print(f"  [seed {seed}] phase 2 - fine-tune top blocks @ {C.LR2}")
    train_phase(model, train_loader, val_loader, epochs_f2, C.LR2,
                C.PATIENCE_F2, val_labels)

    val_auc = auc_of(predict(model, val_loader), val_labels)
    metrics = {"seed": seed, "val_auc": float(val_auc),
               "epochs": [epochs_f1, epochs_f2],
               "n_train": int(len(train_df)), "limit": limit,
               "device": str(DEVICE)}

    if save:
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   C.MODELS_DIR / f"dr_efficientnetb0_seed{seed}.pt")
        with open(C.REPORTS_DIR / f"seed{seed}.json", "w") as f:
            json.dump(metrics, f, indent=2)
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.train")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int, help="train a single seed")
    g.add_argument("--all-seeds", action="store_true",
                   help=f"train the full ensemble {C.SEEDS}")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (60 imgs, 1+1 epochs) for CI plumbing tests")
    ap.add_argument("--no-save", action="store_true",
                    help="do not write weights/metrics (dry run)")
    args = ap.parse_args()

    if args.smoke:
        epochs_f1 = epochs_f2 = 1
        limit = 60
    else:
        epochs_f1, epochs_f2, limit = C.EPOCHS_F1, C.EPOCHS_F2, None

    seeds = list(C.SEEDS) if args.all_seeds else [args.seed]
    train_df, val_df = load_split("train"), load_split("val")

    print(f"device={DEVICE} amp={AMP} | train={len(train_df)} "
          f"val={len(val_df)} | seeds={seeds} | "
          f"epochs={epochs_f1}+{epochs_f2}{' SMOKE' if args.smoke else ''}")
    t0 = time.time()
    for s in seeds:
        train_one_seed(s, train_df, val_df, epochs_f1, epochs_f2,
                       limit=limit, save=not args.no_save)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
