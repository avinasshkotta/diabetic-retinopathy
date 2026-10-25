"""Data layer: raw -> gold.

Extracted from notebook cells 5/11/17 with one structural change:

The notebook re-ran ``train_test_split`` every time it executed.  Deterministic
seed or not, the *frozen test set* guarantee only exists if the split is
materialised once.  Here ``make-split`` writes the three split manifests to
data/gold/splits/*.csv, and every later command only *reads* them.  If the
manifests exist they are authoritative; re-running make-split refuses to
overwrite them unless --force is passed.

Run once:
    python -m dr_pipeline.data make-split
    python -m dr_pipeline.data inspect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from . import config as C

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
# Raw layer
# --------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Read train.csv, attach absolute image paths, validate."""
    if not C.TRAIN_CSV.exists():
        sys.exit(f"train.csv not found at {C.TRAIN_CSV} "
                 f"(set DR_DATA_RAW or create the data/raw junction)")
    df = pd.read_csv(C.TRAIN_CSV).rename(columns={"diagnosis": "grade"})
    df["path"] = df["id_code"].map(lambda x: str(C.IMG_DIR / f"{x}.png"))
    missing = (~df["path"].map(lambda p: Path(p).exists())).sum()
    if missing:
        sys.exit(f"{missing} images missing under {C.IMG_DIR}")
    return df


# --------------------------------------------------------------------------
# Gold layer: frozen split manifests
# --------------------------------------------------------------------------
def make_split(df: pd.DataFrame, force: bool = False) -> None:
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        if (C.SPLIT_DIR / f"{s}.csv").exists() and not force:
            sys.exit(f"{C.SPLIT_DIR / f'{s}.csv'} already exists — split is "
                     f"frozen.  Pass --force only if you intend to invalidate "
                     f"every metric ever measured against it.")

    pool, test = train_test_split(df, test_size=0.20, stratify=df["grade"],
                                  random_state=42)
    train, val = train_test_split(pool, test_size=0.25,
                                  stratify=pool["grade"], random_state=42)

    # same integrity checks the notebook had
    n = len(df)
    assert abs(len(train) / n - 0.60) < 0.01
    assert abs(len(val) / n - 0.20) < 0.01
    assert abs(len(test) / n - 0.20) < 0.01
    assert not set(train.id_code) & set(test.id_code), "train/test leakage"
    assert not set(val.id_code) & set(test.id_code), "val/test leakage"

    for name, part in zip(SPLITS, (train, val, test)):
        part[["id_code", "grade"]].to_csv(C.SPLIT_DIR / f"{name}.csv",
                                          index=False)
    print(f"gold split written to {C.SPLIT_DIR}: "
          f"train {len(train)} / val {len(val)} / test {len(test)}  [FROZEN]")


def load_split(name: str) -> pd.DataFrame:
    """Read a frozen split manifest and re-attach image paths."""
    if name not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    f = C.SPLIT_DIR / f"{name}.csv"
    if not f.exists():
        sys.exit(f"{f} missing — run `python -m dr_pipeline.data make-split` first")
    df = pd.read_csv(f)
    df["path"] = df["id_code"].map(lambda x: str(C.IMG_DIR / f"{x}.png"))
    missing = (~df["path"].map(lambda p: Path(p).exists())).sum()
    if missing:
        sys.exit(f"{missing} images for split '{name}' not found")
    return df


def inspect() -> None:
    for s in SPLITS:
        df = load_split(s)
        counts = df["grade"].value_counts().sort_index().tolist()
        print(f"{s:6s} {len(df):5d} ({len(df) / 3662:.0%})  grades {counts}")


# --------------------------------------------------------------------------
# Dataset / loader (verbatim behaviour from notebook cell 17)
# --------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
IMAGENET_STD = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)


class DRDataset(Dataset):
    """uint8 RGB images -> normalised float CHW tensors + one-hot labels."""

    def __init__(self, frame: pd.DataFrame, train_aug: bool = False):
        import cv2  # heavy import, keep module load light

        imgs = []
        for p in frame["path"]:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            if C.BEN_GRAHAM:
                img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 10),
                                      -4, 128)
            img = cv2.resize(img, C.IMG_SIZE)
            imgs.append(np.ascontiguousarray(img[:, :, ::-1]))  # BGR -> RGB
        self.images = torch.from_numpy(np.stack(imgs))          # N,H,W,3 uint8
        self.labels = torch.tensor(frame["grade"].to_numpy(), dtype=torch.long)
        self.train_aug = train_aug

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        from torchvision.transforms import v2
        import random

        x = self.images[i].permute(2, 0, 1).float().div_(255.0)
        if self.train_aug:
            if random.random() < 0.5:
                x = torch.flip(x, [2])
            if random.random() < 0.5:
                x = torch.flip(x, [1])
            if random.random() < 0.5:
                x = v2.functional.affine(
                    x, angle=random.uniform(-8, 8),
                    translate=[int(random.uniform(-0.05, 0.05 * C.IMG_SIZE[0])),
                               int(random.uniform(-0.05, 0.05 * C.IMG_SIZE[1]))],
                    shear=[0.0], scale=random.uniform(0.95, 1.05),
                    interpolation=v2.InterpolationMode.BILINEAR)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, torch.nn.functional.one_hot(self.labels[i], C.N_GRADES).float()


def make_loader(frame: pd.DataFrame, train_aug: bool = False,
                shuffle: bool | None = None, batch: int | None = None,
                limit: int | None = None) -> DataLoader:
    if limit:
        frame = frame.iloc[:limit]
    ds = DRDataset(frame, train_aug=train_aug)
    return DataLoader(ds, batch_size=batch or C.BATCH,
                      shuffle=train_aug if shuffle is None else shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="dr_pipeline.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ms = sub.add_parser("make-split", help="freeze the gold split manifests")
    ms.add_argument("--force", action="store_true")
    sub.add_parser("inspect", help="show split sizes and grade balance")
    args = ap.parse_args()

    if args.cmd == "make-split":
        make_split(load_raw(), force=args.force)
    else:
        inspect()


if __name__ == "__main__":
    main()
