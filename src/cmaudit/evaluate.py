"""Per-image confusion matrices for OCM and the operational baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import BASELINES
from .metrics import confusion, cover
from .samples import load, valid_mask

OCM = "OmniCloudMask v4"
PRED_DIR = Path("data/preds/ocm")


def ocm_pred(level: str, tid: str) -> np.ndarray:
    with np.load(PRED_DIR / level / f"{tid}.npz") as z:
        return z["pred"]


def _row(tid, roi, method, level, label, pred, valid) -> dict:
    cm = confusion(label, pred, valid)
    return {
        "tortilla:id": tid, "roi_id": roi, "method": method, "level": level,
        "cm": cm.ravel(), "coverage": cm.sum() / max(int((valid & (label < 4)).sum()), 1),
        "true_cover": cover(label, valid),
        "pred_cover": cover(np.where(valid, pred, 255)),
    }


def per_image(man: pd.DataFrame, levels=("l1c", "l2a"), baselines: bool = True) -> pd.DataFrame:
    rows = []
    for tid, roi in tqdm(list(zip(man["tortilla:id"], man["roi_id"])), desc="score",
                         mininterval=10):
        lv = load("l1c", tid)
        valid = valid_mask(lv["bands"])
        label = lv["label"]
        for level in levels:
            v = valid if level == "l1c" else valid_mask(load(level, tid)["bands"])
            rows.append(_row(tid, roi, OCM, level, label, ocm_pred(level, tid), v))
        if baselines:
            ex = load("extra", tid)
            for name, (layer, decode, _) in BASELINES.items():
                rows.append(_row(tid, roi, name, "product", label, decode(ex[layer]), valid))
    return pd.DataFrame(rows)


def stack_cms(df: pd.DataFrame) -> np.ndarray:
    return np.stack(df["cm"].to_numpy()).reshape(-1, 4, 4)
