"""Per-image confusion matrices for OCM and the operational baselines.

All methods and both processing levels are scored on one common support per image (real
509x509 window where OCM's R/G/NIR inputs are present at both L1C and L2A), so every row
compares the same pixels.

A method may abstain on some pixels (NO_DECISION). Two policies are recorded:
- ``cm`` (operational, the headline): an abstention is treated as "not accepted", i.e. the
  pixel is discarded. This is what a pipeline that only uses pixels flagged clear does.
- ``cm_cond`` (conditional): abstentions are dropped and only decided pixels are scored.
``coverage`` is the share of support pixels the method decided.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import BASELINES, NO_DECISION
from .metrics import confusion, cover
from .samples import load, support

OCM = "OmniCloudMask v4"
PRED_DIR = Path("data/preds/ocm")


def ocm_pred(level: str, tid: str) -> np.ndarray:
    with np.load(PRED_DIR / level / f"{tid}.npz") as z:
        return z["pred"]


def operational(pred: np.ndarray) -> np.ndarray:
    """Abstentions become 'not clear' (class 1) so they are discarded, never leaked."""
    return np.where(pred == NO_DECISION, 1, pred).astype(np.uint8)


def score_row(label: np.ndarray, pred: np.ndarray, sup: np.ndarray) -> dict:
    n_sup = int(sup.sum())
    op = operational(pred)
    cm_cond = confusion(label, pred, sup)
    return {
        "cm": confusion(label, op, sup).ravel(),
        "cm_cond": cm_cond.ravel(),
        "support_px": n_sup,
        "coverage": cm_cond.sum() / n_sup if n_sup else np.nan,
        "true_cover": cover(label, sup),
        "pred_cover": cover(op, sup),
    }


def per_image(man: pd.DataFrame, levels=("l1c", "l2a"), baselines: bool = True) -> pd.DataFrame:
    rows = []
    for tid, roi in tqdm(list(zip(man["tortilla:id"], man["roi_id"])), desc="score",
                         mininterval=10):
        label = load("l1c", tid)["label"]
        sup = support(tid)
        base = {"tortilla:id": tid, "roi_id": roi}
        for level in levels:
            rows.append({**base, "method": OCM, "level": level,
                         **score_row(label, ocm_pred(level, tid), sup)})
        if baselines:
            ex = load("extra", tid)
            for name, (layer, decode, _) in BASELINES.items():
                rows.append({**base, "method": name, "level": "product",
                             **score_row(label, decode(ex[layer]), sup)})
    return pd.DataFrame(rows)


def stack_cms(df: pd.DataFrame, col: str = "cm") -> np.ndarray:
    return np.stack(df[col].to_numpy()).reshape(-1, 4, 4)
