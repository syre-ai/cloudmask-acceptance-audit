"""Access to downloaded CloudSEN12+ samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
REAL = 509  # CloudSEN12+ pads 509 -> 512 on the bottom and right; the label pad reads "clear"


def manifest(split: str | None = None, tiers=("clean",)) -> pd.DataFrame:
    m = pd.read_parquet(DATA / "manifest.parquet")
    if split is not None:
        m = m[m["tortilla:data_split"] == split]
    if tiers is not None:
        m = m[m["tier"].isin(tiers)]
    return m.reset_index(drop=True)


def load(level: str, tid: str) -> dict:
    with np.load(DATA / "samples" / level / f"{tid}.npz") as z:
        return {k: z[k] for k in z.files}


def valid_mask(bands: np.ndarray) -> np.ndarray:
    """Real (unpadded) window and at least one nonzero band."""
    v = np.zeros(bands.shape[1:], dtype=bool)
    v[:REAL, :REAL] = True
    return v & (bands != 0).any(0)
