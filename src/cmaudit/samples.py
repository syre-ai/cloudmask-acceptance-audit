"""Access to downloaded CloudSEN12+ samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
REAL = 509  # CloudSEN12+ pads 509 -> 512 on the bottom and right; the label pad reads "clear"


def manifest(split: str | None = None, tiers=("clean",),
             exclude_kappaset_tiles: bool = False) -> pd.DataFrame:
    """Evaluation images. ``tier`` combines the CloudSEN12 footprint screen with the OCM
    custom-scene location screen (data/extra_overlap_flags.parquet, from
    scripts/screen_ocm_training.py)."""
    m = pd.read_parquet(DATA / "manifest.parquet")
    flags_path = DATA / "extra_overlap_flags.parquet"
    if flags_path.exists():
        m = m.merge(pd.read_parquet(flags_path), on="tortilla:id", how="left")
        m.loc[m["ocm_custom_near"].fillna(False) & (m["tier"] == "clean"), "tier"] = "same_location"
        if exclude_kappaset_tiles:
            m = m[~m["kappaset_tile"].fillna(False)]
    if split is not None:
        m = m[m["tortilla:data_split"] == split]
    if tiers is not None:
        m = m[m["tier"].isin(tiers)]
    return m.reset_index(drop=True)


def load(level: str, tid: str) -> dict:
    with np.load(DATA / "samples" / level / f"{tid}.npz") as z:
        return {k: z[k] for k in z.files}


# Stored bands are B02, B03, B04, B8A; OCM sees only B04, B03, B8A.
OCM_INPUTS = [1, 2, 3]


def valid_mask(bands: np.ndarray) -> np.ndarray:
    """Real (unpadded) window and at least one of OCM's inputs (R, G, NIR) nonzero, which is
    exactly where OCM makes a decision (it treats a pixel as nodata only if all inputs are 0)."""
    v = np.zeros(bands.shape[1:], dtype=bool)
    v[:REAL, :REAL] = True
    return v & (bands[OCM_INPUTS] != 0).any(0)


def support(tid: str) -> np.ndarray:
    """Common scoring support for every method and level: valid at both L1C and L2A."""
    return valid_mask(load("l1c", tid)["bands"]) & valid_mask(load("l2a", tid)["bands"])
