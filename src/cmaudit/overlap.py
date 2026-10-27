"""Screen CloudSEN12+ evaluation patches for overlap with OmniCloudMask v4 training data.

Per the OCM combined-training-dataset card (HF NickWright/OmniCloudMask-Combined-Training-Dataset),
v4 was trained on CloudSEN12 *high* labels from the train split only, but also on CloudSEN12
*scribble* labels and CloudSEN12 *2k* patches from all splits (train + val + test). An
evaluation patch can therefore share imagery or location with training data even though its
own label was never trained on.

Tiers (each eval patch gets exactly one):
- ``same_acquisition``: a training patch from the same Sentinel-2 acquisition overlaps it, so
  the model has seen (some of) these exact pixels.
- ``same_location``: a training patch overlaps it, but only from other dates.
- ``clean``: no training patch overlaps it at all.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from .meta import acquisition_key

TIERS = ("same_acquisition", "same_location", "clean")


def ocm_v4_training_mask(df: pd.DataFrame) -> pd.Series:
    """Rows of the CloudSEN12+ index that OCM v4 trained on (any processing level)."""
    high_train = (
        (df["label_type"] == "high")
        & (df["real_proj_shape"] == 509)
        & (df["tortilla:data_split"] == "train")
    )
    scribble_any = df["label_type"] == "scribble"
    big_any = df["real_proj_shape"] == 2000
    return high_train | scribble_any | big_any


def eval_mask(df: pd.DataFrame, split: str) -> pd.Series:
    return (
        (df["label_type"] == "high")
        & (df["real_proj_shape"] == 509)
        & (df["tortilla:data_split"] == split)
    )


def screen(
    evals: gpd.GeoDataFrame, train: gpd.GeoDataFrame, min_overlap_frac: float = 0.0
) -> pd.DataFrame:
    """Assign an overlap tier to every eval patch.

    ``min_overlap_frac`` is the fraction of the eval footprint that a training footprint must
    cover to count (0 = any positive-area intersection).
    """
    ev = evals[["tortilla:id", "s2_id", "geometry"]].copy()
    ev["acq"] = ev["s2_id"].map(acquisition_key)
    tr = train[["tortilla:id", "s2_id", "geometry"]].copy()
    tr["acq"] = tr["s2_id"].map(acquisition_key)

    pairs = gpd.sjoin(ev, tr, how="inner", predicate="intersects", lsuffix="ev", rsuffix="tr")
    if len(pairs):
        tr_geom = tr.geometry.loc[pairs["index_tr"]].to_numpy()
        ev_geom = ev.geometry.loc[pairs.index].to_numpy()
        inter = np.array([a.intersection(b).area for a, b in zip(ev_geom, tr_geom)])
        frac = inter / np.array([g.area for g in ev_geom])
        pairs = pairs.assign(frac=frac)
        pairs = pairs[pairs["frac"] > min_overlap_frac]
    pairs = pairs.assign(same_acq=pairs["acq_ev"] == pairs["acq_tr"]) if len(pairs) else pairs

    out = ev[["tortilla:id"]].copy()
    out["n_train_overlaps"] = 0
    out["max_overlap_frac"] = 0.0
    out["tier"] = "clean"
    if len(pairs):
        g = pairs.groupby(level=0)
        out.loc[g.size().index, "n_train_overlaps"] = g.size()
        out.loc[g["frac"].max().index, "max_overlap_frac"] = g["frac"].max()
        out.loc[g.size().index, "tier"] = "same_location"
        same = g["same_acq"].any()
        out.loc[same[same].index, "tier"] = "same_acquisition"
    return out.reset_index(drop=True)
