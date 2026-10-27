"""Acceptance metrics for a cloud/shadow mask, framed around the decision a pipeline makes.

Classes follow CloudSEN12: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.
A pixel is *accepted* (used downstream) when the mask calls it clear.

Pixel policy ("mask then use the clear pixels"):
- ``leak_rate``: share of accepted pixels that are actually cloud or shadow
  (1 - user's accuracy of clear). Broken down by what leaked (thick/thin/shadow).
- ``discard_rate``: share of truly clear pixels that the mask throws away
  (1 - producer's accuracy of clear).

Scene policy ("keep the scene if its cloud cover is below X%"):
- ``false_accept``: kept scenes whose true cover is above X.
- ``false_reject``: dropped scenes whose true cover is at or below X.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("clear", "thick", "thin", "shadow")
N_CLASSES = 4


def confusion(label: np.ndarray, pred: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """4x4 pixel confusion matrix, rows = truth, cols = prediction."""
    m = (label < N_CLASSES) & (pred < N_CLASSES)
    if valid is not None:
        m &= valid
    idx = label[m].astype(np.int64) * N_CLASSES + pred[m].astype(np.int64)
    return np.bincount(idx, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)


def pixel_summary(cm: np.ndarray) -> dict:
    """Decision-framed metrics from a (summed) confusion matrix."""
    cm = cm.astype(np.float64)
    total = cm.sum()
    accepted = cm[:, 0].sum()
    true_clear = cm[0].sum()
    out = {
        "pixels": total,
        "leak_rate": (accepted - cm[0, 0]) / accepted if accepted else np.nan,
        "leak_thick": cm[1, 0] / accepted if accepted else np.nan,
        "leak_thin": cm[2, 0] / accepted if accepted else np.nan,
        "leak_shadow": cm[3, 0] / accepted if accepted else np.nan,
        "discard_rate": (true_clear - cm[0, 0]) / true_clear if true_clear else np.nan,
        "accepted_share": accepted / total if total else np.nan,
        "true_clear_share": true_clear / total if total else np.nan,
    }
    # Binary "cloud or shadow" vs clear: balanced overall accuracy, the CloudSEN12 headline.
    tp = cm[1:, 1:].sum()
    fn = cm[1:, 0].sum()
    tn = cm[0, 0]
    fp = cm[0, 1:].sum()
    tpr = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    out["boa"] = 0.5 * (tpr + tnr)
    for k, name in enumerate(CLASSES):
        inter = cm[k, k]
        union = cm[k].sum() + cm[:, k].sum() - inter
        out[f"iou_{name}"] = inter / union if union else np.nan
    return out


def cover(label_or_pred: np.ndarray, valid: np.ndarray | None = None) -> float:
    """Fraction of valid pixels that are cloud or shadow (classes 1-3)."""
    m = label_or_pred < N_CLASSES
    if valid is not None:
        m &= valid
    n = m.sum()
    return float((label_or_pred[m] > 0).sum() / n) if n else np.nan


def scene_policy(true_cover: np.ndarray, pred_cover: np.ndarray, threshold: float) -> dict:
    """Error rates of 'accept scene if predicted cover <= threshold'."""
    true_cover = np.asarray(true_cover)
    pred_cover = np.asarray(pred_cover)
    accept = pred_cover <= threshold
    truly_ok = true_cover <= threshold
    n_acc = accept.sum()
    return {
        "threshold": threshold,
        "accepted": int(n_acc),
        "false_accept_rate": float((accept & ~truly_ok).sum() / n_acc) if n_acc else np.nan,
        "false_reject_rate": float((~accept & truly_ok).sum() / truly_ok.sum())
        if truly_ok.sum()
        else np.nan,
        "cover_mae": float(np.mean(np.abs(pred_cover - true_cover))),
        "cover_bias": float(np.mean(pred_cover - true_cover)),
    }


def cluster_bootstrap(
    cms: np.ndarray,
    clusters: np.ndarray,
    stat: str,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of a pixel_summary statistic, resampling clusters
    (CloudSEN12 ROIs: 5 dates of one location are not independent).

    ``cms``: (n_images, 4, 4) per-image confusion matrices; ``clusters``: (n_images,) ids.
    """
    point = pixel_summary(cms.sum(0))[stat]
    codes, inv = np.unique(clusters, return_inverse=True)
    per_cluster = np.zeros((len(codes), N_CLASSES, N_CLASSES))
    np.add.at(per_cluster, inv, cms)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(codes), len(codes))
        draws[b] = pixel_summary(per_cluster[pick].sum(0))[stat]
    a = (1 - ci) / 2
    lo, hi = np.nanquantile(draws, [a, 1 - a])
    return float(point), float(lo), float(hi)


def summarize(cms: np.ndarray, groups: pd.Series, clusters: np.ndarray, stats: list[str],
              n_boot: int = 1000) -> pd.DataFrame:
    """Per-group table of statistics with cluster-bootstrap CIs."""
    rows = []
    for name, ix in groups.groupby(groups).groups.items():
        ix = np.asarray(list(ix))
        row = {"group": name, "images": len(ix), "rois": len(np.unique(clusters[ix]))}
        for s in stats:
            p, lo, hi = cluster_bootstrap(cms[ix], clusters[ix], s, n_boot=n_boot)
            row[s], row[f"{s}_lo"], row[f"{s}_hi"] = p, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)
