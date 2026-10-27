"""Leak/discard trade-off curves for masks that expose a continuous score.

Every method is reduced to a *clear score* per pixel (higher = more confidently clear), on an
integer grid; a pixel is accepted when its score >= t. Per image we keep a histogram
``H[true_class, score]``; summing histograms gives the exact curve at every threshold and keeps
the ROI-cluster bootstrap possible.

- OmniCloudMask: P(clear), stored as uint8 (0..255).
- s2cloudless: 100 - cloud probability (0..100).
- Cloud Score+ cs / cs_cdf: score x 10000, binned to 0..1000 (out-of-range values dropped).
"""

from __future__ import annotations

import numpy as np

N_CLASSES = 4


def score_hist(label: np.ndarray, score: np.ndarray, valid: np.ndarray, n_bins: int) -> np.ndarray:
    m = valid & (label < N_CLASSES) & (score >= 0) & (score < n_bins)
    idx = label[m].astype(np.int64) * n_bins + score[m].astype(np.int64)
    return np.bincount(idx, minlength=N_CLASSES * n_bins).reshape(N_CLASSES, n_bins)


def curve(h: np.ndarray) -> dict[str, np.ndarray]:
    """Leak and discard rates for every threshold t = 0..n_bins (accept score >= t).

    ``h``: (4, n_bins) summed histogram. Index t of each output array is threshold t.
    """
    # accepted[c, t] = pixels of true class c with score >= t
    accepted = np.concatenate([np.cumsum(h[:, ::-1], axis=1)[:, ::-1],
                               np.zeros((N_CLASSES, 1), h.dtype)], axis=1).astype(np.float64)
    acc_total = accepted.sum(0)
    clear_total = h[0].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        leak = (acc_total - accepted[0]) / acc_total
        out = {
            "threshold": np.arange(h.shape[1] + 1),
            "leak_rate": leak,
            "leak_thick": accepted[1] / acc_total,
            "leak_thin": accepted[2] / acc_total,
            "leak_shadow": accepted[3] / acc_total,
            "discard_rate": 1 - accepted[0] / clear_total,
            "accepted_share": acc_total / h.sum(),
        }
    return out


def pick_threshold(c: dict[str, np.ndarray], target: str, limit: float) -> int:
    """Threshold for an operating point chosen on validation data.

    target="leak": lowest-discard threshold whose leak_rate <= limit.
    target="discard": lowest-leak threshold whose discard_rate <= limit.
    """
    if target == "leak":
        ok = np.where(c["leak_rate"] <= limit)[0]
        return int(ok.min()) if len(ok) else int(c["threshold"][-1])
    if target == "discard":
        ok = np.where(c["discard_rate"] <= limit)[0]
        return int(ok.max()) if len(ok) else 0
    raise ValueError(target)
