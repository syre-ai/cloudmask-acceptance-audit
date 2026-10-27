"""Map the operational cloud masks shipped in CloudSEN12+ `extra` onto CloudSEN12 classes
(0 clear, 1 thick, 2 thin, 3 shadow). 255 = no decision (excluded from scoring).

Binary products (cloud vs not) map "cloud" to class 1; only the clear/not-clear decision
metrics (leak, discard, BOA) are meaningful for them, not per-class IoU.

Encodings were checked against the value sets observed in the downloaded test samples
(see scripts/inspect_baselines.py and results/baseline_value_sets.json).
"""

from __future__ import annotations

import numpy as np

NO_DECISION = 255


def qa60(v: np.ndarray) -> np.ndarray:
    """Sentinel-2 L1C QA60 bitmask: opaque cloud and cirrus bits. No shadow class.

    ESA documents bit 10 (opaque) and bit 11 (cirrus); CloudSEN12+ v1.1.2 stores them
    shifted x16 (observed values 0 / 16384 / 32768; 16384 overlaps 83% thick-cloud labels).
    Both encodings are accepted.
    """
    v = v.astype(np.uint32)
    out = np.zeros(v.shape, np.uint8)
    out[(v & ((1 << 11) | (1 << 15))) > 0] = 2
    out[(v & ((1 << 10) | (1 << 14))) > 0] = 1
    return out


def sen2cor(v: np.ndarray) -> np.ndarray:
    """Sentinel-2 L2A scene classification (SCL)."""
    out = np.zeros(v.shape, np.uint8)
    out[v == 3] = 3  # cloud shadow
    out[(v == 8) | (v == 9)] = 1  # cloud medium / high probability
    out[v == 10] = 2  # thin cirrus
    out[v == 0] = NO_DECISION  # no data
    return out


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y <= r * r).astype(np.float32)


def s2cloudless(v: np.ndarray, threshold: float = 0.4, average_over: int = 4,
                dilation_size: int = 2) -> np.ndarray:
    """s2cloudless cloud mask from its stored probability (0-100), reproducing the upstream
    ``CloudDetector.get_mask_from_prob`` defaults: average the probability over a disk of
    radius 4 px (reflect border), threshold with ``> 0.4``, dilate with a disk of radius 2 px.
    Applied on the 10 m grid of the patch. Detects cloud only (no shadow class)."""
    from scipy import ndimage

    p = v.astype(np.float32) / 100.0
    if average_over:
        k = _disk(average_over)
        p = ndimage.convolve(p, k / k.sum(), mode="reflect")
    m = p > threshold
    if dilation_size:
        m = ndimage.binary_dilation(m, structure=_disk(dilation_size).astype(bool))
    return m.astype(np.uint8)


def s2cloudless_raw(v: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Raw per-pixel probability thresholded at > 0.4, no averaging or dilation."""
    return (v.astype(np.float32) / 100.0 > threshold).astype(np.uint8)


def cloudscore(v: np.ndarray, clear_threshold: float = 0.60, scale: float = 1e4) -> np.ndarray:
    """Cloud Score+ (cs or cs_cdf): a clear-sky score in [0, 1]; Google's suggested default
    is to keep pixels with score >= 0.60. Stored as uint16 x ``scale``."""
    s = v.astype(np.float32) / scale
    out = np.where(s >= clear_threshold, 0, 1).astype(np.uint8)
    out[s > 1.0] = NO_DECISION  # ~1% of test pixels carry out-of-range values
    return out


def cloudsen12_schema(v: np.ndarray) -> np.ndarray:
    """Products already in CloudSEN12 classes (UnetMobV2 v1/v2)."""
    out = v.astype(np.uint8).copy()
    out[out > 3] = NO_DECISION
    return out


def sensei(v: np.ndarray) -> np.ndarray:
    """SEnSeI v2 as stored in CloudSEN12+ extra: 0 clear, 1 thick, 2 thin, 3 shadow
    (value-label overlap confirmed: 94% / 93% / 60% / 83%)."""
    return cloudsen12_schema(v)


BASELINES = {
    "QA60 (L1C bitmask)": ("cloudmask_qa60", qa60, False),
    "Sen2Cor SCL (L2A)": ("cloudmask_sen2cor", sen2cor, True),
    "s2cloudless (upstream defaults)": ("cloudmask_s2cloudless", s2cloudless, False),
    "s2cloudless (raw > 0.4)": ("cloudmask_s2cloudless", s2cloudless_raw, False),
    "Cloud Score+ cs": ("cloudmask_cloudscore_cs_v1", cloudscore, False),
    "Cloud Score+ cs_cdf": ("cloudmask_cloudscore_cs_cdf_v1", cloudscore, False),
    "UnetMobV2 v1": ("cloudmask_unetmobv2_v1", cloudsen12_schema, True),
    "UnetMobV2 v2": ("cloudmask_unetmobv2_v2", cloudsen12_schema, True),
    "SEnSeI v2": ("cloudmask_sensei_v2", sensei, True),
}
"""name -> (extra layer, decoder, has_classes).

Provenance caveat: UnetMobV2 v1/v2 were trained by the CloudSEN12 authors on CloudSEN12's own
training split, so like OCM they are in-domain on this test set; the operational products
(QA60, Sen2Cor, s2cloudless, Cloud Score+) were not built on CloudSEN12. QA60 and
s2cloudless have no shadow class, so their leak includes shadow by construction.
"""
