import numpy as np

from cmaudit.metrics import confusion, pixel_summary
from cmaudit.sweep import curve, pick_threshold, score_hist


def test_curve_matches_direct_computation():
    rng = np.random.default_rng(0)
    label = rng.integers(0, 4, 5000)
    # clear pixels get higher scores on average
    score = np.clip(rng.normal(np.where(label == 0, 180, 80), 50), 0, 255).astype(np.int64)
    valid = np.ones_like(label, bool)
    c = curve(score_hist(label, score, valid, 256))
    for t in (0, 50, 128, 200, 256):
        pred = np.where(score >= t, 0, 1)
        s = pixel_summary(confusion(label, pred))
        if t < 256:
            assert np.isclose(c["leak_rate"][t], s["leak_rate"])
            assert np.isclose(c["leak_thin"][t], s["leak_thin"])
        assert np.isclose(c["discard_rate"][t], s["discard_rate"])
    # raising the threshold can only discard more clear pixels
    assert np.all(np.diff(c["discard_rate"]) >= -1e-12)


def test_pick_threshold():
    c = {
        "threshold": np.arange(5),
        "leak_rate": np.array([0.3, 0.2, 0.1, 0.05, np.nan]),
        "discard_rate": np.array([0.0, 0.01, 0.05, 0.2, 1.0]),
    }
    assert pick_threshold(c, "leak", 0.1) == 2
    assert pick_threshold(c, "discard", 0.02) == 1
    assert pick_threshold(c, "leak", 0.01) == 4  # unreachable -> reject everything
