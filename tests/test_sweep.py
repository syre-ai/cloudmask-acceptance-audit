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


def test_abstentions_never_accepted():
    from cmaudit.sweep import native_curve

    label = np.array([0, 0, 1, 0])
    shifted = np.array([1, 3, 2, 0])  # native scores 0, 2, 1 and one abstention (clear pixel)
    h = score_hist(label, shifted, np.ones(4, bool), 4)
    c = native_curve(h)
    # native threshold 0 accepts every decided pixel but not the abstention
    assert np.isclose(c["discard_rate"][0], 1 / 3) and np.isclose(c["leak_rate"][0], 1 / 3)
    assert np.isclose(c["discard_rate"][3], 1.0)  # nothing accepted


def test_executable_rules_match_scored_predicate():
    import sys

    sys.path.insert(0, "scripts")
    from sweep import native_rule

    # OCM: stored q = rint(float32(P) * 255); scored q >= t. The rule names that predicate.
    thr, rule = native_rule("OmniCloudMask v4 | l1c", 218)
    assert "rint(255 x P(clear)) >= 218" in rule and abs(thr - 217.5 / 255) < 1e-12
    for p in (0.8520, 0.8528, 0.8531, 0.8540):  # away from the exact tie
        assert (np.rint(np.float32(p) * 255) >= 218) == (p >= 217.5 / 255)
    # Cloud Score+: bin = v // 10, scored bin >= t  <=>  v >= 10 t  <=>  score >= t / 1000
    thr, rule = native_rule("Cloud Score+ cs | product", 805)
    assert thr == 0.805 and "8050 <= stored value <= 10000" in rule
    for v in (8049, 8050, 8059):
        assert (v // 10 >= 805) == (v >= 8050)
    # s2cloudless: clear score 100 - v >= t  <=>  v <= 100 - t
    thr, rule = native_rule("s2cloudless raw probability | product", 70)
    assert "<= 30%" in rule
