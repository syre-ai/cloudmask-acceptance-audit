import numpy as np
import pandas as pd

from cmaudit.metrics import (
    cluster_bootstrap,
    confusion,
    cover,
    pixel_summary,
    scene_policy,
    summarize,
)


def test_confusion_and_summary_hand_case():
    # truth: 6 clear, 2 thick, 1 thin, 1 shadow
    label = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 3])
    # mask misses the thin cloud and the shadow, and discards one clear pixel
    pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    cm = confusion(label, pred)
    assert cm.sum() == 10 and cm[0, 0] == 5 and cm[2, 0] == 1 and cm[3, 0] == 1
    s = pixel_summary(cm)
    # accepted = 7 (5 clear + thin + shadow); leaked = 2
    assert np.isclose(s["leak_rate"], 2 / 7)
    assert np.isclose(s["leak_thin"], 1 / 7) and np.isclose(s["leak_shadow"], 1 / 7)
    assert np.isclose(s["leak_thick"], 0.0)
    assert np.isclose(s["discard_rate"], 1 / 6)
    assert np.isclose(s["miss_rate"], 2 / 4)
    # cloud/shadow recall 2/4, clear recall 5/6
    assert np.isclose(s["boa"], 0.5 * (0.5 + 5 / 6))


def test_invalid_pixels_ignored():
    label = np.array([0, 1, 99])
    pred = np.array([0, 0, 0])
    valid = np.array([True, False, True])
    cm = confusion(label, pred, valid)
    assert cm.sum() == 1 and cm[0, 0] == 1
    assert cover(np.array([0, 1, 2, 99])) == 2 / 3


def test_scene_policy():
    true = np.array([0.0, 0.05, 0.5, 0.2])
    pred = np.array([0.0, 0.2, 0.05, 0.2])
    r = scene_policy(true, pred, 0.1)
    # accepted: scenes 0 and 2; scene 2 is truly 50% cloudy -> 1 of 2 false accepts
    assert r["accepted"] == 2 and np.isclose(r["false_accept_rate"], 0.5)
    # truly ok: scenes 0, 1; scene 1 rejected -> 1 of 2
    assert np.isclose(r["false_reject_rate"], 0.5)


def test_cluster_bootstrap_point_and_ci():
    rng = np.random.default_rng(1)
    cms = []
    for _ in range(40):
        lab = rng.integers(0, 4, 500)
        pr = np.where(rng.random(500) < 0.8, lab, rng.integers(0, 4, 500))
        cms.append(confusion(lab, pr))
    cms = np.array(cms)
    clusters = np.repeat(np.arange(8), 5)
    p, lo, hi = cluster_bootstrap(cms, clusters, "leak_rate", n_boot=200)
    assert lo <= p <= hi and 0 < p < 1
    tab = summarize(cms, pd.Series(np.repeat(["a", "b"], 20)), clusters, ["boa"], n_boot=50)
    assert set(tab["group"]) == {"a", "b"} and (tab["rois"] == 4).all()


def test_baseline_decoders():
    from cmaudit.baselines import NO_DECISION, cloudscore, qa60, s2cloudless, sen2cor

    assert qa60(np.array([0, 1024, 2048, 16384, 32768], np.uint16)).tolist() == [0, 1, 2, 1, 2]
    assert sen2cor(np.array([0, 3, 4, 8, 9, 10, 11])).tolist() == [NO_DECISION, 3, 0, 1, 1, 2, 0]
    from cmaudit.baselines import s2cloudless_raw

    assert s2cloudless_raw(np.array([0, 40, 41, 100])).tolist() == [0, 0, 1, 1]
    # upstream post-processing: a lone 100% pixel is averaged away; a big blob is kept and grown
    lone = np.zeros((20, 20), np.uint8)
    lone[10, 10] = 100
    assert s2cloudless(lone).sum() == 0
    blob = np.zeros((40, 40), np.uint8)
    blob[10:30, 10:30] = 100
    m = s2cloudless(blob)
    assert m[20, 20] == 1 and m.sum() > 400
    assert cloudscore(np.array([0, 5999, 6000, 10000, 65000])).tolist() == [1, 1, 0, 0, NO_DECISION]


def test_operational_policy_discards_abstentions():
    from cmaudit.baselines import NO_DECISION
    from cmaudit.evaluate import score_row

    label = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, NO_DECISION, NO_DECISION, 0]])
    sup = np.ones_like(label, bool)
    r = score_row(label, pred, sup)
    op = pixel_summary(r["cm"].reshape(4, 4))
    cond = pixel_summary(r["cm_cond"].reshape(4, 4))
    assert r["coverage"] == 0.5
    # operational: abstained clear pixel counts as discarded, abstained cloud is not leaked
    assert np.isclose(op["discard_rate"], 0.5) and np.isclose(op["leak_rate"], 0.5)
    assert np.isclose(cond["discard_rate"], 0.0)
    assert r["pred_cover"] == 0.5 and r["true_cover"] == 0.5


def test_paired_diff_zero_for_identical():
    from cmaudit.metrics import paired_bootstrap_diff

    rng = np.random.default_rng(3)
    cms = np.array([confusion(rng.integers(0, 4, 300), rng.integers(0, 4, 300)) for _ in range(12)])
    p, lo, hi = paired_bootstrap_diff(cms, cms, np.repeat(np.arange(4), 3), "boa", n_boot=50)
    assert p == lo == hi == 0.0
