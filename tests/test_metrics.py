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
