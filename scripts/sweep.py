"""A2: leak/discard trade-off curves and operating points.

Thresholds are chosen on the CloudSEN12+ validation split and reported on the test split, so
the "recommended settings" are not tuned on the numbers they are judged by.

Writes data/results/hists_<split>.npz, results/operating_points_test.csv,
results/curves_test.csv (decimated curve points for plotting).

    uv run python scripts/sweep.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cmaudit.evaluate import PRED_DIR
from cmaudit.samples import load, manifest, valid_mask
from cmaudit.sweep import curve, pick_threshold, score_hist

OUT = Path("data/results")

# method key -> number of integer clear-score bins (see clear_scores)
METHODS = {
    "OmniCloudMask v4 | l1c": 256,
    "OmniCloudMask v4 | l2a": 256,
    "s2cloudless | product": 101,
    "Cloud Score+ cs | product": 1001,
    "Cloud Score+ cs_cdf | product": 1001,
}
# Native scale of the clear score, for reporting thresholds in the product's own units.
TO_NATIVE = {256: 1 / 255, 101: 1 / 100, 1001: 1 / 1000}
OPERATING_POINTS = [("leak", 0.02), ("leak", 0.05), ("discard", 0.02), ("discard", 0.05)]
N_BOOT = 1000


def clear_scores(tid: str) -> dict[str, np.ndarray]:
    ex = load("extra", tid)
    out = {}
    for level in ("l1c", "l2a"):
        with np.load(PRED_DIR / level / f"{tid}.npz") as z:
            out[f"OmniCloudMask v4 | {level}"] = z["probs"][0].astype(np.int64)
    out["s2cloudless | product"] = 100 - ex["cloudmask_s2cloudless"].astype(np.int64)
    for k, layer in (("cs", "cloudmask_cloudscore_cs_v1"), ("cs_cdf", "cloudmask_cloudscore_cs_cdf_v1")):
        v = ex[layer].astype(np.int64)
        out[f"Cloud Score+ {k} | product"] = np.where(v <= 10000, np.rint(v / 10).astype(np.int64), -1)
    return out


def build_hists(split: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    path = OUT / f"hists_{split}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in METHODS}, z["roi"]
    man = manifest(split)
    hists = {k: [] for k in METHODS}
    for tid in tqdm(man["tortilla:id"], desc=f"hists {split}", mininterval=10):
        lv = load("l1c", tid)
        valid = valid_mask(lv["bands"])
        scores = clear_scores(tid)
        for k, nb in METHODS.items():
            hists[k].append(score_hist(lv["label"], scores[k], valid, nb))
    hists = {k: np.stack(v) for k, v in hists.items()}
    roi = man["roi_id"].to_numpy().astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, roi=roi, **hists)
    return hists, roi


def boot_at(h: np.ndarray, roi: np.ndarray, t: int, seed: int = 0) -> dict[str, tuple]:
    codes, inv = np.unique(roi, return_inverse=True)
    per = np.zeros((len(codes),) + h.shape[1:])
    np.add.at(per, inv, h)
    rng = np.random.default_rng(seed)
    draws = {"leak_rate": [], "discard_rate": []}
    for _ in range(N_BOOT):
        c = curve(per[rng.integers(0, len(codes), len(codes))].sum(0))
        for s, vals in draws.items():
            vals.append(c[s][t])
    point = curve(h.sum(0))
    return {s: (point[s][t], *np.nanquantile(draws[s], [0.025, 0.975])) for s in draws}


def main() -> None:
    val, _ = build_hists("validation")
    test, roi = build_hists("test")
    rows, curve_rows = [], []
    for k, nb in METHODS.items():
        cv = curve(val[k].sum(0))
        ct = curve(test[k].sum(0))
        for target, limit in OPERATING_POINTS:
            t = pick_threshold(cv, target, limit)
            b = boot_at(test[k], roi, t)
            reachable = bool(cv[f"{target}_rate"][t] <= limit) and t < nb
            rows.append({
                "method": k, "rule": f"{target} <= {limit:.0%} (chosen on validation)",
                "achievable": reachable,
                "threshold_native": round(t * TO_NATIVE[nb], 4),
                "val_leak": cv["leak_rate"][t], "val_discard": cv["discard_rate"][t],
                "test_leak": b["leak_rate"][0], "test_leak_lo": b["leak_rate"][1],
                "test_leak_hi": b["leak_rate"][2],
                "test_discard": b["discard_rate"][0], "test_discard_lo": b["discard_rate"][1],
                "test_discard_hi": b["discard_rate"][2],
                "test_leak_thin": ct["leak_thin"][t], "test_leak_shadow": ct["leak_shadow"][t],
            })
        step = max(1, nb // 100)
        for t in range(0, nb + 1, step):
            curve_rows.append({"method": k, "threshold_native": t * TO_NATIVE[nb],
                               "leak_rate": ct["leak_rate"][t],
                               "discard_rate": ct["discard_rate"][t]})
    ops = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    ops.to_csv("results/operating_points_test.csv", index=False, float_format="%.4f")
    pd.DataFrame(curve_rows).to_csv("results/curves_test.csv", index=False, float_format="%.5f")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(ops[["method", "rule", "achievable", "threshold_native", "val_leak", "test_leak", "test_leak_lo",
                   "test_leak_hi", "test_discard", "test_discard_lo", "test_discard_hi"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
