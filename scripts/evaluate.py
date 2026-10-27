"""Score OCM and the operational baselines on the CloudSEN12-overlap-screened test set.

All methods are scored on the same common support (see cmaudit.evaluate). Headline metrics
use the operational policy (abstentions discarded); conditional metrics and coverage are
reported alongside. Writes:
- data/results/per_image_test.parquet
- results/overall_test.csv        pixel policy, ROI-cluster bootstrap 95% CIs
- results/paired_vs_ocm_test.csv  paired ROI-bootstrap differences, OCM L1C minus each method
- results/scene_policy_test.csv   scene filter "keep if predicted cover <= X"

    uv run python scripts/evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cmaudit.evaluate import per_image, stack_cms
from cmaudit.metrics import paired_bootstrap_diff, pixel_summary, scene_policy, summarize
from cmaudit.samples import manifest

STATS = ["boa", "leak_rate", "discard_rate", "leak_thick", "leak_thin", "leak_shadow"]
SCENE_THRESHOLDS = [0.05, 0.10, 0.20, 0.40]
REF = "OmniCloudMask v4 | l1c"


def main() -> None:
    man = manifest("test")
    print(f"test images (CloudSEN12-overlap-screened): {len(man)}  ROIs: {man['roi_id'].nunique()}")
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = per_image(man)
    df.to_parquet(out_dir / "per_image_test.parquet")
    df["key"] = df["method"] + " | " + df["level"]
    keys = df["key"].reset_index(drop=True)
    roi = df["roi_id"].to_numpy()

    tab = summarize(stack_cms(df), keys, roi, STATS)
    cond = []
    for key, g in df.groupby("key"):
        s = pixel_summary(stack_cms(g, "cm_cond").sum(0))
        cov = stack_cms(g, "cm_cond").sum() / g["support_px"].sum()
        cond.append({"group": key, "cond_leak_rate": s["leak_rate"],
                     "cond_discard_rate": s["discard_rate"], "coverage": cov})
    tab = tab.merge(pd.DataFrame(cond), on="group").sort_values("boa", ascending=False)
    Path("results").mkdir(exist_ok=True)
    tab.to_csv("results/overall_test.csv", index=False, float_format="%.4f")

    ref = df[df["key"] == REF].set_index("tortilla:id")
    diffs = []
    for key, g in df.groupby("key"):
        if key == REF:
            continue
        g = g.set_index("tortilla:id").loc[ref.index]
        row = {"method": key}
        for s in ["boa", "leak_rate", "discard_rate"]:
            p, lo, hi = paired_bootstrap_diff(stack_cms(ref), stack_cms(g), ref["roi_id"].to_numpy(), s)
            row.update({f"d_{s}": p, f"d_{s}_lo": lo, f"d_{s}_hi": hi})
        diffs.append(row)
    pd.DataFrame(diffs).to_csv("results/paired_vs_ocm_test.csv", index=False, float_format="%.4f")

    rows = []
    for key, g in df.groupby("key"):
        for t in SCENE_THRESHOLDS:
            rows.append({"key": key, **scene_policy(g["true_cover"], g["pred_cover"], t)})
    pd.DataFrame(rows).to_csv("results/scene_policy_test.csv", index=False, float_format="%.4f")

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        cols = ["group", "boa", "boa_lo", "boa_hi", "leak_rate", "leak_rate_lo", "leak_rate_hi",
                "discard_rate", "leak_thin", "leak_shadow", "coverage", "cond_discard_rate"]
        print(tab[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(pd.DataFrame(diffs).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    print(np.round(stack_cms(df[df["key"] == REF]).sum(0) / 1e6, 2), "(OCM L1C, Mpx, rows=truth)")


if __name__ == "__main__":
    main()
