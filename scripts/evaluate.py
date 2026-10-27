"""Score OCM and the operational baselines on the clean CloudSEN12+ test set.

Writes data/results/per_image_test.parquet and results/overall_test.csv (pixel policy, with
ROI-cluster bootstrap 95% CIs) and results/scene_policy_test.csv.

    uv run python scripts/evaluate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cmaudit.evaluate import per_image, stack_cms
from cmaudit.metrics import scene_policy, summarize
from cmaudit.samples import manifest

STATS = ["boa", "leak_rate", "discard_rate", "leak_thick", "leak_thin", "leak_shadow"]
SCENE_THRESHOLDS = [0.05, 0.10, 0.20, 0.40]


def main() -> None:
    man = manifest("test")
    print(f"test images (clean tier): {len(man)}  ROIs: {man['roi_id'].nunique()}")
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = per_image(man)
    df.to_parquet(out_dir / "per_image_test.parquet")

    df["key"] = df["method"] + " | " + df["level"]
    cms = stack_cms(df)
    tab = summarize(cms, df["key"].reset_index(drop=True), df["roi_id"].to_numpy(), STATS)
    cov = df.groupby("key")["coverage"].mean().rename("coverage")
    tab = tab.merge(cov, left_on="group", right_index=True).sort_values("boa", ascending=False)
    Path("results").mkdir(exist_ok=True)
    tab.to_csv("results/overall_test.csv", index=False, float_format="%.4f")

    rows = []
    for key, g in df.groupby("key"):
        for t in SCENE_THRESHOLDS:
            rows.append({"key": key, **scene_policy(g["true_cover"], g["pred_cover"], t)})
    pd.DataFrame(rows).to_csv("results/scene_policy_test.csv", index=False, float_format="%.4f")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        cols = ["group", "boa", "boa_lo", "boa_hi", "leak_rate", "leak_rate_lo", "leak_rate_hi",
                "discard_rate", "leak_thin", "leak_shadow", "coverage"]
        print(tab[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(np.round(stack_cms(df[df["key"] == "OmniCloudMask v4 | l1c"]).sum(0) / 1e6, 2),
          "(OCM L1C confusion, Mpx, rows=truth)")


if __name__ == "__main__":
    main()
