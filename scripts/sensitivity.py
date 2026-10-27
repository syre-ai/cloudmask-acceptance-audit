"""Sensitivity of the headline to the one training source we cannot localize below ~110 km:
Kappaset scenes name only their MGRS tile. Re-score with every test image on a Kappaset tile
removed and compare. Writes results/sensitivity_kappaset.csv.

    uv run python scripts/sensitivity.py
"""

import pandas as pd

from cmaudit.evaluate import stack_cms
from cmaudit.metrics import cluster_bootstrap
from cmaudit.samples import manifest

STATS = ["boa", "leak_rate", "discard_rate"]


def main() -> None:
    per = pd.read_parquet("data/results/per_image_test.parquet")
    per = per[(per["method"] == "OmniCloudMask v4") & (per["level"] == "l1c")]
    rows = []
    for name, man in (("all screened test images", manifest("test")),
                      ("excluding Kappaset MGRS tiles", manifest("test", exclude_kappaset_tiles=True))):
        d = per[per["tortilla:id"].isin(man["tortilla:id"])]
        row = {"subset": name, "images": len(d), "rois": d["roi_id"].nunique()}
        for s in STATS:
            p, lo, hi = cluster_bootstrap(stack_cms(d), d["roi_id"].to_numpy(), s)
            row.update({s: p, f"{s}_lo": lo, f"{s}_hi": hi})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv("results/sensitivity_kappaset.csv", index=False, float_format="%.4f")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
