"""Check the encodings of the CloudSEN12+ `extra` cloud masks before scoring them.

For each layer: observed values (or a histogram for continuous scores) and, per value, the
share of expert-label classes it overlaps on the test set. Writes
results/baseline_value_sets.json.

    uv run python scripts/inspect_baselines.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cmaudit.download import EXTRA_LAYERS
from cmaudit.samples import load, manifest, valid_mask

CONTINUOUS = {"cloudmask_cloudscore_cs_v1", "cloudmask_cloudscore_cs_cdf_v1"}


def main() -> None:
    ids = manifest("test")["tortilla:id"].tolist()[::3]
    joint = {k: defaultdict(lambda: np.zeros(4, np.int64)) for k in EXTRA_LAYERS}
    for tid in ids:
        lv = load("l1c", tid)
        ex = load("extra", tid)
        v = valid_mask(lv["bands"])
        lab = lv["label"][v]
        for k in EXTRA_LAYERS:
            x = ex[k][v]
            if k in CONTINUOUS:
                x = (x.astype(np.int64) // 1000) * 1000  # coarse bins
            for val in np.unique(x):
                joint[k][int(val)] += np.bincount(lab[x == val], minlength=4)
    out = {}
    for k, d in joint.items():
        rows = {}
        for val, c in sorted(d.items()):
            n = int(c.sum())
            rows[str(val)] = {"pixels": n, "label_share": [round(float(a) / n, 3) for a in c]}
        out[k] = rows
    Path("results").mkdir(exist_ok=True)
    Path("results/baseline_value_sets.json").write_text(json.dumps(out, indent=1))
    for k, rows in out.items():
        print(k)
        for val, r in list(rows.items())[:16]:
            print(f"  {val:>6}  n={r['pixels']:>9}  clear/thick/thin/shadow={r['label_share']}")


if __name__ == "__main__":
    main()
