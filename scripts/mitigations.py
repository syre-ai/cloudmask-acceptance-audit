"""A5: tested mitigations, chosen on validation and reported on test.

M1  Buffer the cloud/shadow mask by k pixels (standard practice; targets the edge misses
    seen in mostly clear scenes). k in 0..8 is evaluated on validation; we report the k that
    meets a 2% leak target with the least discard, and compare it with plain thresholding of
    P(clear) at the same leak (A2) to see which lever costs less data.
M2  Scene gate: "keep scene if cloud+shadow cover <= 10%", estimated from OCM vs from the
    product's Sen2Cor SCL. Reports false accepts and the pixel leak that actually reaches
    the pipeline inside kept scenes (after masking with OCM).

Writes results/mitigation_buffer.csv and results/mitigation_scene_gate.csv.

    uv run python scripts/mitigations.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from cmaudit.baselines import BASELINES, NO_DECISION
from cmaudit.evaluate import ocm_pred, operational, score_row, stack_cms
from cmaudit.metrics import cluster_bootstrap, confusion, cover, pixel_summary
from cmaudit.samples import load, manifest, support

KS = list(range(9))
GATE = 0.10
OUT = Path("data/results")


def buffered(pred: np.ndarray, k: int) -> np.ndarray:
    """Grow every non-clear decision by k pixels (disk); abstentions stay abstentions."""
    if k == 0:
        return pred
    y, x = np.ogrid[-k : k + 1, -k : k + 1]
    disk = x * x + y * y <= k * k
    grown = ndimage.binary_dilation((pred > 0) & (pred != NO_DECISION), structure=disk)
    out = pred.copy()
    out[grown & (pred == 0)] = 1
    return out


def buffer_cms(split: str) -> pd.DataFrame:
    path = OUT / f"buffer_cms_{split}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    man = manifest(split)
    rows = []
    for tid, roi in tqdm(list(zip(man["tortilla:id"], man["roi_id"])), desc=f"buffer {split}",
                         mininterval=10):
        lab = load("l1c", tid)["label"]
        sup = support(tid)
        pred = ocm_pred("l1c", tid)
        for k in KS:
            r = score_row(lab, buffered(pred, k), sup)
            rows.append({"tortilla:id": tid, "roi_id": roi, "k": k, "cm": r["cm"]})
    df = pd.DataFrame(rows)
    df.to_parquet(path)
    return df


def m1() -> pd.DataFrame:
    val, test = buffer_cms("validation"), buffer_cms("test")
    rows = []
    for k in KS:
        v = pixel_summary(stack_cms(val[val["k"] == k]).sum(0))
        t = test[test["k"] == k]
        row = {"k_px": k, "val_leak": v["leak_rate"], "val_discard": v["discard_rate"]}
        for s in ("leak_rate", "discard_rate"):
            p, lo, hi = cluster_bootstrap(stack_cms(t), t["roi_id"].to_numpy(), s, n_boot=500)
            row.update({f"test_{s}": p, f"test_{s}_lo": lo, f"test_{s}_hi": hi})
        rows.append(row)
    tab = pd.DataFrame(rows)
    ok = tab[tab["val_leak"] <= 0.02]
    tab["chosen_for_2pct_leak"] = tab["k_px"] == (ok["k_px"].min() if len(ok) else -1)
    return tab


def m2() -> pd.DataFrame:
    man = manifest("test")
    rows = []
    for tid, roi in tqdm(list(zip(man["tortilla:id"], man["roi_id"])), desc="gate", mininterval=10):
        lab = load("l1c", tid)["label"]
        sup = support(tid)
        ocm = ocm_pred("l1c", tid)
        scl = BASELINES["Sen2Cor SCL (L2A)"][1](load("extra", tid)["cloudmask_sen2cor"])
        rows.append({"tortilla:id": tid, "roi_id": roi, "true_cover": cover(lab, sup),
                     "cover_ocm": cover(operational(ocm), sup),
                     "cover_scl": cover(operational(scl), sup),
                     "cm_ocm": confusion(lab, operational(ocm), sup).ravel()})
    df = pd.DataFrame(rows)
    out = []
    for gate_by in ("cover_ocm", "cover_scl"):
        kept = df[df[gate_by] <= GATE]
        cms = np.stack(kept["cm_ocm"].to_numpy()).reshape(-1, 4, 4)
        leak, lo, hi = cluster_bootstrap(cms, kept["roi_id"].to_numpy(), "leak_rate", n_boot=500)
        out.append({
            "gate_cover_from": {"cover_ocm": "OmniCloudMask", "cover_scl": "Sen2Cor SCL"}[gate_by],
            "threshold": GATE, "scenes_kept": len(kept), "scenes_total": len(df),
            "kept_truly_over_threshold": float((kept["true_cover"] > GATE).mean()),
            "truly_ok_scenes_rejected": float(((df[gate_by] > GATE) & (df["true_cover"] <= GATE)).sum()
                                              / max((df["true_cover"] <= GATE).sum(), 1)),
            "pixel_leak_in_kept_after_ocm_mask": leak, "leak_lo": lo, "leak_hi": hi,
        })
    return pd.DataFrame(out)


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    t1 = m1()
    t1.to_csv("results/mitigation_buffer.csv", index=False, float_format="%.4f")
    t2 = m2()
    t2.to_csv("results/mitigation_scene_gate.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 220):
        print(t1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(t2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
