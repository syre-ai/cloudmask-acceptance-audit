"""A4: SIMULATED input degradations on the clean test set (L1C), OCM v4.

For every perturbation x severity, OCM is re-run on degraded inputs and scored against the
unchanged expert label on the common support. Effects are paired differences from the clean
run (same images), with ROI-cluster bootstrap 95% CIs.

For coarser resolution an *oracle floor* is also reported: the expert label itself reduced to
the coarse grid (block majority) and mapped back to 10 m. That is the best any mask at that
resolution could do against a 10 m label, so OCM's loss beyond it is the model's share.

A per-band gain/offset check is run once: OCM normalizes each band per image, so predictions
should be unchanged apart from clipping; reported as pixel agreement with the clean run.

Writes data/results/perturb_cms.parquet, results/perturbations_test.csv and
results/gain_offset_check.json.

    uv run python scripts/perturbations.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cmaudit.baselines import NO_DECISION
from cmaudit.evaluate import ocm_pred, score_row, stack_cms
from cmaudit.infer import load_models, ocm_input, predict_probs
from cmaudit.metrics import paired_bootstrap_diff
from cmaudit.perturb import SUITE, gain_offset, upsample_pred
from cmaudit.samples import load, manifest, support

OUT = Path("data/results/perturb_cms.parquet")
CHUNK = 32
STATS = ["boa", "leak_rate", "discard_rate", "miss_rate"]


def to_pred(p: np.ndarray) -> np.ndarray:
    pred = p.argmax(0).astype(np.uint8)
    pred[p.sum(0) == 0] = NO_DECISION
    return pred


def label_majority(label: np.ndarray, factor: int) -> np.ndarray:
    """Coarse-grid oracle: majority class per block of the (edge-padded) real label."""
    x = np.pad(label[:509, :509], ((0, 1), (0, 1)), mode="edge")
    n = 510 // factor
    blocks = x.reshape(n, factor, n, factor).transpose(0, 2, 1, 3).reshape(n, n, -1)
    counts = np.stack([(blocks == k).sum(-1) for k in range(4)], -1)
    return counts.argmax(-1).astype(np.uint8)


def run(models, ids: list[str]) -> pd.DataFrame:
    rows = []
    jobs = [(name, s) for name, (_, _, sevs, _) in SUITE.items() for s in sevs]
    jobs.append(("Oracle floor (label at coarse grid)", None))
    for name, sev in tqdm(jobs, desc="perturbations"):
        if name.startswith("Oracle"):
            for f in SUITE["Coarser resolution"][2]:
                for tid in ids:
                    lab = load("l1c", tid)["label"]
                    pred = upsample_pred(label_majority(lab, f), f)
                    rows.append({"tortilla:id": tid, "perturbation": name, "severity": f,
                                 **score_row(lab, pred, support(tid))})
            continue
        fn, kw, _, _ = SUITE[name]
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i : i + CHUNK]
            imgs = [fn(ocm_input(load("l1c", t)["bands"]), **{kw: sev}) for t in chunk]
            probs = predict_probs(models, imgs, dtype=torch.float32)
            for tid, p in zip(chunk, probs):
                pred = to_pred(p)
                if name == "Coarser resolution":
                    pred = upsample_pred(pred, sev)
                rows.append({"tortilla:id": tid, "perturbation": name, "severity": sev,
                             **score_row(load("l1c", tid)["label"], pred, support(tid))})
    return pd.DataFrame(rows)


def gain_offset_check(models, ids: list[str]) -> dict:
    agree, total = 0, 0
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i : i + CHUNK]
        imgs = [gain_offset(ocm_input(load("l1c", t)["bands"]), gains=(0.9, 1.0, 1.15),
                            offsets=(0.0, 0.01, -0.005)) for t in chunk]
        probs = predict_probs(models, imgs, dtype=torch.float32)
        for tid, p in zip(chunk, probs):
            sup = support(tid)
            agree += int((to_pred(p)[sup] == ocm_pred("l1c", tid)[sup]).sum())
            total += int(sup.sum())
    return {"perturbation": "per-band gain (0.9, 1.0, 1.15) and offset (0, +0.01, -0.005) "
            "reflectance on R, G, NIR", "images": len(ids), "pixel_agreement_with_clean":
            agree / total}


def main() -> None:
    man = manifest("test")
    ids = man["tortilla:id"].tolist()
    roi_of = dict(zip(man["tortilla:id"], man["roi_id"]))
    models = load_models(dtype=torch.float32)
    if OUT.exists():
        df = pd.read_parquet(OUT)
    else:
        df = run(models, ids)
        df.to_parquet(OUT)

    clean = pd.read_parquet("data/results/per_image_test.parquet")
    clean = clean[(clean["method"] == "OmniCloudMask v4") & (clean["level"] == "l1c")]
    clean = clean.set_index("tortilla:id").loc[ids]
    roi = np.array([roi_of[t] for t in ids])
    rows = []
    for (name, sev), g in df.groupby(["perturbation", "severity"], sort=False):
        g = g.set_index("tortilla:id").loc[ids]
        row = {"perturbation": name, "severity": sev, "simulated": True}
        for s in STATS:
            p, lo, hi = paired_bootstrap_diff(stack_cms(g), stack_cms(clean), roi, s)
            row.update({f"d_{s}": p, f"d_{s}_lo": lo, f"d_{s}_hi": hi})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv("results/perturbations_test.csv", index=False, float_format="%.4f")
    check = gain_offset_check(models, ids[::5])
    Path("results/gain_offset_check.json").write_text(json.dumps(check, indent=1))
    with pd.option_context("display.width", 220):
        print(tab[["perturbation", "severity", "d_boa", "d_boa_lo", "d_boa_hi", "d_leak_rate",
                   "d_leak_rate_lo", "d_leak_rate_hi", "d_discard_rate", "d_miss_rate"]]
              .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(check)


if __name__ == "__main__":
    main()
