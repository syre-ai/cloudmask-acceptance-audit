"""Picture examples of each failure mode, picked by rule (not by hand) from the test set.

Maxima are avoided on purpose: the single worst scenes are dominated by thin cloud over snow
and ice, where the expert label itself is least certain. Rules:
- water: the scene with the most *thick*-cloud pixels over water accepted as clear,
- mostly clear (1-30% cover), snow/ice < 5% of the scene: 90th-percentile OCM miss rate,
- heavy (70-99% cover), snow/ice < 5%: 90th-percentile OCM discard rate.
Panels: true-color image, expert label, OCM mask, error map. Writes results/figures/examples.png.

    uv run python scripts/examples.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from cmaudit import figstyle as fs
from cmaudit.evaluate import ocm_pred
from cmaudit.metrics import pixel_summary
from cmaudit.samples import REAL, load, support

CLASS_COLORS = ["#fcfcfb", "#2a78d6", "#86b6ef", "#52514e"]  # clear, thick, thin, shadow
ERR_COLORS = {"leak": fs.SERIES[1], "discard": fs.SERIES[2]}


def rgb(bands: np.ndarray) -> np.ndarray:
    """True color with a per-image 2-98 percentile stretch (bright scenes stay readable)."""
    x = bands[[2, 1, 0]].astype(np.float32)  # B04, B03, B02
    valid = x.sum(0) > 0
    lo, hi = np.percentile(x[:, valid], [2, 98])
    return np.clip((x - lo) / max(hi - lo, 1), 0, 1).transpose(1, 2, 0)


def pick() -> list[tuple[str, str]]:
    per = pd.read_parquet("data/results/per_image_test.parquet")
    per = per[(per["method"] == "OmniCloudMask v4") & (per["level"] == "l1c")].copy()
    s = per["cm"].map(lambda c: pixel_summary(np.asarray(c).reshape(4, 4)))
    per["miss"] = s.map(lambda d: d["miss_rate"])
    per["discard"] = s.map(lambda d: d["discard_rate"])
    per["leaked_px"] = per["cm"].map(lambda c: np.asarray(c).reshape(4, 4)[1:, 0].sum())
    lc = pd.read_parquet("data/results/landcover_cms.parquet")
    lc = lc[lc["method"] == "OmniCloudMask v4"].copy()
    lc["px"] = lc["cm"].map(lambda c: np.asarray(c).sum())
    tot = lc.groupby("tortilla:id")["px"].sum()
    snow_share = (lc[lc["landcover"] == "Snow and ice"].groupby("tortilla:id")["px"].sum() / tot
                  ).reindex(per["tortilla:id"]).fillna(0).to_numpy()
    per = per.assign(snow_share=snow_share)
    water = lc[lc["landcover"] == "Water"].copy()
    water["thick_leak"] = water["cm"].map(lambda c: np.asarray(c).reshape(4, 4)[1, 0])
    a = water.sort_values("thick_leak").iloc[-1]["tortilla:id"]

    def p90(df: pd.DataFrame, col: str) -> str:
        df = df.sort_values(col).reset_index(drop=True)
        return df.iloc[int(0.9 * (len(df) - 1))]["tortilla:id"]

    ok = per[per["snow_share"] < 0.05]
    b = p90(ok[(ok["true_cover"] > 0.01) & (ok["true_cover"] <= 0.30)], "miss")
    c = p90(ok[(ok["true_cover"] > 0.70) & (ok["true_cover"] <= 0.99)], "discard")
    return [(a, "Thick cloud over water accepted"), (b, "Cloud missed, mostly clear scene"),
            (c, "Clear ground discarded, cloudy scene")]


def main() -> None:
    fs.apply()
    cases = pick()
    fig, axes = plt.subplots(len(cases), 4, figsize=(9.6, 2.6 * len(cases)))
    cmap = ListedColormap(CLASS_COLORS)
    for row, (tid, title) in zip(axes, cases):
        lv = load("l1c", tid)
        sup = support(tid)[:REAL, :REAL]
        lab = lv["label"][:REAL, :REAL]
        pred = ocm_pred("l1c", tid)[:REAL, :REAL]
        err = np.zeros(lab.shape + (4,))
        leak = sup & (pred == 0) & (lab > 0)
        disc = sup & (pred > 0) & (lab == 0)
        for m, col in ((leak, ERR_COLORS["leak"]), (disc, ERR_COLORS["discard"])):
            err[m] = plt.matplotlib.colors.to_rgba(col)
        img = rgb(lv["bands"])[:REAL, :REAL]
        row[0].imshow(img)
        row[1].imshow(np.where(lab < 4, lab, 0), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        row[2].imshow(np.where(pred < 4, pred, 0), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        row[3].imshow(img * 0.45 + 0.55)
        row[3].imshow(err, interpolation="nearest")
        row[0].set_ylabel(title, fontsize=9, color=fs.INK)
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    for ax, t in zip(axes[0], ["True color (Sentinel-2 L1C)", "Expert label", "OmniCloudMask v4",
                               "Errors"]):
        ax.set_title(t, fontsize=10)
    handles = [Patch(color=CLASS_COLORS[1], label="thick cloud"),
               Patch(color=CLASS_COLORS[2], label="thin cloud"),
               Patch(color=CLASS_COLORS[3], label="cloud shadow"),
               Patch(color=ERR_COLORS["leak"], label="leak: cloud/shadow accepted"),
               Patch(color=ERR_COLORS["discard"], label="discard: clear rejected")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.05, "Scenes (5.1 km) picked by rule, not by hand: worst water case; "
             "90th-percentile cases otherwise (snow/ice scenes excluded because thin cloud over "
             "snow is ambiguous even for expert labels).\n" + "; ".join(t for t, _ in cases),
             fontsize=7, color=fs.MUTED)
    fig.tight_layout()
    fs.save(fig, "examples")
    print(cases)


if __name__ == "__main__":
    main()
