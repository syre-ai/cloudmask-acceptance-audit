"""Report/site figures from the committed result tables (no data/ access needed).

    uv run python scripts/figures.py
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from cmaudit import figstyle as fs

CURVES = {
    "OmniCloudMask v4 | l1c": ("OmniCloudMask v4", fs.SERIES[0]),
    "Cloud Score+ cs | product": ("Cloud Score+ (cs)", fs.SERIES[1]),
    "s2cloudless raw probability | product": ("s2cloudless (raw probability)", fs.SERIES[2]),
}
POINTS = {
    "Sen2Cor SCL (L2A) | product": "Sen2Cor SCL",
    "QA60 (L1C bitmask) | product": "QA60",
    "SEnSeI v2 | product": "SEnSeI v2",
}


def n_scenes() -> int:
    ov = pd.read_csv("results/overall_test.csv").set_index("group")
    return int(ov.loc["OmniCloudMask v4 | l1c", "images"])


def tradeoff() -> None:
    curves = pd.read_csv("results/curves_test.csv")
    overall = pd.read_csv("results/overall_test.csv").set_index("group")
    ops = pd.read_csv("results/operating_points_test.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, (name, color) in CURVES.items():
        c = curves[curves["method"] == key].dropna()
        c = c[(c["discard_rate"] <= 0.6) & (c["leak_rate"] <= 0.35)]
        ax.plot(c["discard_rate"], c["leak_rate"], color=color, label=name)
        if key.startswith("OmniCloudMask"):
            continue  # its annotated operating points sit where an end label would collide
        end = c.iloc[(c["discard_rate"] - 0.6).abs().argmin()]
        ax.annotate(name, (end["discard_rate"], end["leak_rate"]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=fs.INK_2)
    # OCM default (argmax) and the validation-chosen 2% leak point
    d = overall.loc["OmniCloudMask v4 | l1c"]
    fs.dot(ax, d["discard_rate"], d["leak_rate"], fs.SERIES[0])
    ax.annotate(f"OCM default: leak {d['leak_rate']:.1%}, discard {d['discard_rate']:.1%}",
                (d["discard_rate"], d["leak_rate"]), xytext=(10, 12), textcoords="offset points",
                fontsize=9, color=fs.INK)
    o = ops[(ops["method"] == "OmniCloudMask v4 | l1c") & ops["rule"].str.startswith("leak <= 2%")].iloc[0]
    fs.dot(ax, o["test_discard"], o["test_leak"], fs.SERIES[0])
    ax.annotate(f"OCM tuned for 2% leak: discard {o['test_discard']:.1%}",
                (o["test_discard"], o["test_leak"]), xytext=(14, 9), textcoords="offset points",
                fontsize=9, color=fs.INK)
    for key, name in POINTS.items():
        r = overall.loc[key]
        fs.dot(ax, r["discard_rate"], r["leak_rate"], fs.MUTED)
        ax.annotate(name, (r["discard_rate"], r["leak_rate"]), xytext=(7, -3),
                    textcoords="offset points", fontsize=9, color=fs.INK_2)
    ax.set_xlim(0, 0.72)
    ax.set_ylim(0, 0.35)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Clear pixels thrown away (discard rate)")
    ax.set_ylabel("Accepted pixels that are cloud or shadow (leak rate)")
    ax.set_title(f"Cost of a cleaner mask: leak vs discard on {n_scenes()} test scenes")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0, -0.17, "Curves sweep each product's own score threshold; grey dots are "
            "class-only products at their fixed rules. CloudSEN12+ test split, overlap-screened.",
            transform=ax.transAxes, fontsize=8, color=fs.MUTED)
    fs.save(fig, "tradeoff")


def landcover() -> None:
    lc = pd.read_csv("results/subgroups_landcover.csv")
    overall = pd.read_csv("results/overall_test.csv").set_index("group").loc["OmniCloudMask v4 | l1c"]
    d = lc[lc["method"] == "OmniCloudMask v4"].sort_values("leak_rate")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = range(len(d))
    ax.axvline(overall["leak_rate"], color=fs.AXIS, lw=1, zorder=1)
    ax.text(overall["leak_rate"], -0.9, f" all scenes {overall['leak_rate']:.1%}",
            fontsize=8, color=fs.MUTED, va="center")
    ax.hlines(list(y), d["leak_rate_lo"], d["leak_rate_hi"], color=fs.SERIES[0], lw=2)
    for yi, (_, r) in zip(y, d.iterrows()):
        fs.dot(ax, r["leak_rate"], yi, fs.SERIES[0])
        ax.text(r["leak_rate_hi"] + 0.004, yi, f"{r['leak_rate']:.1%}  ({r['rois']} sites)",
                va="center", fontsize=8, color=fs.INK_2)
    ax.set_yticks(list(y), d["group"])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(-1.3, len(d) - 0.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Leak rate: accepted pixels that are cloud or shadow (95% CI)")
    ax.set_title("OmniCloudMask leaks most over snow, ice and water")
    ax.text(0, -0.2, "Land cover from ESA WorldCover under each pixel. CI: bootstrap over sites "
            f"(5 dates per site). CloudSEN12+ test, {n_scenes()} scenes.", transform=ax.transAxes,
            fontsize=8, color=fs.MUTED)
    fs.save(fig, "landcover_leak")


def regimes() -> None:
    sc = pd.read_csv("results/subgroups_scene.csv")
    d = sc[(sc["dimension"] == "Cloud regime")]
    order = ["clear (<1%)", "light (1-30%)", "mixed (30-70%)", "heavy (70-99%)", "overcast (>99%)"]
    names = {"OmniCloudMask v4": fs.SERIES[0], "Cloud Score+ cs": fs.SERIES[1],
             "Sen2Cor SCL (L2A)": fs.SERIES[2]}
    # separate y axes: the two panels cover different regimes (sharing would relabel one)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    panels = [("miss_rate", "Cloud/shadow let through (miss rate)", order[1:]),
              ("discard_rate", "Clear pixels thrown away (discard rate)", order[:-1])]
    for ax, (stat, title, groups) in zip(axes, panels):
        for k, (m, color) in enumerate(names.items()):
            r = d[d["method"] == m].set_index("group").loc[groups]
            ys = [i + (k - 1) * 0.22 for i in range(len(groups))]
            ax.hlines(ys, r[f"{stat}_lo"], r[f"{stat}_hi"], color=color, lw=2)
            for yy, v in zip(ys, r[stat]):
                fs.dot(ax, v, yy, color)
            ax.plot([], [], "o", color=color, label=m)
        ax.set_yticks(range(len(groups)), groups)
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Failures depend on how cloudy the scene is", x=0.02, ha="left",
                 fontweight="bold", fontsize=11)
    fig.text(0.02, -0.04, "Scenes grouped by expert-labeled cloud+shadow cover. Points with 95% "
             "site-bootstrap CIs. Miss rate is undefined for clear scenes, discard for overcast.",
             fontsize=8, color=fs.MUTED)
    fig.tight_layout()
    fs.save(fig, "regimes")


SEV_LABEL = {
    "Sensor noise": lambda v: f"σ {v:g}",
    "Optical blur": lambda v: f"{v:g} px",
    "Coarser resolution": lambda v: f"{int(v) * 10} m",
    "Coarse data on a 10 m grid": lambda v: f"{int(v) * 10} m",
    "Coarse data on a 10 m grid (nearest)": lambda v: f"{int(v) * 10} m",
    "Clip only (16-bit)": lambda v: f"clip {v:g}",
    "NIR misregistration": lambda v: f"{int(v)} px",
    "8-bit JPEG delivery": lambda v: f"q{int(v)}",
    "8-bit lossless (clip level)": lambda v: f"clip {v:g}",
}


def stress() -> None:
    t = pd.read_csv("results/perturbations_test.csv")
    t = t[~t["perturbation"].str.startswith("Majority-label")]
    names = list(dict.fromkeys(t["perturbation"]))
    cols = 3
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.0, 2.4 * rows), sharey=True)
    axes = axes.ravel()
    lo = min(t["d_leak_rate_lo"].min(), t["d_discard_rate_lo"].min())
    hi = max(t["d_leak_rate_hi"].max(), t["d_discard_rate_hi"].max())
    for ax, name in zip(axes, names):
        g = t[t["perturbation"] == name].reset_index(drop=True)
        x = range(len(g))
        ax.axhline(0, color=fs.AXIS, lw=1)
        for stat, color, off, lab in (("leak_rate", fs.SERIES[1], -0.08, "leak"),
                                       ("discard_rate", fs.SERIES[0], 0.08, "discard")):
            xs = [i + off for i in x]
            ax.vlines(xs, g[f"d_{stat}_lo"], g[f"d_{stat}_hi"], color=color, lw=2)
            ax.plot(xs, g[f"d_{stat}"], "o", ms=6, mfc=color, mec=fs.SURFACE, mew=1.5, label=lab)
        ax.set_xticks(list(x), [SEV_LABEL[name](v) for v in g["severity"]], fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_ylim(lo - 0.02, hi + 0.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="x", visible=False)
    for ax in axes[len(names):]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Stress test (simulated): change in leak and discard vs the clean run",
                 x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.text(0.01, -0.02, f"Paired over the same {n_scenes()} test scenes; bars are 95% site-bootstrap CIs. "
             "Points above zero are worse.", fontsize=8, color=fs.MUTED)
    fig.tight_layout()
    fs.save(fig, "stress")


def main() -> None:
    fs.apply()
    tradeoff()
    landcover()
    regimes()
    stress()


if __name__ == "__main__":
    main()
