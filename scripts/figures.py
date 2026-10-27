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
    "s2cloudless | product": ("s2cloudless", fs.SERIES[2]),
}
POINTS = {
    "Sen2Cor SCL (L2A) | product": "Sen2Cor SCL",
    "QA60 (L1C bitmask) | product": "QA60",
    "SEnSeI v2 | product": "SEnSeI v2",
}


def tradeoff() -> None:
    curves = pd.read_csv("results/curves_test.csv")
    overall = pd.read_csv("results/overall_test.csv").set_index("group")
    ops = pd.read_csv("results/operating_points_test.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, (name, color) in CURVES.items():
        c = curves[curves["method"] == key].dropna()
        c = c[(c["discard_rate"] <= 0.6) & (c["leak_rate"] <= 0.35)]
        ax.plot(c["discard_rate"], c["leak_rate"], color=color, label=name)
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
                (o["test_discard"], o["test_leak"]), xytext=(10, -16), textcoords="offset points",
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
    ax.set_title("Cost of a cleaner mask: leak vs discard on 970 test scenes")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0, -0.17, "Curves sweep each product's own score threshold; grey dots are "
            "class-only products at their fixed rules. CloudSEN12+ test split, overlap-screened.",
            transform=ax.transAxes, fontsize=8, color=fs.MUTED)
    fs.save(fig, "tradeoff")


def main() -> None:
    fs.apply()
    tradeoff()


if __name__ == "__main__":
    main()
