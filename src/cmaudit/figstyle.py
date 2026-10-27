"""Shared matplotlib styling for report and site figures.

Palette and chrome follow a validated categorical palette (first three slots pass all-pairs
CVD and normal-vision checks on the light surface; aqua is below 3:1 contrast, so every
series is also direct-labeled). Thin marks, hairline grid, text in ink colors only.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def apply() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2, "text.color": INK, "lines.linewidth": 2,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def save(fig, stem: str, out_dir: str = "results/figures") -> None:
    from pathlib import Path

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{stem}.png")
    fig.savefig(f"{out_dir}/{stem}.svg")
    plt.close(fig)


def dot(ax, x, y, color, size=8, **kw):
    """Marker >= 8 px with a 2 px surface ring."""
    return ax.plot([x], [y], "o", ms=size, mfc=color, mec=SURFACE, mew=2, zorder=5, **kw)
