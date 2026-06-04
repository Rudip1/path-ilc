"""
Shared, publication-quality matplotlib style for every figure in the project.

    from plot_style import use_style, C, bar_log, label_bars
    use_style()

One consistent look across all scripts: clean sans-serif, no top/right spines,
a soft y-grid behind the data, and a single colorblind-safe (Okabe-Ito) palette
with *semantic* names so "no correction" / "naive" / "learned" / "oracle" are the
same colour in every plot.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -- Okabe-Ito colourblind-safe palette ------------------------------------
OKABE = {
    "black": "#000000", "blue": "#0072B2", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "orange": "#E69F00",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}


class C:
    """Semantic colours — use these so a concept looks identical everywhere."""
    NO_CORR = OKABE["grey"]        # baseline / no correction (neutral "before")
    NAIVE = OKABE["orange"]        # naive table reuse
    LEARNED = OKABE["green"]       # learned correction (NN / linear) — the good result
    ILC = OKABE["green"]           # after-ILC result
    BASE = OKABE["blue"]           # generic primary series / re-learn
    ORACLE = OKABE["blue"]         # oracle re-learn
    WARM = OKABE["vermillion"]     # "bad"/degraded series, temperature
    AXES = [OKABE["blue"], OKABE["vermillion"], OKABE["green"]]   # x, y, z


def use_style():
    """Apply the project-wide rcParams. Call once at import time per script."""
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "semibold",
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#b0b0b0",
        "grid.linestyle": "-", "grid.linewidth": 0.6, "grid.alpha": 0.30,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
        "lines.linewidth": 1.9, "lines.markersize": 5.5,
        "legend.fontsize": 9, "legend.framealpha": 0.92,
        "legend.edgecolor": "#cccccc", "legend.fancybox": False,
    })


def label_bars(ax, values, fmt="{:,.0f}", dy=1.04, fontsize=9.5):
    """Place value labels just above each bar (multiplicative offset for log)."""
    for xi, v in enumerate(values):
        ax.text(xi, v * dy, fmt.format(v), ha="center", va="bottom",
                fontsize=fontsize, color="#222222")


def bar_log(ax, labels, values, colors, ylabel, title=None,
            annotate_reduction=True):
    """A professional log-scale bar chart.

    Log scale keeps a huge baseline from squashing the small (important) bars.
    Value labels sit above each bar; optionally the reduction factor relative to
    the first (baseline) bar is annotated on the later bars.
    """
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values))
    bars = ax.bar(x, values, width=0.62, color=colors, zorder=3,
                  edgecolor="white", linewidth=0.7)
    ax.set_yscale("log")
    lo = 10 ** np.floor(np.log10(values.min() * 0.6))
    ax.set_ylim(lo, values.max() * 2.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    label_bars(ax, values)
    if annotate_reduction and len(values) > 1:
        base = values[0]
        for xi in range(1, len(values)):
            fac = base / values[xi]
            if fac >= 1.5:
                ax.text(xi, values[xi] * 0.5, f"{fac:.0f}×\nlower",
                        ha="center", va="top", fontsize=8.5,
                        color="#333333", linespacing=0.95)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.margins(x=0.04)
    return bars
