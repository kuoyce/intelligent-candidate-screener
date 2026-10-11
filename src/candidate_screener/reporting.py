"""Chart styling shared by the report notebooks: one look, validated colours, thin marks.

Categorical slots 1-3 of the reference palette (blue, orange, aqua) pass the CVD and
normal-vision checks for up to three series. Aqua is below 3:1 contrast on the light
surface, so every chart that uses it also carries direct labels or a table.
"""
from __future__ import annotations

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "axes.axisbelow": True,
        "ytick.color": TEXT_2, "text.color": TEXT, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "font.size": 10, "legend.frameon": False})


def dot_ci(ax, labels, values, lows, highs, color=SERIES[0], ref=None, ref_label=None):
    """Horizontal dot + CI whisker per row, value labelled at the dot."""
    ys = range(len(labels))[::-1]
    for y, v, lo, hi in zip(ys, values, lows, highs):
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([v], [y], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{v:.2f}", (hi, y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TEXT_2, fontsize=9)
    ax.set_yticks(list(ys), labels)
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
        if ref_label:
            ax.set_ylim(-0.6, len(labels) - 0.1)
            ax.annotate(ref_label, (ref, len(labels) - 0.45), xytext=(4, 0),
                        textcoords="offset points", color=TEXT_2, fontsize=8)
