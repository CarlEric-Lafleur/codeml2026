"""Shared figure style and the Pareto plot. Figure text is in French: it goes on the slides."""
import os

import matplotlib

# Scripts draw off-screen; a notebook kernel sets MPLBACKEND itself and keeps its inline backend.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import CENTRE, FIGURES_DIR, REMOTE

GROUP_COLORS = {CENTRE: "#2a78d6", REMOTE: "#eb6834"}
GROUP_NAMES = {CENTRE: "Grands centres", REMOTE: "Régions éloignées"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ACCENT = "#2a78d6"
BASELINE_COLOR = "#52514e"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "lines.linewidth": 2,
        "lines.markersize": 7,
        "legend.frameon": False,
    })


def save(fig, name):
    """Save a figure under figures/ and return its path."""
    path = FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


# Comparison methods on the Pareto figure: (colour, marker). Shapes repeat what colour says.
COMPARISON_STYLES = [("#eb6834", "^"), ("#1baf7a", "D")]


def pareto(sweep, baseline, chosen_strength, panels, title, xlabel, ylabel, comparisons=(), points=(),
           name="pareto.png"):
    """Equity gap against utility along the strength sweep, one panel per way of measuring them.

    `sweep` is a table with one row per strength; `baseline` a dict with the same metrics;
    `panels` a list of (x column, y column, panel title). `comparisons` is a list of
    (label, sweep table) drawn as thinner lines; `points` a list of (label, metrics dict)
    drawn as single markers. The gap is drawn in absolute value.
    """
    apply_style()
    columns = min(3, len(panels))
    rows = -(-len(panels) // columns)
    height = 3.6 * rows + (0.9 if rows == 1 else 0)  # a single row still needs room for the legend
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, height), squeeze=False)
    chosen = sweep[sweep["strength"] == chosen_strength].iloc[0]
    for ax, (x, y, panel_title) in zip(axes.flat, panels):
        for (label, other), (color, marker) in zip(comparisons, COMPARISON_STYLES):
            ax.plot(other[x].abs(), other[y], "-", marker=marker, color=color, linewidth=1.3, markersize=5.5,
                    label=label, markeredgecolor=SURFACE, markeredgewidth=1)
        ax.plot(sweep[x].abs(), sweep[y], "-o", color=ACCENT, label="Modèle retenu, correction de 0 à 1",
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        for strength in (sweep["strength"].min(), sweep["strength"].max()):
            row = sweep[sweep["strength"] == strength].iloc[0]
            ax.annotate(f"{strength:g}", (abs(row[x]), row[y]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=9, color=MUTED)
        ax.scatter([abs(chosen[x])], [chosen[y]], s=220, facecolors="none", edgecolors=INK,
                   linewidths=1.8, zorder=4, label="Point retenu")
        ax.scatter([abs(baseline[x])], [baseline[y]], marker="s", s=70, color=BASELINE_COLOR,
                   zorder=3, label="Modèle en production")
        for (label, metrics), marker in zip(points, ["X", "P"]):
            ax.scatter([abs(metrics[x])], [metrics[y]], marker=marker, s=70, color=MUTED, zorder=3, label=label)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlim(left=min(0, ax.get_xlim()[0]))
        ax.margins(x=0.1, y=0.18)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    first = [i for i, label in enumerate(labels) if label.startswith(("Modèle retenu", "Point retenu"))]
    order = first + [i for i in range(len(labels)) if i not in first]
    handles, labels = [handles[i] for i in order], [labels[i] for i in order]
    fig.suptitle(title, x=0.01, y=1 - 0.1 / height, ha="left", fontweight="bold", fontsize=13)
    fig.legend(handles, labels, loc="upper left", ncols=3, bbox_to_anchor=(0.005, 1 - 0.33 / height), fontsize=9.5)
    fig.supxlabel(xlabel, fontsize=10.5, color=MUTED)
    fig.supylabel(ylabel, fontsize=10.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / height))
    path = save(fig, name)
    plt.close(fig)
    return path
