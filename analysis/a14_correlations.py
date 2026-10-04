"""Correlation matrix of the columns: pooled, then inside each group.

Pearson on the numeric columns (income and distance in logs), the group indicator and the
committee's decision. The nominal columns (program, region, postal prefix) have no Pearson
correlation; a03 tests them with chi-square and ANOVA.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from common import GROUP, GROUPS, MODEL_NUMERIC, load, plots, save, section

LABELS = {"r": "Cote R", "linc": "Revenu (log)", "hrs": "Heures de travail", "ldist": "Distance (log)",
          "fg": "Première génération", "rem": "Région éloignée", "y": "Octroi (comité)"}
# Diverging: one hue per sign, neutral midpoint.
CMAP = LinearSegmentedColormap.from_list("corr", [plots.GROUP_COLORS[GROUPS[1]], "#ecebe7", plots.ACCENT])

history, _ = load()
columns = MODEL_NUMERIC + ["y"]
matrices = {"Les deux groupes": history[columns[:-1] + ["rem", "y"]].corr()}
for group in GROUPS:
    part = history[history[GROUP] == group]
    matrices[f"{plots.GROUP_NAMES[group]} (n = {len(part)})"] = part[columns].corr()

for title, corr in matrices.items():
    section(title)
    print(corr.round(3).to_string())


def heatmap(ax, corr, title):
    image = ax.imshow(corr, cmap=CMAP, vmin=-1, vmax=1)
    names = [LABELS[c] for c in corr.columns]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i, row in enumerate(corr.to_numpy()):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.5,
                    color=plots.SURFACE if abs(value) > 0.6 else plots.INK)
    # Surface-coloured gaps between the cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(names))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(names))], minor=True)
    ax.grid(visible=False)
    ax.grid(which="minor", color=plots.SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(title, fontsize=11)
    return image


plots.apply_style()
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), layout="constrained",
                         width_ratios=[len(m) for m in matrices.values()])
for ax, (title, corr) in zip(axes, matrices.items()):
    image = heatmap(ax, corr, title)
fig.suptitle("Corrélations entre les colonnes", x=0.01, ha="left", fontweight="bold", fontsize=13)
fig.colorbar(image, ax=axes, shrink=0.7, pad=0.015, label="Corrélation de Pearson").outline.set_visible(False)
save(fig, "a14_correlations.png")
