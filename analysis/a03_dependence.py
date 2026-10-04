"""Columns together: which depend on each other, and do the links hold inside a group?"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import GROUP, GROUPS, MODEL_NUMERIC, load, pd, plots, save, section, share_hists

history, _ = load()

section("Correlations, both groups pooled")
print(history[MODEL_NUMERIC + ["rem"]].corr().round(3).to_string())

section("Correlations inside each group")
worst = {}
for group, part in history.groupby(GROUP):
    corr = part[MODEL_NUMERIC].corr()
    print(f"\n{group} (n={len(part)})\n{corr.round(3).to_string()}")
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    worst[group] = off.max().max()
print("\nlargest absolute within-group correlation:", {g: round(v, 3) for g, v in worst.items()},
      f"| noise level 2/sqrt(n) = {2 / np.sqrt(4000):.3f}")

section("Program against the other columns")
print("share of each program per group:")
print(pd.crosstab(history["prog"], history[GROUP], normalize="columns").round(3).to_string())
print(f"program x group chi-square p = {stats.chi2_contingency(pd.crosstab(history['prog'], history[GROUP]))[1]:.2g}")
rows = {}
for group, part in history.groupby(GROUP):
    samples = [g for _, g in part.groupby("prog")]
    rows[group] = {col: stats.f_oneway(*[s[col] for s in samples]).pvalue for col in MODEL_NUMERIC}
print("ANOVA p-values across programs, inside each group:")
print(pd.DataFrame(rows).round(3).to_string())

section("Shape of each column inside a group")
rows = []
for group, part in history.groupby(GROUP):
    for col in ["r", "linc", "ldist"]:
        x = part[col]
        rows.append({"group": group, "column": col, "mean": x.mean(), "sd": x.std(),
                     "skew": stats.skew(x), "excess kurtosis": stats.kurtosis(x)})
print(pd.DataFrame(rows).set_index(["group", "column"]).round(3).to_string())
print("\nhours worked: a Poisson count has variance = mean")
print(history.groupby(GROUP)["hrs"].agg(["mean", "var"]).round(2).to_string())
print("\nfirst generation: share per group")
print(history.groupby(GROUP)["fg"].mean().round(3).to_string())

plots.apply_style()
samples = {plots.GROUP_NAMES[g]: history[history[GROUP] == g] for g in GROUPS}
colors = {plots.GROUP_NAMES[g]: plots.GROUP_COLORS[g] for g in GROUPS}
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
share_hists(axes, samples, colors, alpha=0.6)
axes[0, 0].legend(loc="upper left", fontsize=9)
fig.suptitle("Profils des deux groupes", x=0.02, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "a03_profils.png")
