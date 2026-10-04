"""What the committee ignored: columns that add nothing to its decision once R, income, hours and group are in."""
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats

from common import COMMITTEE_FORMULA, GROUP, GROUPS, committee_fit, load, logit, pd, plots, save, section

history, _ = load()
base = committee_fit(history)

section("Likelihood-ratio test of adding each column to the committee model")
additions = {
    "first generation": "fg",
    "program": "C(prog)",
    "distance (log)": "ldist",
    "distance (km)": "dist",
    "region inside the group": "C(reg)",
    "postal prefix inside the group": "C(cp)",
    "first generation x group": "fg + fg:rem",
}
rows = []
for name, term in additions.items():
    formula = COMMITTEE_FORMULA.replace("+ rem", f"+ {term}") if "C(reg)" in term or "C(cp)" in term else f"{COMMITTEE_FORMULA} + {term}"
    m = logit(formula, history)
    extra = int(m.df_model - base.df_model)
    rows.append({"added": name, "extra terms": extra, "LR": 2 * (m.llf - base.llf),
                 "p": stats.chi2.sf(2 * (m.llf - base.llf), extra)})
print(pd.DataFrame(rows).set_index("added").round(3).to_string())

section("Raw committee grant rate by first generation and by program, per group")
print(history.pivot_table(index="fg", columns=GROUP, values="y", aggfunc=["mean", "size"]).round(3).to_string())
print(history.pivot_table(index="prog", columns=GROUP, values="y", aggfunc=["mean", "size"]).round(3).to_string())

section("Raw committee grant rate by income quintile and by hours band, per group")
print(history.pivot_table(index=pd.qcut(history["inc"], 5), columns=GROUP, values="y", aggfunc="mean", observed=True).round(3).to_string())
print(history.pivot_table(index=pd.cut(history["hrs"], [0, 7, 10, 13, 16, 30]), columns=GROUP, values="y", aggfunc="mean", observed=True).round(3).to_string())

section("Who are the first-generation applicants?")
print(history.groupby([GROUP, "fg"])[["r", "inc", "hrs", "dist", "y"]].mean().round(2).to_string())

# Figure: what moves the committee's grant rate inside each group, and what does not.
plots.apply_style()
panels = [
    (pd.qcut(history["inc"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]), "Quintile de revenu familial"),
    (pd.cut(history["hrs"], [0, 7, 10, 13, 16, 30], labels=["≤ 7", "8–10", "11–13", "14–16", "≥ 17"]),
     "Heures de travail par semaine"),
    (history["fg"].map({0: "Non", 1: "Oui"}), "Première génération universitaire"),
    (history["prog"], "Programme d'études"),
]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey=True)
for ax, (key, label) in zip(axes.flat, panels):
    table = history.pivot_table(index=key, columns=GROUP, values="y", aggfunc="mean", observed=True)
    for group in GROUPS:
        ax.plot(table.index.astype(str), table[group], "-o", color=plots.GROUP_COLORS[group],
                label=plots.GROUP_NAMES[group], markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(label)
    ax.set_ylim(0, 0.8)
    if label.startswith("Programme"):
        ax.tick_params(axis="x", labelsize=8)
axes[0, 0].legend(loc="upper left")
axes[0, 0].set_ylabel("Taux d'octroi du comité")
axes[1, 0].set_ylabel("Taux d'octroi du comité")
fig.suptitle("Le comité suit le revenu et les heures, pas la première génération ni le programme",
             x=0.07, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "a05_colonnes_ignorees.png")
