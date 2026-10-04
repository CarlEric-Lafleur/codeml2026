"""Who gets a grant: the committee, the production model and the corrected model, seen applicant by applicant."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from common import CENTRE, GROUP, GROUPS, REMOTE, load, pd, plots, save, section
from equialgo import audit, references

history, candidates = load()
_, _, production = audit.baseline_decisions(history, candidates)
corrected = audit.retained_decisions(candidates)
production, corrected = production.astype(bool), corrected.astype(bool)
remote = (candidates[GROUP] == REMOTE).to_numpy()

REFUSED = "#c9c8c3"
APPLICANTS = "#c9c8c3"
MODELS = {"Modèle en production": (production, plots.BASELINE_COLOR, "s"),
          "Modèle corrigé": (corrected, plots.ACCENT, "o")}


def pct(value, digits=0):
    return f"{100 * value:.{digits}f} %".replace(".", ",")


# The corrected model's cutoff, in R points: every grant sits on or above it.
merit = candidates["r"] + references.COMMITTEE_HOURS * candidates["hrs"]
cutoff = merit[corrected].min()
assert (merit[~corrected] < cutoff).all(), "the retained model is not a cutoff on R + hours"
print(f"corrected model: granted when R + {references.COMMITTEE_HOURS} x hours >= {cutoff:.2f}")

section("Grant rate per decider and group")
# The committee decided on the 10,000; an equal-sized sample keeps the three rows comparable to the eye.
sample = history.sample(len(candidates), random_state=0)
deciders = [("Comité (historique)", sample, sample["y"].to_numpy().astype(bool)),
            ("Modèle en production", candidates, production),
            ("Modèle corrigé", candidates, corrected)]
print(pd.DataFrame({name: pd.Series(granted).groupby(table[GROUP].to_numpy()).mean()
                    for name, table, granted in deciders}).round(3).to_string())

# Figure 1: every applicant in the R score x hours plane, one row per decider, one column per group.
plots.apply_style()
rng = np.random.default_rng(0)
fig, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=True, sharey=True)
r_range = np.array([candidates["r"].min(), candidates["r"].max()])
for row, (name, table, granted) in zip(axes, deciders):
    jitter = rng.uniform(-0.4, 0.4, len(table))
    for ax, group in zip(row, GROUPS):
        member = (table[GROUP] == group).to_numpy()
        for mask, color, z in ((member & ~granted, REFUSED, 1), (member & granted, plots.GROUP_COLORS[group], 2)):
            ax.scatter(table["r"][mask], (table["hrs"] + jitter)[mask], s=5, color=color, linewidths=0, zorder=z)
        ax.plot(r_range, (cutoff - r_range) / references.COMMITTEE_HOURS, color=plots.INK, linewidth=1.2,
                linestyle="--", zorder=3)
        ax.set_title(f"{name} · {plots.GROUP_NAMES[group].lower()}", fontsize=11)
        ax.text(0.98, 0.95, f"{pct(granted[member].mean())} d'octrois", transform=ax.transAxes, ha="right", va="top",
                fontsize=10.5, color=plots.INK, bbox={"facecolor": plots.SURFACE, "edgecolor": "none", "pad": 2})
        ax.grid(visible=False)
axes[0, 0].set_ylim(-1, candidates["hrs"].max() + 1)
for ax in axes[-1]:
    ax.set_xlabel("Cote R")
for ax in axes[:, 0]:
    ax.set_ylabel("Heures de travail par semaine")
handles = [plt.Line2D([], [], marker="o", linestyle="", color=plots.GROUP_COLORS[g], markersize=7) for g in GROUPS]
handles += [plt.Line2D([], [], marker="o", linestyle="", color=REFUSED, markersize=7),
            plt.Line2D([], [], color=plots.INK, linewidth=1.2, linestyle="--")]
labels = [f"Octroi, {plots.GROUP_NAMES[g].lower()}" for g in GROUPS] + ["Refus", "Seuil du modèle corrigé"]
fig.suptitle("Qui reçoit une bourse : un point par candidat", x=0.01, y=0.995, ha="left", fontweight="bold", fontsize=14)
fig.legend(handles, labels, loc="upper left", ncols=4, bbox_to_anchor=(0.005, 0.975), fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "a16_carte_octrois.png")
plt.close(fig)

section("Grant rate per characteristic, production against corrected (4,000)")
keys = {
    "Cote R": pd.cut(candidates["r"], audit.R_BAND_EDGES, labels=audit.R_BAND_LABELS, right=False),
    "Heures de travail par semaine": pd.cut(candidates["hrs"], [0, 5, 10, 15, 20, 100], right=False,
                                            labels=["< 5", "5–9", "10–14", "15–19", "≥ 20"]),
    "Revenu familial": pd.qcut(candidates["inc"], 4, labels=["quart le plus bas", "2e quart", "3e quart",
                                                             "quart le plus élevé"]),
    "Première génération universitaire": candidates["fg"].map({0: "non", 1: "oui"}),
    "Programme d'études": candidates["prog"],
    "Région": candidates["reg"].astype(pd.CategoricalDtype(
        ["Montreal", "Capitale-Nationale", "Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"])),
}
frame = pd.DataFrame({name: granted for name, (granted, _, _) in MODELS.items()})
rates = {title: frame.groupby(np.asarray(key.astype(str)), sort=False).mean().reindex(
    [str(c) for c in (key.cat.categories if hasattr(key, "cat") else sorted(key.unique()))])
    for title, key in keys.items()}
sizes = {title: pd.Series(np.asarray(key.astype(str))).value_counts() for title, key in keys.items()}
for title, table in rates.items():
    print(f"\n{title}\n{table.assign(n=sizes[title]).round(3).to_string()}")

# Figure 2: one dumbbell panel per characteristic. The R score spans 0 to 100 %; the others share one scale.
fig, axes = plt.subplots(3, 2, figsize=(11, 9.5))
others = pd.concat([t for title, t in rates.items() if title != "Cote R"])
shared = (max(0, others.min().min() - 0.06), others.max().max() + 0.06)
for ax, (title, table) in zip(axes.flat, rates.items()):
    y = np.arange(len(table))[::-1]
    before, after = (table[name] for name in MODELS)
    ax.hlines(y, before, after, color=plots.GRID, linewidth=3, zorder=1)
    for name, (_, color, marker) in MODELS.items():
        ax.scatter(table[name], y, s=70, marker=marker, color=color, edgecolor=plots.SURFACE, linewidth=1.5,
                   zorder=2 + (marker == "o"), label=name)
    ax.axvline(corrected.mean(), color=plots.MUTED, linewidth=1, linestyle=":")
    ax.set_yticks(y, table.index)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_xlim(*((-0.04, 1.04) if title == "Cote R" else shared))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_title(title, fontsize=11)
handles, labels = axes.flat[0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=plots.MUTED, linewidth=1, linestyle=":"))
fig.suptitle("Taux d'octroi selon le profil du candidat, avant et après correction (4 000 candidats)",
             x=0.01, y=0.995, ha="left", fontweight="bold", fontsize=13)
fig.legend(handles, labels + ["Taux d'octroi global"], loc="upper left", ncols=3, bbox_to_anchor=(0.005, 0.97), fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "a16_taux_par_profil.png")
plt.close(fig)

section("Who the grantees are: share of each profile among applicants and among grantees (4,000)")
traits = {
    "Régions éloignées": remote,
    "Première génération universitaire": candidates["fg"].to_numpy() == 1,
    "Revenu familial sous la médiane": (candidates["inc"] < candidates["inc"].median()).to_numpy(),
    "Travaille 15 h ou plus par semaine": (candidates["hrs"] >= 15).to_numpy(),
    "Cote R sous la médiane": (candidates["r"] < candidates["r"].median()).to_numpy(),
}
series = {"Tous les candidats": (np.ones(len(candidates), dtype=bool), APPLICANTS),
          "Boursiers du modèle en production": (production, plots.BASELINE_COLOR),
          "Boursiers du modèle corrigé": (corrected, plots.ACCENT)}
shares = pd.DataFrame({name: {trait: flag[rows].mean() for trait, flag in traits.items()}
                       for name, (rows, _) in series.items()})
print(shares.round(3).to_string())

# Figure 3: grouped bars, one group per trait.
fig, ax = plt.subplots(figsize=(9, 5.6))
y = np.arange(len(shares))[::-1]
height = 0.26
for i, (name, (_, color)) in enumerate(series.items()):
    bars = ax.barh(y + (1 - i) * height, shares[name], height=height - 0.03, color=color, label=name)
    ax.bar_label(bars, labels=[pct(v) for v in shares[name]], padding=3, color=plots.INK, fontsize=9)
ax.set_yticks(y, shares.index)
ax.set_xlim(0, shares.max().max() * 1.15)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.grid(axis="y", visible=False)
ax.set_xlabel("Part des candidats ou des boursiers ayant ce profil")
ax.set_title("Qui sont les boursiers ?", pad=34)
ax.legend(loc="lower left", ncols=3, bbox_to_anchor=(-0.42, 1.0), fontsize=9.5)
save(fig, "a16_profil_boursiers.png")
plt.close(fig)

section("Who changes hands between the production model and the corrected model")
moves = pd.DataFrame({"gagne une bourse": corrected & ~production, "perd une bourse": production & ~corrected})
print(moves.groupby(candidates[GROUP].to_numpy()).sum().to_string())
for name, rows in moves.items():
    print(f"{name}: {rows.sum()} candidats; moyennes: " + candidates.loc[rows.to_numpy(), ["r", "inc", "hrs", "fg"]]
          .mean().round(2).to_string().replace("\n", ", "))
