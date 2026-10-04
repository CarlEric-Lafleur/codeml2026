"""Which individual decisions of the production model are biased: counterfactual flip, merit benchmark, SHAP by group."""
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

from common import CENTRE, GROUP, GROUPS, REMOTE, load, pd, plots, save, section
from equialgo import references
from equialgo.features import ALL, encode
from equialgo.pipeline import BASELINE_SPEC, complete, decide

history, candidates = load()
remote = (candidates[GROUP] == REMOTE).to_numpy()
LABELS = {"cote_r_equivalent": "Cote R", "revenu_familial_estime": "Revenu familial",
          "heures_travail_semaine": "Heures de travail", "distance_domicile_campus_km": "Distance au campus",
          "premiere_generation_universitaire": "Première génération", "programme_etudes": "Programme",
          "region_administrative": "Région", "code_postal_3": "Code postal (3 car.)"}
LOCATION = ["region_administrative", "code_postal_3", "distance_domicile_campus_km"]

# The production model, refitted here to keep the fitted forest (the scorer returns only probabilities).
X, Xt = encode("all", history, candidates)
forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=42).fit(X, history["y"])


def probability(table):
    return forest.predict_proba(encode("all", history, table)[1])[:, 1]


p = probability(candidates)
granted = p >= 0.5
assert (granted == decide(complete(BASELINE_SPEC), history, candidates)).all(), "not the production model"
print(f"production model on the 4,000: grant rate {granted.mean():.4f}; "
      f"Centre {granted[~remote].mean():.3f}, Eloignee {granted[remote].mean():.3f}")

section("Counterfactual flip: same applicant, location columns of the other group")
# Typical location of each group: its largest region, that region's most common prefix, the group's median distance.
typical = {}
for group, part in history.groupby(GROUP):
    region = part["reg"].mode()[0]
    typical[group] = {"region_administrative": region,
                      "code_postal_3": part.loc[part["reg"] == region, "cp"].mode()[0],
                      "distance_domicile_campus_km": part["dist"].median()}
print(pd.DataFrame(typical).to_string())

swaps = {"region": LOCATION[:1], "postal prefix": LOCATION[1:2], "distance": LOCATION[2:],
         "region + postal prefix": LOCATION[:2], "region + postal prefix + distance": LOCATION}
rows, moved = [], {}
for group, other in ((REMOTE, CENTRE), (CENTRE, REMOTE)):
    member = candidates[GROUP] == group
    for name, columns in swaps.items():
        swapped = candidates[member].assign(**{c: typical[other][c] for c in columns})
        q = probability(swapped)
        before = granted[member.to_numpy()]
        moved[group, name] = pd.Series(q, index=candidates.index[member])
        rows.append({"group": group, "swapped to the other group": name, "n": int(member.sum()),
                     "grant rate before": before.mean(), "after": (q >= 0.5).mean(),
                     "refused -> granted": int(((q >= 0.5) & ~before).sum()),
                     "granted -> refused": int(((q < 0.5) & before).sum()),
                     "mean change in probability": (q - p[member.to_numpy()]).mean()})
print(pd.DataFrame(rows).set_index(["group", "swapped to the other group"]).round(3).to_string())

as_centre = moved[REMOTE, "region + postal prefix + distance"]
flip = pd.Series(False, index=candidates.index)
flip[as_centre.index] = (as_centre >= 0.5).to_numpy() & ~granted[remote]
print(f"\nremote applicants refused because of where they live: {flip.sum()} of {remote.sum()} "
      f"({flip.sum() / remote.sum():.1%}), {flip.sum() / (remote & ~granted).sum():.1%} of the remote refusals")
print("their profile against the other remote refusals (means):")
refused_remote = candidates[remote & ~granted].assign(flipped=flip[remote & ~granted])
print(refused_remote.groupby("flipped")[["r", "inc", "hrs", "dist", "fg"]].mean().round(2).to_string())
print("R score of the flipped: " + ", ".join(f"{q:.0%} {v:.2f}" for q, v in
                                              candidates.loc[flip, "r"].quantile([0.05, 0.5, 0.95]).items()))

section("Merit benchmark: who the production model refuses and a merit standard funds (standard at 40%)")
rows = []
for name in ("R3_merit_hours", "R1_merit"):
    deserving = references.label(name, candidates).astype(bool)
    for group, member in ((CENTRE, ~remote), (REMOTE, remote)):
        missed = member & deserving & ~granted
        rows.append({"standard": name, "group": group, "deserving": int((member & deserving).sum()),
                     "refused by the model (false negatives)": int(missed.sum()),
                     "share of the deserving": missed.sum() / (member & deserving).sum(),
                     "granted, not deserving": int((member & ~deserving & granted).sum()),
                     "false negatives that the flip test recovers": int((missed & flip).sum())})
print(pd.DataFrame(rows).set_index(["standard", "group"]).round(3).to_string())
deserving = references.label("R3_merit_hours", candidates).astype(bool)
print(f"\nflipped remote applicants: {flip.sum()}, of which deserving under R + hours: {(flip & deserving).sum()}, "
      f"under R alone: {(flip & references.label('R1_merit', candidates).astype(bool)).sum()}")

section("SHAP on the production model: mean contribution per column, in probability points")
values = shap.TreeExplainer(forest).shap_values(Xt.astype(float), check_additivity=False)
values = values[..., 1] if values.ndim == 3 else values[1]
source = [next(c for c in ALL if col == c or col.startswith(c + "_")) for col in Xt.columns]
contrib = pd.DataFrame(values, columns=Xt.columns, index=candidates.index).T.groupby(source).sum().T[ALL]
contrib["location (region + postal + distance)"] = contrib[LOCATION].sum(axis=1)
samples = {"all applicants": np.ones(len(candidates), dtype=bool), "refused": ~granted, "flipped": flip.to_numpy()}
table = pd.concat({name: (100 * contrib[rows_]).groupby(candidates.loc[rows_, GROUP]).mean().T
                   for name, rows_ in samples.items()}, axis=1)
table["all applicants", "gap"] = table["all applicants", REMOTE] - table["all applicants", CENTRE]
print(table.round(1).to_string())
base = p.mean() - contrib[ALL].sum(axis=1).mean()
print(f"\nmean probability {p.mean():.3f} = base {base:.3f} + contributions; group gap in mean probability "
      f"{p[remote].mean() - p[~remote].mean():+.3f} = sum of the per-column gaps above")
location = contrib["location (region + postal + distance)"]
print(f"location contribution, remote applicants: mean {100 * location[remote].mean():+.1f} points, "
      f"from {100 * location[remote].min():+.1f} to {100 * location[remote].max():+.1f}; "
      f"Centre: mean {100 * location[~remote].mean():+.1f}")
owed = remote & ~granted & (p - location.to_numpy() + location[~remote].mean() >= 0.5)
print(f"remote refusals that pass 0.5 once the location contribution is set to the Centre mean: {owed.sum()} "
      f"({(owed & flip.to_numpy()).sum()} of them are also flipped by the counterfactual test)")

# Figure: mean contribution of each column to the production model's probability, per group.
shown = (100 * contrib[ALL]).groupby(candidates[GROUP]).mean().T.rename(index=LABELS)
shown = shown.loc[(shown[REMOTE] - shown[CENTRE]).sort_values(ascending=False).index]
plots.apply_style()
fig, ax = plt.subplots(figsize=(8, 4.6))
y = np.arange(len(shown))
for offset, group in zip((0.2, -0.2), GROUPS):
    bars = ax.barh(y + offset, shown[group], height=0.38, color=plots.GROUP_COLORS[group], label=plots.GROUP_NAMES[group])
    ax.bar_label(bars, labels=[f"{v:+.1f}".replace(".", ",") for v in shown[group]], padding=3, color=plots.INK, fontsize=8)
ax.axvline(0, color=plots.MUTED, linewidth=1)
ax.set_yticks(y, shown.index)
ax.set_xlim(shown.min().min() - 2.5, shown.max().max() + 2.5)
ax.grid(axis="y", visible=False)
ax.legend(loc="lower left")
ax.set_xlabel("Contribution moyenne à la probabilité d'octroi (SHAP, points de pourcentage)")
ax.set_title("Par où le modèle en production pénalise les régions éloignées")
save(fig, "a12_shap_groupes.png")
