"""Do ensembles of models trained on cleaned decisions (R score and hours only) do better than one model, or than the rule?

Every model sees the R score and the hours worked, nothing else, and is trained on the
history with the biased decisions dropped, down-weighted or relabelled (targets.DEBIASED).
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np

from common import GROUP, load, pd, plots, save, section
from equialgo import evaluate, references
from equialgo.allocate import top_k
from equialgo.data import WEIGHT
from equialgo.pipeline import complete
from equialgo.scorers import ENSEMBLE_MEMBERS, SCORERS, rank_average
from equialgo.targets import DEBIASED, training_set

history, candidates = load()
groups = candidates[GROUP]
K = int(round(0.40 * len(candidates)))
rule_score = references.score("R3_merit_hours", candidates).to_numpy()
rule = top_k(rule_score, K)
# The panel without R + hours: every model here aims at it, so it would flatter them all.
OTHERS = [name for name in references.PLAUSIBLE if name != "R3_merit_hours"]


def trained_score(target, scorer, seed=42):
    spec = complete({"target": target, "features": "merit_hours", "scorer": scorer, "seed": seed})
    train, y, weights = training_set(target, history)
    if weights is not None:
        train = train.assign(**{WEIGHT: weights})
    return SCORERS[scorer](train, y, candidates, spec)


def measure(score):
    d = top_k(score, K)
    panel = evaluate.against_panel(d, candidates, names=OTHERS)
    return {"parity gap": evaluate.parity_gap(d, groups), "same decision as the rule": evaluate.agreement(d, rule),
            "decisions that differ": int((d != rule).sum()), "worst case points, other hypotheses": panel["points"].min()}


scores = {(t, s): trained_score(t, s) for t, s in itertools.product(DEBIASED, ENSEMBLE_MEMBERS)}

section("Single models: one cleaning method, one estimator (R score and hours, 40%)")
single = pd.DataFrame({key: measure(score) for key, score in scores.items()}).T
single.index.names = ["target", "scorer"]
print(single.round(3).to_string())

section("Ensembles (rank average of the members)")
blends = {}
for target in DEBIASED:
    blends[f"3 estimators, {target}"] = rank_average([scores[target, s] for s in ENSEMBLE_MEMBERS])
for scorer in ENSEMBLE_MEMBERS:
    blends[f"5 cleaning methods, {scorer}"] = rank_average([scores[t, scorer] for t in DEBIASED])
blends["all 15 models"] = rank_average(list(scores.values()))
blends["forest + boosting, 5 cleaning methods"] = rank_average(
    [scores[t, s] for t in DEBIASED for s in ("gradient_boosting", "random_forest")])
ensembles = pd.DataFrame({name: measure(score) for name, score in blends.items()}).T
print(ensembles.round(3).to_string())
reference = pd.DataFrame({"rule R + 0.145 x hours (retained)": measure(rule_score)}).T
print(reference.round(3).to_string(header=False))

section("Does averaging help? Members against their ensemble")
rows = {}
for name, members in {**{f"3 estimators, {t}": [(t, s) for s in ENSEMBLE_MEMBERS] for t in DEBIASED},
                      **{f"5 cleaning methods, {s}": [(t, s) for t in DEBIASED] for s in ENSEMBLE_MEMBERS},
                      "all 15 models": list(scores)}.items():
    member_rows = single.loc[members]
    rows[name] = {"members: mean agreement with the rule": member_rows["same decision as the rule"].mean(),
                  "members: best": member_rows["same decision as the rule"].max(),
                  "ensemble": ensembles.loc[name, "same decision as the rule"],
                  "members: mean |parity gap|": member_rows["parity gap"].abs().mean(),
                  "ensemble |parity gap|": abs(ensembles.loc[name, "parity gap"])}
print(pd.DataFrame(rows).T.round(3).to_string())

section("Is the forest's disagreement with the rule signal or noise? Same forest, five seeds, soft weights")
seeds = {seed: trained_score("weight_flipped", "random_forest", seed) for seed in range(5)}
decisions = {seed: top_k(score, K) for seed, score in seeds.items()}
pairs = [(decisions[a] != decisions[b]).mean() for a, b in itertools.combinations(decisions, 2)]
print(f"two seeds differ on {np.mean(pairs):.1%} of decisions; one seed differs from the rule on "
      f"{np.mean([(d != rule).mean() for d in decisions.values()]):.1%}; "
      f"the five seeds averaged differ from the rule on {(top_k(rank_average(list(seeds.values())), K) != rule).mean():.1%}")

section("Where the full ensemble and the rule disagree")
d = top_k(blends["all 15 models"], K)
differ = d != rule
table = candidates[differ].assign(granted_by=np.where(d[differ] == 1, "ensemble only", "rule only"))
print(table.groupby([GROUP, "granted_by"]).agg(n=("r", "size"), r=("r", "mean"), hrs=("hrs", "mean")).round(2).to_string())
margin = np.abs(rule_score - np.sort(rule_score)[-K])
print(f"distance to the rule's cutoff, in R points: median {np.median(margin[differ]):.2f} where they differ, "
      f"{np.median(margin):.2f} over all applicants")

# Figure: each single model and each ensemble, parity gap against agreement with the rule.
plots.apply_style()
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.scatter(single["same decision as the rule"], single["parity gap"], s=45, color=plots.MUTED, label="Un modèle seul")
ax.scatter(ensembles["same decision as the rule"], ensembles["parity gap"], s=70, marker="D", color=plots.ACCENT,
           edgecolor=plots.SURFACE, linewidth=1.2, label="Ensemble")
ax.scatter([1], [reference["parity gap"].iloc[0]], s=110, marker="*", color=plots.INK, label="Règle retenue (cote R + heures)")
ax.axhline(0, color=plots.MUTED, linewidth=1, linestyle=":")
ax.set_xlabel("Part des 4 000 décisions identiques à la règle retenue")
ax.set_ylabel("Écart de parité (Centre - Éloignée)")
ax.set_title("Plus on moyenne de modèles, plus on retombe sur la règle retenue")
ax.legend(loc="best")
save(fig, "a13_ensembles.png")
