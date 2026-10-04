"""The published anchor: which hypotheses about the reference put the baseline's equal-opportunity gap near 0.270?"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from common import GROUP, load, pd, plots, save, section
from equialgo import evaluate, references
from equialgo.data import LABEL, R_SCORE, split_history
from equialgo.pipeline import BASELINE_SPEC, complete, decide

ANCHOR = evaluate.PUBLISHED_BASELINE_GAP
RATES = np.round(np.arange(0.30, 0.501, 0.02), 2)
IN_BUDGET = [r for r in RATES if evaluate.BUDGET[0] <= r <= evaluate.BUDGET[1]]
TOLERANCE = 2  # standard errors

history, candidates = load()
spec = complete(BASELINE_SPEC)
decisions = np.asarray(decide(spec, history, candidates))
train, test = split_history(history)
held = np.asarray(decide(spec, train, test))
groups = candidates[GROUP].to_numpy()

section("Against the committee's own label (held-out 30% of the history)")
print("true-positive rate per group:", evaluate.true_positive_rates(held, test[LABEL], test[GROUP]).round(3).to_dict())
print(f"equal-opportunity gap {evaluate.equal_opportunity_gap(held, test[LABEL], test[GROUP]):.3f}: "
      f"measured against the committee the baseline looks almost fair; the published gap is {ANCHOR}.")

section("Baseline equal-opportunity gap on the 4,000, by hypothesis (rows) and reference rate (columns)")
grid = pd.DataFrame({rate: {name: evaluate.equal_opportunity_gap(decisions, references.label(name, candidates, rate), groups)
                            for name in references.SCORES} for rate in RATES})
print(grid.round(3).to_string())

section("At a 40% reference rate: gap, standard error, agreement; and the same gap on the held-out history")
rows = {}
for name in references.SCORES:
    label = references.label(name, candidates)
    rows[name] = {"EO gap": grid.loc[name, 0.40], "se": evaluate.bootstrap_se(decisions, label, groups)["eo_gap"],
                  "agreement": evaluate.agreement(decisions, label),
                  "held-out gap": evaluate.equal_opportunity_gap(held, references.label(name, test), test[GROUP])}
at40 = pd.DataFrame(rows).T
at40["distance in se"] = (at40["EO gap"] - ANCHOR) / at40["se"]
print(at40.round(3).to_string())

section(f"Compatibility: some reference rate inside the budget band puts the gap within {TOLERANCE} se of {ANCHOR}")
distance = (grid[IN_BUDGET] - ANCHOR).abs().div(at40["se"], axis=0)
verdict = pd.DataFrame({"closest gap": [grid.loc[n, distance.loc[n].idxmin()] for n in grid.index],
                        "at rate": distance.idxmin(axis=1), "distance in se": distance.min(axis=1),
                        "compatible": distance.min(axis=1) <= TOLERANCE}, index=grid.index)
kept_by_anchor = references.PLAUSIBLE + references.DROPPED_BY_FEEDBACK  # the feedback came later
verdict["in PLAUSIBLE"] = [n in kept_by_anchor for n in verdict.index]
print(verdict.round(3).to_string())
mismatch = verdict.index[verdict["compatible"] != verdict["in PLAUSIBLE"]].tolist()
print("references.PLAUSIBLE matches this rule." if not mismatch else f"references.PLAUSIBLE differs from this rule on: {mismatch}")

section("What the anchor cannot settle: the hours weight (R + w x hours, 40% rate)")
weights = [0, 0.03, 0.06, 0.09, 0.12, 0.145, 0.18, 0.22]
sweep = {}
for w in weights:
    score = candidates[R_SCORE] + w * candidates["hrs"]
    label = (score.rank(method="first", ascending=False) <= round(0.40 * len(score))).astype(int).to_numpy()
    deserving = pd.Series(label).groupby(groups).mean()
    sweep[w] = {"EO gap": evaluate.equal_opportunity_gap(decisions, label, groups),
                "deserving Centre": deserving["Centre"], "deserving Eloignee": deserving["Eloignee"]}
print(pd.DataFrame(sweep).T.rename_axis("w").round(3).to_string())

section("Noise in the reference (sd in R points added to the score, 40% rate, mean of 50 draws)")
rng = np.random.default_rng(0)
rows = {}
for name in ("R1_merit", "R3_merit_hours"):
    for sd in (0, 0.5, 1, 2, 3):
        gaps, agreements = [], []
        for _ in range(50):
            score = references.score(name, candidates) + rng.normal(0, sd, len(candidates))
            label = (score >= score.quantile(0.60)).astype(int).to_numpy()
            gaps.append(evaluate.equal_opportunity_gap(decisions, label, groups))
            agreements.append(evaluate.agreement(decisions, label))
        rows[(name, sd)] = {"EO gap": np.mean(gaps), "baseline agreement": np.mean(agreements)}
print(pd.DataFrame(rows).T.round(3).to_string())

section("Estimated automated points of the baseline under each plausible hypothesis (assumed formulas)")
print(evaluate.against_panel(decisions, candidates, kept_by_anchor).round(3).to_string())

# Figure: one panel per hypothesis, the baseline's gap against the reference rate, with the published value.
plots.apply_style()
order = verdict.sort_values("distance in se").index
fig, axes = plt.subplots(2, 5, figsize=(15, 5.6), sharex=True, sharey=True)
for ax, name in zip(axes.flat, order):
    keep = name in kept_by_anchor
    ax.axvspan(*evaluate.BUDGET, color=plots.GRID, alpha=0.6, linewidth=0)
    ax.axhline(ANCHOR, color=plots.GROUP_COLORS["Eloignee"], linewidth=1.3)
    ax.plot(RATES, grid.loc[name], "-o", markersize=4, color=plots.ACCENT if keep else plots.MUTED,
            markeredgecolor=plots.SURFACE, markeredgewidth=1)
    ax.set_title(references.LABELS[name], fontsize=9)
    ax.text(0.97, 0.05, "conservé" if keep else "écarté", transform=ax.transAxes, ha="right", fontsize=9,
            color=plots.INK if keep else plots.MUTED)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xticks([0.30, 0.40, 0.50])
axes[0, 0].text(0.302, ANCHOR + 0.012, "écart publié : 0,270", fontsize=8.5, color=plots.MUTED)
fig.supxlabel("Part des candidats que l'étalon juge méritants (bande grise : budget de 36 % à 44 %)", fontsize=10.5, color=plots.MUTED)
fig.supylabel("Écart d'égalité des chances du modèle en production", fontsize=10.5, color=plots.MUTED)
fig.suptitle("Quels étalons retrouvent l'écart publié ?", x=0.06, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "a08_ancre_taux.png")
