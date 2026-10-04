"""The 4,000: same distributions as the 10,000? What does the baseline do on them, and under each candidate standard?"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy import stats

from common import (CENTRE, GROUP, MODEL_NUMERIC, REMOTE, STANDARD_LABELS, load, pd, plots, save,
                    section, standards, top_share)
from equialgo import evaluate
from equialgo.pipeline import BASELINE_SPEC, complete, decide

history, candidates = load()

section("Group and region mix")
print(pd.concat({"history": history["reg"].value_counts(normalize=True),
                 "candidates": candidates["reg"].value_counts(normalize=True)}, axis=1).round(3).to_string())
print(pd.concat({"history": history[GROUP].value_counts(normalize=True),
                 "candidates": candidates[GROUP].value_counts(normalize=True)}, axis=1).round(3).to_string())

section("Per group: mean in each file, and Kolmogorov-Smirnov p-value")
rows = []
for group in (CENTRE, REMOTE):
    h, c = history[history[GROUP] == group], candidates[candidates[GROUP] == group]
    for col in MODEL_NUMERIC:
        rows.append({"group": group, "column": col, "history": h[col].mean(), "candidates": c[col].mean(),
                     "KS p": stats.ks_2samp(h[col], c[col]).pvalue})
print(pd.DataFrame(rows).set_index(["group", "column"]).round(3).to_string())
print(f"smallest of {len(rows)} p-values: {min(r['KS p'] for r in rows):.3f} "
      f"(with {len(rows)} tests, one below {0.05:.2f} is expected by chance about {1 - 0.95 ** len(rows):.0%} of the time)")

section("Within-group correlations in the 4,000")
for group, part in candidates.groupby(GROUP):
    corr = part[MODEL_NUMERIC].corr()
    print(f"{group}: largest absolute off-diagonal correlation "
          f"{corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max():.3f} (noise level {2 / np.sqrt(len(part)):.3f})")

section("The baseline (production model trained on all 10,000) on the 4,000")
decisions = np.asarray(decide(complete(BASELINE_SPEC), history, candidates))
print(f"grant rate {decisions.mean():.4f} ({decisions.sum()} grants)")
print("per group:", evaluate.selection_rates(decisions, candidates[GROUP]).round(3).to_dict(),
      f"| parity gap {evaluate.parity_gap(decisions, candidates[GROUP]):.3f}")
print("per region:", evaluate.selection_rates(decisions, candidates["reg"]).round(3).to_dict())
print(f"top 40% R cutoff on the 4,000: {candidates['r'].quantile(0.6):.2f}")

section("Preview of the published anchor: baseline equal-opportunity gap under each candidate standard")
print("The README gives 0.270 against the hidden reference. Assumes it was measured on these 4,000.")
scores = standards(candidates)
rng = np.random.default_rng(0)
rows = []
for name in scores:
    for rate in (0.30, 0.40, 0.50):
        label = top_share(scores[name], rate).to_numpy()
        tpr = evaluate.true_positive_rates(decisions, label, candidates[GROUP])
        row = {"standard": name, "rate": rate, "TPR Centre": tpr[CENTRE], "TPR Eloignee": tpr[REMOTE],
               "EO gap": tpr[CENTRE] - tpr[REMOTE], "agreement": evaluate.agreement(decisions, label)}
        if rate == 0.40:
            boot = []
            for _ in range(300):
                i = rng.integers(0, len(label), len(label))
                boot.append(evaluate.equal_opportunity_gap(decisions[i], label[i], candidates[GROUP].to_numpy()[i]))
            row["gap se"] = np.std(boot)
        rows.append(row)
table = pd.DataFrame(rows).set_index(["standard", "rate"])
print(table.round(3).to_string())
at40 = table.xs(0.40, level="rate")
print("\nat a 40% rate, distance of the gap to 0.270 in standard errors:")
print(((at40["EO gap"] - 0.270) / at40["gap se"]).round(1).sort_values(key=abs).to_string())

# Figure: the baseline's grant rate per region on the 4,000.
plots.apply_style()
by_region = evaluate.selection_rates(decisions, candidates["reg"]).sort_values()
group_of = candidates.groupby("reg")[GROUP].first()
fig, ax = plt.subplots(figsize=(7.5, 3.6))
bars = ax.barh(by_region.index, by_region.values, height=0.6,
               color=[plots.GROUP_COLORS[group_of[r]] for r in by_region.index])
ax.bar_label(bars, labels=[f"{v:.1%}".replace(".", ",").replace("%", " %") for v in by_region.values],
             padding=4, color=plots.INK)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, by_region.max() * 1.18)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Taux d'octroi du modèle en production, 4 000 candidats")
ax.set_title("Le modèle reproduit l'écart du comité sur les nouveaux candidats")
handles = [plt.Rectangle((0, 0), 1, 1, color=plots.GROUP_COLORS[g]) for g in (CENTRE, REMOTE)]
ax.legend(handles, [plots.GROUP_NAMES[g] for g in (CENTRE, REMOTE)], loc="lower right")
save(fig, "a07_baseline_regions.png")

# Figure: the baseline's equal-opportunity gap under each candidate standard, against the published 0.270.
shown = at40.rename(index=STANDARD_LABELS).sort_values("EO gap")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.errorbar(shown["EO gap"], range(len(shown)), xerr=1.96 * shown["gap se"], fmt="o", color=plots.ACCENT,
            ecolor=plots.MUTED, elinewidth=1.2, capsize=3, markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
ax.axvline(0.270, color=plots.GROUP_COLORS[REMOTE], linewidth=1.5)
ax.text(0.270, len(shown) - 0.45, " écart publié : 0,270", color=plots.GROUP_COLORS[REMOTE], va="center", fontsize=9)
ax.set_yticks(range(len(shown)), shown.index)
ax.set_ylim(-0.6, len(shown) - 0.1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Écart d'égalité des chances du modèle en production (étalon à 40 %, IC 95 %)")
ax.set_title("Quels étalons sont compatibles avec l'écart publié ?")
save(fig, "a07_ancre.png")
