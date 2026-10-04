"""Should the hours bonus stop at a number of hours?

The retained rule gives 0.145 R point per hour worked, without limit. The capped rule
(`rule_r_hours_capped`, experiment e009) counts the hours up to a cap only. Three questions:
did the committee itself stop rewarding hours past some point, what a cap changes on the
4,000, and how each cap fares against the fitted reference and the other hypotheses.
"""
import matplotlib.pyplot as plt

from common import CENTRE, GROUP, REMOTE, load, logit, np, pd, plots, save, section
from equialgo import evaluate, references
from equialgo.allocate import top_k

history, candidates = load()
groups = candidates[GROUP].to_numpy()
RATE, NOISE, WEIGHT = 0.40, 0.5, references.COMMITTEE_HOURS  # the fitted reference: a09
K = int(round(RATE * len(candidates)))
CAPS = [6, 8, 10, 12, 15, 20, 30]
OTHERS = [name for name in references.PLAUSIBLE if name != "R3_merit_hours"]

section("Hours worked per week: share of each group above each cap (history)")
print(pd.DataFrame({cap: history.groupby(GROUP)["hrs"].apply(lambda h: (h > cap).mean()) for cap in CAPS}).round(3).to_string())

section("Did the committee stop rewarding hours past a cap? Its logit with the hours split at the cap (weights in R points)")
linear = logit("y ~ r + linc + hrs + rem", history)
rows = {"no cap (retained)": {"per hour up to the cap": linear.params["hrs"] / linear.params["r"], "log-likelihood": linear.llf}}
for cap in CAPS[:-1]:
    data = history.assign(below=np.minimum(history["hrs"], cap), above=np.maximum(history["hrs"] - cap, 0))
    split = logit("y ~ r + linc + below + above + rem", data)
    only = logit("y ~ r + linc + below + rem", data)
    b = split.params
    rows[f"cap {cap}"] = {"per hour up to the cap": b["below"] / b["r"], "per hour past the cap": b["above"] / b["r"],
                          "se past": split.bse["above"] / b["r"], "z, past = up to": split.t_test("below - above = 0").tvalue.item(),
                          "log-likelihood": split.llf, "log-likelihood, nothing past the cap": only.llf}
print(pd.DataFrame(rows).T.round(3).to_string(na_rep=""))

section("Each cap on the 4,000, against the retained rule, the fitted reference and the other hypotheses")
score = lambda cap: (candidates["r"] + WEIGHT * np.minimum(candidates["hrs"], cap)).to_numpy()
rule = top_k(score(np.inf), K)
noisy = score(np.inf)[None, :] + NOISE * np.random.default_rng(0).normal(size=(30, len(candidates)))
fitted = (noisy >= np.quantile(noisy, 1 - RATE, axis=1, keepdims=True)).astype(int)
rows = {}
for cap in [0] + CAPS:
    d = top_k(score(cap), K)
    panel = evaluate.against_panel(d, candidates, names=OTHERS)
    rows["R alone" if cap == 0 else "no cap (retained)" if cap == 30 else f"cap {cap}"] = {
        "applicants above the cap": int((candidates["hrs"] > cap).sum()), "decisions that differ": int((d != rule).sum()),
        "rate Centre": d[groups == CENTRE].mean(), "rate Eloignee": d[groups == REMOTE].mean(),
        "parity gap": evaluate.parity_gap(d, groups), "accuracy, fitted reference": (fitted == d).mean(),
        "EO gap, fitted reference": np.mean([evaluate.equal_opportunity_gap(d, label, groups) for label in fitted]),
        "worst case points, other hypotheses": panel["points"].min(), "worst": panel["points"].idxmin()[:3]}
table = pd.DataFrame(rows).T
print(table.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"standard error of the accuracy over the 30 noise draws: {(fitted == rule).mean(axis=1).std():.4f}")

section("Who loses a grant under a cap of 10 or 15 hours")
for cap in (10, 15):
    d = top_k(score(cap), K)
    lost, won = candidates[(rule == 1) & (d == 0)], candidates[(rule == 0) & (d == 1)]
    print(f"cap {cap}: {len(lost)} lose (Eloignee {(lost[GROUP] == REMOTE).mean():.0%}, mean hours {lost['hrs'].mean():.1f}, "
          f"mean R {lost['r'].mean():.2f}); {len(won)} gain (Eloignee {(won[GROUP] == REMOTE).mean():.0%}, "
          f"mean hours {won['hrs'].mean():.1f}, mean R {won['r'].mean():.2f})")

# Figure: grant rate per group as the cap moves.
plots.apply_style()
caps = np.arange(0, 27)
rates = {g: [top_k(score(cap), K)[groups == g].mean() for cap in caps] for g in (CENTRE, REMOTE)}
fig, ax = plt.subplots(figsize=(8, 4.6))
for g in (CENTRE, REMOTE):
    ax.plot(caps, rates[g], marker="o", markersize=3, color=plots.GROUP_COLORS[g], label=plots.GROUP_NAMES[g])
ax.set_xlabel("Plafond des heures comptées (heures par semaine)")
ax.set_ylabel("Taux d'octroi")
ax.set_title("Plafonner les heures comptées rouvre l'écart entre les groupes")
ax.legend()
save(fig, "a17_plafond_heures.png")
