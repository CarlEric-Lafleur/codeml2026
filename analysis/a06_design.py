"""The synthetic design: rebuild the generator from the findings, check it, and see what it leaves open for the reference."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from scipy.special import expit

from common import (CENTRE, GROUP, REMOTE, STANDARD_LABELS, committee_fit, group_dots, load, pd, plots, save,
                    section, standards, top_share)

history, _ = load()
fit = committee_fit(history)

section("Reconstructed generator: independent columns inside each group, then the committee logit")
params = history.groupby(GROUP).agg(r_mean=("r", "mean"), r_sd=("r", "std"), linc_mean=("linc", "mean"),
                                    linc_sd=("linc", "std"), hrs_mean=("hrs", "mean"), fg=("fg", "mean"))
print(params.round(3).to_string())

rng = np.random.default_rng(0)
sims = []
for _ in range(200):
    parts = []
    for group, n in history[GROUP].value_counts().items():
        g = params.loc[group]
        parts.append(pd.DataFrame({GROUP: group, "r": rng.normal(g.r_mean, g.r_sd, n),
                                   "linc": rng.normal(g.linc_mean, g.linc_sd, n),
                                   "hrs": rng.poisson(g.hrs_mean, n), "rem": int(group == REMOTE)}))
    sim = pd.concat(parts, ignore_index=True)
    sim["y"] = rng.random(len(sim)) < expit(fit.predict(sim, which="linear"))
    band = pd.cut(sim["r"], [0, 26, 28, 30, 100], labels=["< 26", "26-28", "28-30", ">= 30"])
    stat = sim.groupby(GROUP)["y"].mean().to_dict()
    stat.update({f"{b} {g}": v for (b, g), v in sim.groupby([band, GROUP], observed=True)["y"].mean().items()})
    sims.append(stat)
sims = pd.DataFrame(sims)
band = pd.cut(history["r"], [0, 26, 28, 30, 100], labels=["< 26", "26-28", "28-30", ">= 30"])
actual = history.groupby(GROUP)["y"].mean().to_dict()
actual.update({f"{b} {g}": v for (b, g), v in history.groupby([band, GROUP], observed=True)["y"].mean().items()})
check = pd.DataFrame({"actual": actual, "simulated mean": sims.mean(), "sim 2.5%": sims.quantile(0.025),
                      "sim 97.5%": sims.quantile(0.975)})
check["inside"] = check["actual"].between(check["sim 2.5%"], check["sim 97.5%"])
print("\ncommittee grant rate, actual against 200 simulations of the reconstructed generator:")
print(check.round(3).to_string())

section("Candidate standards at a 40% rate, on the history: who they call deserving")
scores = standards(history)
labels = scores.apply(top_share)
rows = {}
for name in labels:
    rate = labels[name].groupby(history[GROUP]).mean()
    rows[name] = {"rate Centre": rate[CENTRE], "rate Eloignee": rate[REMOTE], "gap": rate[CENTRE] - rate[REMOTE],
                  "agreement with committee": (labels[name] == history["y"]).mean(),
                  "committee TPR Centre": history.loc[(labels[name] == 1) & (history["rem"] == 0), "y"].mean(),
                  "committee TPR Eloignee": history.loc[(labels[name] == 1) & (history["rem"] == 1), "y"].mean()}
table = pd.DataFrame(rows).T
table["committee EO gap"] = table["committee TPR Centre"] - table["committee TPR Eloignee"]
print(table.round(3).to_string())

section("How much the candidate standards disagree with each other (share of applicants labelled differently)")
names = list(labels)
disagreement = pd.DataFrame([[(labels[a] != labels[b]).mean() for b in names] for a in names],
                            index=names, columns=range(1, len(names) + 1))
print(disagreement.round(3).to_string())

section("R score cutoff of the top 40%, and where it leaves each group")
cut = history["r"].quantile(0.6)
print(f"cutoff {cut:.2f}; share above it per group: {(history['r'] >= cut).groupby(history[GROUP]).mean().round(3).to_dict()}")
print(f"hours weight the committee used: {fit.params['hrs'] / fit.params['r']:.3f} R points per hour; "
      f"mean hours gap {params.loc[REMOTE, 'hrs_mean'] - params.loc[CENTRE, 'hrs_mean']:.2f} h, "
      f"worth {(params.loc[REMOTE, 'hrs_mean'] - params.loc[CENTRE, 'hrs_mean']) * fit.params['hrs'] / fit.params['r']:.2f} R points; "
      f"mean R gap {params.loc[CENTRE, 'r_mean'] - params.loc[REMOTE, 'r_mean']:.2f}")

# Figure: share of each group a candidate standard would call deserving, at a 40% rate overall.
plots.apply_style()
shown = table.rename(columns={"rate Centre": CENTRE, "rate Eloignee": REMOTE}).rename(index=STANDARD_LABELS)
shown = shown.sort_values("gap")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
group_dots(ax, shown, "Part du groupe jugée méritante (40 % au total)")
ax.axvline(0.40, color=plots.MUTED, linewidth=1, linestyle=":")
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_title("Selon l'étalon retenu, le groupe jugé méritant change")
ax.legend(loc="upper right")
save(fig, "a06_etalons.png")
