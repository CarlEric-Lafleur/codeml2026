"""The committee's decision: what moves it, in what shape, and what the remote penalty is worth."""
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from common import COMMITTEE_FORMULA, GROUP, GROUPS, coef_table, committee_fit, load, logit, pd, plots, save, section

history, _ = load()
fit = committee_fit(history)
p = fit.predict(history)

section(f"Logit of the committee's decision: {COMMITTEE_FORMULA}")
print(coef_table(fit).to_string())
print("\nin R-point equivalents (coefficient / R coefficient):")
print((fit.params.drop("Intercept") / fit.params["r"]).round(3).to_string())
sd = history[["r", "linc", "hrs"]].std()
print("\neffect of one standard deviation, in R points:")
print((fit.params[sd.index] * sd / fit.params["r"]).round(2).to_string())

section("Shape: is it additive?")
probit = smf.probit(COMMITTEE_FORMULA, history).fit(disp=0)
print(f"log-likelihood: logit {fit.llf:.1f}, probit {probit.llf:.1f}")
alternatives = {
    "income in dollars instead of log": "y ~ r + inc + hrs + rem",
    "squares and R x income": "y ~ r + linc + hrs + rem + I(r**2) + I(linc**2) + I(hrs**2) + r:linc",
    "every slope different per group": "y ~ (r + linc + hrs) * rem",
    "five regions instead of two groups": "y ~ r + linc + hrs + C(reg)",
}
rows = []
for name, formula in alternatives.items():
    m = logit(formula, history)
    extra = int(m.df_model - fit.df_model)
    lr = 2 * (m.llf - fit.llf)
    rows.append({"alternative": name, "AIC": m.aic, "extra terms": extra,
                 "LR test p": stats.chi2.sf(lr, extra) if extra > 0 else np.nan})
print(f"base model AIC {fit.aic:.1f}")
print(pd.DataFrame(rows).set_index("alternative").round(3).to_string())

section("Is the remote penalty constant? Group coefficient re-estimated on slices")
slices = {"R band": pd.cut(history["r"], [0, 27, 28.5, 30, 100], labels=["< 27", "27-28.5", "28.5-30", ">= 30"]),
          "income tercile": pd.qcut(history["inc"], 3, labels=["low", "middle", "high"]),
          "hours tercile": pd.qcut(history["hrs"], 3, labels=["low", "middle", "high"]),
          "first generation": history["fg"]}
rows = []
for name, key in slices.items():
    for value, part in history.groupby(key, observed=True):
        m = logit(COMMITTEE_FORMULA, part)
        rows.append({"slice": name, "value": value, "n": len(part), "group coef": m.params["rem"], "se": m.bse["rem"]})
print(pd.DataFrame(rows).set_index(["slice", "value"]).round(2).to_string())
by_region = logit("y ~ r + linc + hrs + C(reg, Treatment('Montreal'))", history)
print("\nper region, against Montreal:")
print(coef_table(by_region).filter(like="reg", axis=0).to_string())

section("How much of the decision does the model explain?")
ceiling = np.maximum(p, 1 - p).mean()
print(f"AUC {roc_auc_score(history['y'], p):.4f}; accuracy at 0.5: {((p > 0.5) == history['y']).mean():.4f}; "
      f"best accuracy any model could reach if this logit is the truth: {ceiling:.4f}")
print(f"the unexplained part behaves like logistic noise of sd {np.pi / np.sqrt(3) / fit.params['r']:.2f} R points")
flexible = pd.get_dummies(history[["r", "inc", "hrs", "dist", "fg", "prog", "reg", "cp"]]).astype(float)
p_gb = cross_val_predict(GradientBoostingClassifier(random_state=0), flexible, history["y"], cv=5,
                         method="predict_proba")[:, 1]
print(f"gradient boosting on every column, 5-fold: AUC {roc_auc_score(history['y'], p_gb):.4f}, "
      f"accuracy {((p_gb > 0.5) == history['y']).mean():.4f} -> a flexible model finds nothing the logit misses")
decile = pd.qcut(p, 10, duplicates="drop")
calibration = history.groupby(decile, observed=True).agg(predicted=("y", lambda s: p[s.index].mean()), observed=("y", "mean"))
print("calibration by decile of predicted probability:")
print(calibration.round(3).to_string())

section("Borderline applicants: predicted probability between 10% and 90%")
border = (p > 0.1) & (p < 0.9)
print(f"{border.mean():.1%} of applications; per group:")
print(history.assign(borderline=border).groupby(GROUP)["borderline"].mean().round(3).to_string())
print("their profile against the rest:")
print(history.groupby(border)[["r", "inc", "hrs", "fg", "rem", "y"]].mean().round(2).to_string())
index = history["r"] + (fit.params["linc"] * history["linc"] + fit.params["hrs"] * history["hrs"]) / fit.params["r"]
print("R score at which the committee grants one time in two, for the median income and hours of each group:")
for group, part in history.groupby(GROUP):
    rem = int(part["rem"].iloc[0])
    other = fit.params["linc"] * part["linc"].median() + fit.params["hrs"] * part["hrs"].median() + fit.params["rem"] * rem
    print(f"  {group}: {-(fit.params['Intercept'] + other) / fit.params['r']:.2f}")

section("What the penalty is worth: remote applicants re-scored by the same model")
remote = history[history["rem"] == 1]
centre = history[history["rem"] == 0]
scenarios = {"as recorded": remote["y"].mean(),
             "group penalty removed": fit.predict(remote.assign(rem=0)).mean()}
no_income = logit("y ~ r + hrs + rem", history)
scenarios["group penalty removed, income ignored (refit without income)"] = no_income.predict(remote.assign(rem=0)).mean()
table = pd.DataFrame({"remote grant rate": scenarios})
table["grants among 4,000"] = (table["remote grant rate"] * len(remote)).round()
table["gap to Centre"] = centre["y"].mean() - table["remote grant rate"]
print(table.round(3).to_string())
print(f"Centre grant rate: {centre['y'].mean():.3f}")
rng = np.random.default_rng(0)
draws = []
for _ in range(500):
    sample = history.iloc[rng.integers(0, len(history), len(history))]
    m = committee_fit(sample)
    r = sample[sample["rem"] == 1]
    draws.append((m.params["rem"] / m.params["r"], m.predict(r.assign(rem=0)).mean() - r["y"].mean()))
draws = np.array(draws)
lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
print(f"bootstrap 95% (500 draws): penalty {lo[0]:.2f} to {hi[0]:.2f} R points; "
      f"remote grant rate lost to it {lo[1]:.3f} to {hi[1]:.3f}")

# Figure: grant rate against R score in 1-point bins, with the fitted curve at each group's median profile.
plots.apply_style()
fig, ax = plt.subplots(figsize=(7.5, 4.5))
bins = pd.cut(history["r"], np.arange(20, 37, 1))
grid = np.linspace(20, 36, 200)
for group in GROUPS:
    part = history[history[GROUP] == group]
    rate = part.groupby(bins[part.index], observed=True)["y"].agg(["mean", "size"])
    rate = rate[rate["size"] >= 20]
    ax.plot([i.mid for i in rate.index], rate["mean"], "o", color=plots.GROUP_COLORS[group],
            label=plots.GROUP_NAMES[group], markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
    curve = fit.predict(pd.DataFrame({"r": grid, "linc": part["linc"].median(), "hrs": part["hrs"].median(),
                                      "rem": part["rem"].iloc[0]}))
    ax.plot(grid, curve, "-", color=plots.GROUP_COLORS[group], linewidth=1.5)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Cote R")
ax.set_ylabel("Taux d'octroi du comité")
ax.set_title("Même courbe, décalée : le comité exige plus des régions éloignées")
ax.legend(loc="upper left")
save(fig, "a04_courbe_comite.png")

# Figure: what the committee weighed, one standard deviation of each column expressed in R points.
effects = (fit.params[sd.index] * sd / fit.params["r"]).rename({"r": "Cote R (+1 écart-type)",
                                                                 "linc": "Revenu familial (+1 écart-type)",
                                                                 "hrs": "Heures de travail (+1 écart-type)"})
effects["Région éloignée"] = fit.params["rem"] / fit.params["r"]
effects = effects.sort_values()
fig, ax = plt.subplots(figsize=(7.5, 3.4))
bars = ax.barh(effects.index, effects.values, height=0.6,
               color=[plots.GROUP_COLORS["Eloignee"] if v < 0 else plots.ACCENT for v in effects.values])
ax.bar_label(bars, labels=[f"{v:+.2f}".replace(".", ",") for v in effects.values], padding=4, color=plots.INK)
ax.axvline(0, color=plots.MUTED, linewidth=1)
ax.set_xlim(effects.min() - 0.6, effects.max() + 0.6)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Effet sur la décision, en points de cote R")
ax.set_title("Ce que le comité a pesé")
save(fig, "a04_poids_comite.png")
