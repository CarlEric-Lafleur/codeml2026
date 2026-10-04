"""R + hours in depth: what weight does the history support for hours worked, and does the program of study matter?"""
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import f1_score

from common import CENTRE, COMMITTEE_FORMULA, GROUP, REMOTE, load, logit, pd, section
from equialgo.data import ID, ROOT

history, candidates = load()
base = logit(COMMITTEE_FORMULA, history)


def weight(model, term="hrs"):
    """(ratio, standard error) of a coefficient to the R-score coefficient: the term in R points."""
    p, v = model.params, model.cov_params()
    ratio = p[term] / p["r"]
    se = abs(ratio) * np.sqrt(v.loc[term, term] / p[term] ** 2 + v.loc["r", "r"] / p["r"] ** 2
                              - 2 * v.loc[term, "r"] / (p[term] * p["r"]))
    return ratio, se


section("Hours weight in R points per hour, estimated on the committee's decisions")
rows = {"committee logit (R, log income, hours, group)": weight(base),
        "same, probit": weight(smf.probit(COMMITTEE_FORMULA, history).fit(disp=0)),
        "Centre only": weight(logit("y ~ r + linc + hrs", history[history[GROUP] == CENTRE])),
        "Eloignee only": weight(logit("y ~ r + linc + hrs", history[history[GROUP] == REMOTE])),
        "group left out (R, log income, hours)": weight(logit("y ~ r + linc + hrs", history)),
        "group and income left out (R, hours)": weight(logit("y ~ r + hrs", history))}
table = pd.DataFrame(rows, index=["weight", "se"]).T
table["low 95%"], table["high 95%"] = table["weight"] - 1.96 * table["se"], table["weight"] + 1.96 * table["se"]
print(table.round(3).to_string())
print("Leaving the group out halves the weight: hours worked are a proxy of the remote regions, so a model\n"
      "that does not see the group charges part of the penalty to the hours.")

section("Is the hours effect a straight line? Effect of each hours band, in R points, against its mean hours")
history["band"] = pd.cut(history["hrs"], [0, 5, 8, 11, 14, 17, 30])
banded = logit("y ~ r + linc + C(band) + rem", history)
effect = (banded.params.filter(like="band") / banded.params["r"]).to_numpy()
mean_hours = history.groupby("band", observed=True)["hrs"].mean().to_numpy()
print(pd.DataFrame({"mean hours": mean_hours, "effect (R points)": np.r_[0, effect],
                    "slope from first band": np.r_[np.nan, effect / (mean_hours[1:] - mean_hours[0])]}).round(3).to_string(index=False))

section("Terms added to the committee logit: likelihood-ratio tests")
tests = {"hours squared": "y ~ r + linc + hrs + I(hrs ** 2) + rem", "R squared": "y ~ r + I(r ** 2) + linc + hrs + rem",
         "R x hours": "y ~ r * hrs + linc + rem", "hours x group": "y ~ r + linc + hrs * rem",
         "program": "y ~ r + linc + hrs + rem + C(prog)", "program x R": "y ~ r * C(prog) + linc + hrs + rem",
         "program x hours": "y ~ r + linc + hrs * C(prog) + rem", "program x group": "y ~ r + linc + hrs + rem * C(prog)"}
rows = {}
for name, formula in tests.items():
    model = logit(formula, history)
    stat, df = 2 * (model.llf - base.llf), model.df_model - base.df_model
    rows[name] = {"LR": stat, "df": int(df), "p": stats.chi2.sf(stat, df)}
print(pd.DataFrame(rows).T.round(3).to_string())
print(f"{len(tests)} tests: the smallest p-value expected by chance is about {1 / (len(tests) + 1):.2f}.")
inter = logit(tests["R x hours"], history).params
print("with the R x hours term, the local hours weight is",
      {f"R {r0}": round((inter["hrs"] + inter["r:hrs"] * r0) / (inter["r"] + inter["r:hrs"] * 11), 3) for r0 in (26, 28.5, 31)},
      "(28.5 is where the 40% cutoff sits)")

section("Program of study: effect in the committee's decision (R points, Arts et lettres = 0) and link to R score")
program = logit(tests["program"], history)
print(pd.DataFrame({"effect": program.params.filter(like="prog") / program.params["r"],
                    "se": program.bse.filter(like="prog") / program.params["r"]}).round(3).to_string())
both = pd.concat([history, candidates])
print("mean R score per program, all 14,000:", both.groupby("prog")["r"].mean().round(2).to_dict())
for group, part in both.groupby(GROUP):
    p = stats.f_oneway(*[g["r"].to_numpy() for _, g in part.groupby("prog")]).pvalue
    print(f"  R score differs between programs inside {group}? ANOVA p = {p:.3f}")

section("Weight that calls the same share of each group deserving (top 40% of R + w x hours)")
for name, df in (("history", history), ("candidates", candidates)):
    gaps = {}
    for w in np.arange(0.10, 0.2001, 0.005):
        score = df["r"] + w * df["hrs"]
        label = score >= score.quantile(0.60)
        gaps[round(w, 3)] = label[df[GROUP] == CENTRE].mean() - label[df[GROUP] == REMOTE].mean()
    gaps = pd.Series(gaps)
    print(f"{name}: parity at w = {gaps.abs().idxmin()} (gap {gaps[gaps.abs().idxmin()]:+.4f}); at 0.145 the gap is {gaps[0.145]:+.4f}")

section("What a different weight would change on the 4,000")
ours = candidates["r"] + 0.145 * candidates["hrs"]
ours = (ours >= ours.quantile(0.60)).to_numpy()
rows = {}
for w in (0.10, 0.13, 0.16, 0.18, 0.22):
    other = candidates["r"] + w * candidates["hrs"]
    other = (other >= other.quantile(0.60)).to_numpy()
    rows[w] = {"decisions that differ from w = 0.145": (ours != other).mean(),
               "grant rate Centre": other[candidates[GROUP] == CENTRE].mean(),
               "grant rate Eloignee": other[candidates[GROUP] == REMOTE].mean()}
print(pd.DataFrame(rows).T.rename_axis("w").round(3).to_string())

section("Scorer feedback: can it tell a program effect from noise?")
log = pd.read_csv(ROOT / "submissions" / "log.csv").dropna(subset=["returned_accuracy"])
decisions = np.array([pd.read_csv(ROOT / "submissions" / f).set_index(ID).loc[candidates[ID], "decision_octroi"].to_numpy()
                      for f in log["file"]])
accuracy, f1 = log["returned_accuracy"].to_numpy() / 100, log["returned_f1_macro"].to_numpy() / 100
def error(score):
    """rms distance (points of %) between the returned scores and those predicted if the reference is the top 40% of `score`."""
    score = np.asarray(score, dtype=float)
    label = (score >= np.quantile(score, 0.60)).astype(int)
    acc = (decisions == label).mean(1)
    f = np.array([f1_score(label, d, average="macro") for d in decisions])
    return 100 * float(np.sqrt((((acc - accuracy) ** 2).mean() + ((f - f1) ** 2).mean()) / 2))
rule = candidates["r"] + 0.18 * candidates["hrs"]
draws = np.random.default_rng(0).normal(size=(30, len(candidates)))
rows = {"R + 0.18 x hours, nothing else": error(rule),
        "+ noise of 0.5 R point (mean of 30 draws)": np.mean([error(rule + 0.5 * z) for z in draws])}
for name in sorted(candidates["prog"].unique()):
    for bonus in (-1, 1):
        rows[f"{bonus:+d} R point for {name}"] = error(rule + bonus * (candidates["prog"] == name))
rows["R centred inside each program"] = error(rule - candidates["prog"].map(both.groupby("prog")["r"].mean()))
print(pd.Series(rows).round(2).to_string())
print("None of the submitted rules reads the program, so any term independent of them (a program bonus,\n"
      "or plain noise) improves the fit about equally: the feedback cannot tell them apart.")
