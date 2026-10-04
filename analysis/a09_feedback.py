"""The scorer's feedback: what do the returned accuracies say the hidden reference rewards?

Reads submissions/log.csv. The feedback is used to choose between definitions of the
reference, never to tune the rule we submit.
"""
import numpy as np
from sklearn.metrics import f1_score

from common import GROUP, load, pd, section
from equialgo import evaluate, references
from equialgo.data import ID, ROOT

_, candidates = load()
log = pd.read_csv(ROOT / "submissions" / "log.csv").dropna(subset=["returned_accuracy"])
decisions = np.array([pd.read_csv(ROOT / "submissions" / f).set_index(ID).loc[candidates[ID], "decision_octroi"].to_numpy()
                      for f in log["file"]])
accuracy, f1 = log["returned_accuracy"].to_numpy() / 100, log["returned_f1_macro"].to_numpy() / 100
names = [f[:2] for f in log["file"]]

section("The feedback is on a subset of unknown size: how much sampling noise is that?")
print("The scores are cut to two decimals, so they do not reveal the subset size. Our predictions are on all 4,000.")
for n in (500, 1000, 2000):
    print(f"subset of {n}: standard error of an accuracy near 94% is {100 * np.sqrt(0.94 * 0.06 / n):.1f} point, "
          f"near 84% {100 * np.sqrt(0.84 * 0.16 / n):.1f}")

section("Reference rate implied by accuracy and macro F1 together (our files grant 40%)")
def macro_f1(q, a, p=0.40):
    tp = (a - 1 + p + q) / 2
    return 0.5 * (2 * tp / (p + q) + 2 * (1 - p - q + tp) / (2 - p - q))
rates = np.linspace(0.30, 0.50, 201)
for name, a, m in zip(names, accuracy, f1):
    if np.isnan(m):  # only the accuracy was reported for this file
        continue
    ok = rates[np.abs(macro_f1(rates, a) - m) < 0.0005]
    print(f"file {name}: {ok.min():.3f} to {ok.max():.3f}")

section("Each panel hypothesis: accuracy it predicted for each file, and rms error (accuracy and F1, points of %)")
def predicted(label):
    return (decisions == label).mean(1), np.array([f1_score(label, d, average="macro") for d in decisions])
def error(acc, f):
    return 100 * float(np.sqrt((((acc - accuracy) ** 2).mean() + np.nanmean((f - f1) ** 2)) / 2))
rows = {}
for name in references.SCORES:
    acc, f = predicted(references.label(name, candidates))
    rows[name] = {"rms error": error(acc, f), **dict(zip(names, 100 * acc))}
rows["returned"] = {"rms error": np.nan, **dict(zip(names, 100 * accuracy))}
print(pd.DataFrame(rows).T.sort_values("rms error").round(2).to_string(na_rep=""))

section("R + w x hours + noise (sd in R points), top 40%: rms error by hours weight and noise")
r, hours = candidates["r"].to_numpy(), candidates["hrs"].to_numpy()
draws = np.random.default_rng(0).normal(size=(30, len(candidates)))
def family(score, sd=0.0, rate=0.40):
    """Mean predicted (accuracy, F1) over noise draws for a reference = top `rate` of score + noise."""
    out = []
    for z in (draws if sd > 0 else draws[:1]):
        s = score + sd * z
        out.append(predicted((s >= np.quantile(s, 1 - rate)).astype(int)))
    acc, f = zip(*out)
    return np.mean(acc, axis=0), np.mean(f, axis=0)
weights, noises = [0.0, 0.05, 0.10, 0.145, 0.18, 0.22, 0.26], [0.0, 0.25, 0.5, 0.75, 1.0]
grid = pd.DataFrame({sd: {w: error(*family(r + w * hours, sd)) for w in weights} for sd in noises})
print(grid.rename_axis("hours weight").rename_axis("noise sd", axis=1).round(2).to_string())
best_w, best_sd = grid.stack().idxmin()
acc, _ = family(r + best_w * hours, best_sd)
print(f"best: weight {best_w}, noise {best_sd}; predicted accuracies {np.round(100 * acc, 1).tolist()} "
      f"against returned {np.round(100 * accuracy, 1).tolist()}")

section(f"Adding one more term to R + {best_w} x hours + noise {best_sd}: does the fit improve?")
base = r + best_w * hours
extras = {"log income": candidates["linc"], "first generation": candidates["fg"], "remote group": candidates["rem"],
          "log distance": candidates["ldist"]}
rows = {term: {w: error(*family(base + w * values.to_numpy(), best_sd)) for w in (-0.5, -0.25, 0.25, 0.5)}
        for term, values in extras.items()}
print(f"without extra term: {grid.loc[best_w, best_sd]:.2f}")
print(pd.DataFrame(rows).T.rename_axis("weight in R points", axis=1).round(2).to_string())

section("Under the best fit: agreement ceiling, and the equal-opportunity gap of each file")
s = base[None, :] + best_sd * draws
labels = (s >= np.quantile(s, 0.60, axis=1, keepdims=True)).astype(int)
ideal = (base >= np.quantile(base, 0.60)).astype(int)
print(f"the noise-free version of the fitted reference agrees with it at {100 * (labels == ideal).mean():.1f}%: "
      f"no rule can do better; file 02 got {100 * accuracy[names.index('02')]:.2f}%.")
groups = candidates[GROUP].to_numpy()
rows = {}
for name, d in zip(log["file"], decisions):
    gaps = [evaluate.equal_opportunity_gap(d, label, groups) for label in labels]
    rows[name] = {"EO gap": np.mean(gaps), "sd over noise draws": np.std(gaps),
                  "equity points": np.mean([evaluate.equity_points(g) for g in gaps])}
print(pd.DataFrame(rows).T.round(3).to_string())
