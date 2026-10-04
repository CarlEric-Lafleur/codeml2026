"""Models trained to learn the difference between the programs of study.

Two sources can teach a difference. The history: models of the committee's decisions that
may read the program (one logit per program, then a logistic regression, a gradient boosting
and a random forest with the program as an input), each compared with the same model trained
on a shuffled program column. The scorer's feedback: program offsets fitted to the returned
scores of the scored files, compared with the same fit on returns simulated without any
program effect. Differences are in R points, Arts et lettres = 0.
"""
import itertools

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import COMMITTEE_FORMULA, load, logit, np, pd, plots, save, section
from equialgo import references
from equialgo.allocate import top_k
from equialgo.data import ID, ROOT

history, candidates = load()
PROGRAMS = sorted(history["prog"].unique())
OTHER = PROGRAMS[1:]  # Arts et lettres is the zero
RATE, NOISE, SHUFFLES = 0.40, 0.5, 5
K = int(round(RATE * len(candidates)))
base = references.score("R3_merit_hours", candidates).to_numpy()
rule = top_k(base, K)

section("One committee logit per program: do the weights or the bar differ? (R points)")
pooled = logit(COMMITTEE_FORMULA, history)
median = history[["linc", "hrs"]].median()
rows = {}
for program, part in history.groupby("prog"):
    b = logit(COMMITTEE_FORMULA, part).params
    rows[program] = {"decisions": len(part), "hours weight": b["hrs"] / b["r"], "income weight": b["linc"] / b["r"],
                     "remote penalty": -b["rem"] / b["r"],
                     "R score granted half the time (Centre, median income and hours)":
                         -(b["Intercept"] + b["linc"] * median["linc"] + b["hrs"] * median["hrs"]) / b["r"]}
per_program = pd.DataFrame(rows).T
print(per_program.round(3).to_string())
for name, formula in {"a bar per program": "y ~ r + linc + hrs + rem + C(prog)",
                      "a bar and weights per program": "y ~ (r + linc + hrs + rem) * C(prog)"}.items():
    model = logit(formula, history)
    stat, df = 2 * (model.llf - pooled.llf), model.df_model - pooled.df_model
    print(f"{name} against one committee for all: LR {stat:.1f}, df {df:.0f}, p = {stats.chi2.sf(stat, df):.2f}")

section("Models that may read the program: the difference each one learns, against a shuffled program column")
NUMERIC = ["r", "linc", "hrs", "rem"]
MODELS = {"logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000)),
          "gradient boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0),
          "random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=0, n_jobs=-1)}


def design(df, program=None, with_program=True):
    """Model inputs; `program` overrides every row's program (array or one name)."""
    X = df[NUMERIC].astype(float).reset_index(drop=True)
    if with_program:
        programs = df["prog"].to_numpy() if program is None else np.broadcast_to(program, len(df))
        for name in OTHER:
            X[name] = (programs == name).astype(float)
    return X


def log_odds(model, X):
    return np.log(np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99) / np.clip(model.predict_proba(X)[:, 0], 0.01, 0.99))


def learned(make, programs):
    """What the model trained with `programs` as the program column adds for each program, in R points.

    Measured on the 4,000, on the applicants the model is unsure of (the others cannot change
    decision): the log-odds with every applicant placed in the program, minus the same in Arts
    et lettres, over what one R point is worth to the model there.
    """
    model = make().fit(design(history, programs), history["y"])
    at = lambda program, shift=0.0: log_odds(model, design(candidates.assign(r=candidates["r"] + shift), program))
    zero = at(PROGRAMS[0])
    unsure = np.abs(zero) < 2
    per_r_point = (at(PROGRAMS[0], 0.5) - at(PROGRAMS[0], -0.5))[unsure].mean()
    return pd.Series({name: (at(name) - zero)[unsure].mean() / per_r_point for name in OTHER})


rng = np.random.default_rng(0)
shuffled = [rng.permutation(history["prog"].to_numpy()) for _ in range(SHUFFLES)]
offsets, chance = {}, {}
for name, make in MODELS.items():
    offsets[name] = learned(make, history["prog"].to_numpy())
    chance[name] = np.abs(pd.concat([learned(make, programs) for programs in shuffled])).max()
table = pd.DataFrame(offsets).T
table["largest learned"] = table[OTHER].abs().max(axis=1)
table[f"largest on a shuffled column ({SHUFFLES} shuffles)"] = pd.Series(chance)
table["decisions the difference moves"] = [
    int((top_k(base + candidates["prog"].map(offsets[name]).fillna(0).to_numpy(), K) != rule).sum()) for name in MODELS]
print(table.round(3).to_string())

section("Does reading the program help predict the committee? Log-loss on held-out decisions (5 folds)")
folds = StratifiedKFold(5, shuffle=True, random_state=0)
rows = {}
for name, make in MODELS.items():
    held = {flag: cross_val_predict(make(), design(history, with_program=flag), history["y"], cv=folds, method="predict_proba")[:, 1]
            for flag in (False, True)}
    per_row = [-(history["y"] * np.log(p) + (1 - history["y"]) * np.log(1 - p)) for p in map(lambda p: np.clip(p, 1e-6, 1 - 1e-6), held.values())]
    gain = per_row[0] - per_row[1]
    rows[name] = {"without the program": log_loss(history["y"], held[False]), "with the program": log_loss(history["y"], held[True]),
                  "gain": gain.mean(), "se of the gain": gain.std() / np.sqrt(len(gain))}
print(pd.DataFrame(rows).T.round(4).to_string())

section("Program offsets fitted to the scorer's returns (grid of -1 to +1 by 0.5 per program, noise 0.5)")
log = pd.read_csv(ROOT / "submissions" / "log.csv").dropna(subset=["returned_accuracy"])
files = np.array([pd.read_csv(ROOT / "submissions" / f).set_index(ID).loc[candidates[ID], "decision_octroi"].to_numpy()
                  for f in log["file"]], dtype=float)
returned = np.r_[log["returned_accuracy"].to_numpy(), log["returned_f1_macro"].to_numpy()]
draws = NOISE * np.random.default_rng(0).normal(size=(10, len(candidates)))
prog = candidates["prog"].to_numpy()


def scores(labels):
    """Accuracy and macro F1 (%) of every scored file against each label: (labels, 2 x files)."""
    labels = labels.astype(float)
    tp, tn = labels @ files.T, (1 - labels) @ (1 - files).T
    granted, deserving, n = files.sum(1)[None, :], labels.sum(1)[:, None], files.shape[1]
    f1 = (2 * tp / (granted + deserving) + 2 * tn / (2 * n - granted - deserving)) / 2
    return 100 * np.hstack([(tp + tn) / n, f1])


def reference(offset, noise):
    noisy = base + offset + noise
    return (noisy >= np.quantile(noisy, 1 - RATE, axis=-1, keepdims=True)).astype(int)


GRID = [dict(zip(OTHER, values)) for values in itertools.product((-1.0, -0.5, 0.0, 0.5, 1.0), repeat=len(OTHER))]
NONE = GRID.index(dict.fromkeys(OTHER, 0.0))
predicted = np.array([scores(reference(pd.Series(prog).map(offset).fillna(0).to_numpy(), draws)).mean(0) for offset in GRID])


def fit(returns):
    """(best grid point, rms error without a program effect, rms error at the best point)."""
    errors = np.sqrt(((predicted - returns) ** 2).mean(1))
    return int(errors.argmin()), errors[NONE], errors.min()


best, without, fitted_error = fit(returned)
print(f"{len(files)} scored files, {len(GRID)} grid points. rms error without a program effect {without:.2f}, at the best point "
      f"{fitted_error:.2f} (gain {without - fitted_error:.2f})")
print("best offsets:", GRID[best])
# The same fit on returns that hold no program effect: how much does the grid gain by chance?
fresh = NOISE * np.random.default_rng(1).normal(size=(40, len(candidates)))
gains = np.array([np.subtract(*fit(row)[1:]) for row in scores(reference(0.0, fresh))])
print(f"gain of the same fit on 40 simulated returns without any program effect (all 4,000 scored): "
      f"median {np.median(gains):.2f}, 90th percentile {np.quantile(gains, 0.9):.2f}, max {gains.max():.2f}")
print(f"share of the simulated returns with a gain of at least {without - fitted_error:.2f}: {(gains >= without - fitted_error).mean():.0%}")
print("The scorer uses a subset, so chance gains on the real returns are larger than these.")

# Figure: what each model learns for each program, against the range a shuffled column gives.
plots.apply_style()
fig, ax = plt.subplots(figsize=(8.5, 4.6))
y = np.arange(len(OTHER))
for offset, (name, label) in zip((-0.2, 0.0, 0.2), {"logistic regression": "Régression logistique",
                                                   "gradient boosting": "Gradient boosting", "random forest": "Forêt aléatoire"}.items()):
    ax.errorbar(offsets[name][OTHER], y + offset, xerr=chance[name], fmt="o", markersize=5, capsize=3, label=label)
ax.axvline(0, color=plots.MUTED, linewidth=1)
ax.set_xlim(-0.5, 0.5)
ax.set_yticks(y, OTHER)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Écart appris par rapport à Arts et lettres (points de cote R) ; barre : plus grand écart appris sur un programme mélangé")
ax.set_title("Aucun modèle n'apprend de différence entre les programmes")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
save(fig, "a19_programme_modeles.png")
