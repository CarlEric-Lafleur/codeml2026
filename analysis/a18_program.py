"""Does the hidden reference read the program of study, and what file would tell?

The committee gave no program any weight (a10) and none of the scored files reads the
program, so the feedback so far cannot tell a program effect from noise. Three questions:
how large a program effect the history and the feedback leave room for, what a rule that
favours one program (`rule_r_hours_program`, experiment e010) changes on the 4,000, and how
its returned score would move if the reference favoured that program, another one, or none.
"""
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from common import GROUP, load, logit, np, pd, plots, save, section
from equialgo import evaluate, references
from equialgo.allocate import top_k
from equialgo.data import ID, ROOT

history, candidates = load()
groups, prog = candidates[GROUP].to_numpy(), candidates["prog"].to_numpy()
PROGRAMS = sorted(candidates["prog"].unique())
RATE, NOISE = 0.40, 0.5  # the fitted reference: a09
K = int(round(RATE * len(candidates)))
BONUSES = [0.5, 1.0]
OTHERS = [name for name in references.PLAUSIBLE if name != "R3_merit_hours"]
base = references.score("R3_merit_hours", candidates).to_numpy()
rule = top_k(base, K)
draws = NOISE * np.random.default_rng(0).normal(size=(30, len(candidates)))


def favouring(program, bonus):
    """Priority score of R + hours with `bonus` R points for `program` (None: no program favoured)."""
    return base if program is None else base + bonus * (prog == program)


def reference(program=None, bonus=0.0):
    """30 draws of a reference: top 40% of R + hours + the program bonus + noise of 0.5 R point."""
    noisy = favouring(program, bonus)[None, :] + draws
    return (noisy >= np.quantile(noisy, 1 - RATE, axis=1, keepdims=True)).astype(int)


section("The committee: effect of each program in R points (Arts et lettres = 0), and the size it rules out")
committee = logit("y ~ r + linc + hrs + rem + C(prog)", history)
effect = committee.params.filter(like="prog") / committee.params["r"]
se = committee.bse.filter(like="prog") / committee.params["r"]
print(pd.DataFrame({"effect": effect, "se": se, "95% low": effect - 1.96 * se, "95% high": effect + 1.96 * se}).round(3).to_string())
print("The committee says nothing about the reference: it was built independently of it.")

section("The 4,000 per program: size, share remote, grant rate under the retained rule")
print(pd.DataFrame({"applicants": candidates.groupby("prog").size(),
                    "share Eloignee": candidates.groupby("prog")["rem"].mean(),
                    "mean R": candidates.groupby("prog")["r"].mean(), "mean hours": candidates.groupby("prog")["hrs"].mean(),
                    "rate, retained rule": pd.Series(rule, index=candidates.index).groupby(candidates["prog"]).mean()}).round(3).to_string())

section("Feedback so far: rms error (points of %) of the returned scores if the reference also favours one program")
log = pd.read_csv(ROOT / "submissions" / "log.csv").dropna(subset=["returned_accuracy"])
files = np.array([pd.read_csv(ROOT / "submissions" / f).set_index(ID).loc[candidates[ID], "decision_octroi"].to_numpy()
                  for f in log["file"]])
accuracy, f1 = log["returned_accuracy"].to_numpy() / 100, log["returned_f1_macro"].to_numpy() / 100


def error(labels):
    """rms distance between the returned scores and those predicted, mean and sd over the draws of the reference."""
    out = [100 * np.sqrt((((files == label).mean(1) - accuracy) ** 2).mean() / 2
                         + ((np.array([f1_score(label, d, average="macro") for d in files]) - f1) ** 2).mean() / 2)
           for label in labels]
    return np.mean(out), np.std(out)


rows = {"no program effect": dict(zip(["rms error", "sd over draws"], error(reference())))}
for program in PROGRAMS:
    for bonus in (-1.0, 1.0):
        rows[f"{bonus:+.0f} R point for {program}"] = dict(zip(["rms error", "sd over draws"], error(reference(program, bonus))))
print(f"{len(files)} scored files, none reads the program")
print(pd.DataFrame(rows).T.round(2).to_string())

section("A rule that favours one program, on the 4,000 (accuracy against the fitted reference, which has no program effect)")
fitted = reference()
rows = {"retained rule": {"decisions that differ": 0, "rate, favoured program": np.nan, "parity gap": evaluate.parity_gap(rule, groups),
                          "accuracy, fitted reference": (fitted == rule).mean(),
                          "worst case points, other hypotheses": evaluate.against_panel(rule, candidates, names=OTHERS)["points"].min()}}
within = top_k(pd.Series(base).groupby(prog).rank(pct=True).to_numpy(), K)
tries = {f"+{bonus} for {program}": top_k(favouring(program, bonus), K) for bonus in BONUSES for program in PROGRAMS}
tries["same rate in every program"] = within
for name, d in tries.items():
    program = name.split(" for ")[-1]
    rows[name] = {"decisions that differ": int((d != rule).sum()),
                  "rate, favoured program": d[prog == program].mean() if program in PROGRAMS else np.nan,
                  "parity gap": evaluate.parity_gap(d, groups), "accuracy, fitted reference": (fitted == d).mean(),
                  "worst case points, other hypotheses": evaluate.against_panel(d, candidates, names=OTHERS)["points"].min()}
print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.3f}", na_rep=""))

section("What the score of a +0.5 file would say: its accuracy minus the retained rule's, in points of %, against the same reference")
# The truth is unknown, so each file is read under several references: none favoured, the
# file's own program favoured or disfavoured, another program favoured (mean over the four).
truths = {"no program effect": lambda p, q: reference(), "reference: +0.5 for it": lambda p, q: reference(p, 0.5),
          "reference: +1 for it": lambda p, q: reference(p, 1.0), "reference: -1 for it": lambda p, q: reference(p, -1.0),
          "reference: +1 for another": lambda p, q: reference(q, 1.0)}


def shifts(bonus):
    """(mean, sd over the draws) of each file's accuracy minus the retained rule's, per program and reference."""
    rows, spread = {}, {}
    for program in PROGRAMS:
        d = tries[f"+{bonus} for {program}"]
        rows[program], spread[program] = {}, {}
        for name, truth in truths.items():
            partners = [q for q in PROGRAMS if q != program] if "another" in name else [program]
            delta = np.concatenate([100 * ((truth(program, q) == d).mean(1) - (truth(program, q) == rule).mean(1)) for q in partners])
            rows[program][name], spread[program][name] = delta.mean(), delta.std()
    return pd.DataFrame(rows).T, pd.DataFrame(spread).T


table, spread = shifts(0.5)
print(table.round(2).to_string())
print("sd over the draws of the reference's noise (the subset sampling comes on top):")
print(spread.round(2).to_string())
print("retained rule's own accuracy under each reference:",
      {name: round(100 * np.mean([(truth(p, PROGRAMS[0] if p != PROGRAMS[0] else PROGRAMS[1]) == rule).mean() for p in PROGRAMS]), 2)
       for name, truth in truths.items()})
print("the same with a bonus of 1 R point in the file (twice the decisions moved):")
print(shifts(1.0)[0].round(2).to_string())

section("The same rate in every program: its accuracy minus the retained rule's")
centred = base - pd.Series(base).groupby(prog).transform("mean").to_numpy() + draws
centred = (centred >= np.quantile(centred, 1 - RATE, axis=1, keepdims=True)).astype(int)
for name, truth in {"no program effect": fitted, "reference: R + hours centred inside each program": centred}.items():
    delta = 100 * ((truth == within).mean(1) - (truth == rule).mean(1))
    print(f"{name}: {delta.mean():+.2f} (sd {delta.std():.2f})")

# Figure: the shift of each file's score under each reference.
plots.apply_style()
fig, ax = plt.subplots(figsize=(8.5, 4.8))
y = np.arange(len(PROGRAMS))
for offset, (name, label) in zip((-0.24, -0.08, 0.08, 0.24), {
        "reference: +1 for it": "La référence favorise ce programme (+1 pt)", "no program effect": "Aucun effet du programme",
        "reference: +1 for another": "La référence favorise un autre programme (+1 pt)",
        "reference: -1 for it": "La référence défavorise ce programme (-1 pt)"}.items()):
    ax.errorbar(table[name], y + offset, xerr=2 * spread[name], fmt="o", markersize=5, capsize=3, label=label)
ax.axvline(0, color=plots.GRID, linewidth=1)
ax.set_yticks(y, PROGRAMS)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlabel("Exactitude du fichier « +0,5 pt pour ce programme » moins celle de la règle retenue (points de %)")
ax.set_title("Ce que le score d'une règle lisant le programme permettrait de conclure")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
save(fig, "a18_programme.png")
