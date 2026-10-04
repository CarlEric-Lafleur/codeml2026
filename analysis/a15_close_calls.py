"""Does a second behaviour for the close calls do better than the retained rule everywhere?

A close call is an applicant within a band of the cutoff of R + 0.145 x hours. The `close_*`
scorers keep the rule outside the band and rank the applicants inside it by something else
(experiment e008). Three questions: how many close calls there are, whether the committee
itself decided them differently, and what each second behaviour changes.
"""
import importlib.util

import matplotlib.pyplot as plt

from common import CENTRE, GROUP, REMOTE, load, logit, np, pd, plots, save, section
from equialgo import evaluate, references
from equialgo.allocate import top_k
from equialgo.data import ROOT
from equialgo.pipeline import decide, expand
from equialgo.scorers import close_calls

history, candidates = load()
groups = candidates[GROUP].to_numpy()
RATE, NOISE = 0.40, 0.5  # the fitted reference: top 40% of R + hours + noise worth 0.5 R point (a09)
K = int(round(RATE * len(candidates)))
rule_score = references.score("R3_merit_hours", candidates).to_numpy()
rule = top_k(rule_score, K)
OTHERS = [name for name in references.PLAUSIBLE if name != "R3_merit_hours"]

module_spec = importlib.util.spec_from_file_location("e008", ROOT / "experiments" / "e008_close_calls.py")
e008 = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(e008)
BANDS = e008.BANDS

# The fitted reference, drawn 30 times: one 0/1 label per draw.
noisy = rule_score[None, :] + NOISE * np.random.default_rng(0).normal(size=(30, len(candidates)))
fitted = (noisy >= np.quantile(noisy, 1 - RATE, axis=1, keepdims=True)).astype(int)

section("How many close calls on the 4,000, and how sure is the rule about them?")
rows = {}
for band in BANDS + [2.0]:
    margin, close = close_calls(rule_score, RATE, band)
    rows[f"within {band} R point"] = {
        "applicants": int(close.sum()), "share": close.mean(),
        "Centre": int((close & (groups == CENTRE)).sum()), "Eloignee": int((close & (groups == REMOTE)).sum()),
        "granted by the rule": int(rule[close].sum()),
        "rule = fitted reference, inside": (fitted[:, close] == rule[close]).mean(),
        "outside": (fitted[:, ~close] == rule[~close]).mean(),
        "share of the rule's errors inside": (fitted[:, close] != rule[close]).sum() / (fitted != rule).sum()}
print(pd.DataFrame(rows).T.round(3).to_string())

section("Did the committee decide its own close calls differently? Its logit refitted on them (weights in R points)")
committee = logit("y ~ r + linc + hrs + rem + fg", history)
z = committee.fittedvalues / committee.params["r"]  # distance to the committee's own cutoff, in R points
def in_r_points(model):
    b, se = model.params, model.bse
    out = {f"{term}": b[term] / b["r"] for term in ("linc", "hrs", "rem", "fg")}
    out.update({"rem se": se["rem"] / b["r"], "fg se": se["fg"] / b["r"], "noise (R points)": 1.81 / b["r"], "rows": int(model.nobs)})
    return out
rows = {"all decisions": in_r_points(committee)}
for band in (1.0, 2.0):
    rows[f"close: within {band} R point of its cutoff"] = in_r_points(logit("y ~ r + linc + hrs + rem + fg", history[z.abs() <= band]))
    rows[f"clear: beyond {band}"] = in_r_points(logit("y ~ r + linc + hrs + rem + fg", history[z.abs() > band]))
print(pd.DataFrame(rows).T.round(3).to_string())
close = history[z.abs() <= 1.0]
extra = logit("y ~ r + linc + hrs + rem + fg + C(prog) + ldist", close)
print(f"close calls (within 1.0), program and distance added: largest |z| among the added terms "
      f"{extra.tvalues.drop(['Intercept', 'r', 'linc', 'hrs', 'rem', 'fg']).abs().max():.2f}")

section("Each second behaviour (e008): what changes against the rule alone, on the 4,000")
def measure(d):
    panel = evaluate.against_panel(d, candidates, names=OTHERS)
    accuracy = (fitted == d).mean(axis=1)
    gaps = [evaluate.equal_opportunity_gap(d, label, groups) for label in fitted]
    return {"decisions that differ": int((d != rule).sum()), "parity gap": evaluate.parity_gap(d, groups),
            "accuracy, fitted reference": accuracy.mean(), "EO gap, fitted reference": np.mean(gaps),
            "worst case points, other hypotheses": panel["points"].min(), "worst": panel["points"].idxmin()[:3]}
rows = {("rule alone (retained)", 0.0): measure(rule)}
for spec in (full for spec in e008.SPECS for full in expand(spec)):
    rows[spec["scorer"], spec["band"]] = measure(decide(spec, history, candidates))
table = pd.DataFrame(rows).T
table.index.names = ["scorer", "band"]
print(table.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"standard error of the accuracy over the 30 noise draws: {(fitted == rule).mean(axis=1).std():.4f}")

# Figure: accuracy against the fitted reference, by band, one line per second behaviour.
plots.apply_style()
NAMES = {"close_r": "Cote R seule", "close_first_gen": "Première génération d'abord", "close_low_income": "Faible revenu d'abord",
         "close_remote": "Régions éloignées d'abord", "close_boosting": "Modèle entraîné (tout l'historique)",
         "close_local_logit": "Logit entraîné sur les cas serrés", "close_local_boosting": "Boosting entraîné sur les cas serrés"}
fig, ax = plt.subplots(figsize=(8, 4.6))
for scorer, name in NAMES.items():
    line = table.xs(scorer, level="scorer")["accuracy, fitted reference"].astype(float)
    ax.plot([0.0] + list(line.index), [rows["rule alone (retained)", 0.0]["accuracy, fitted reference"]] + list(line), marker="o", label=name)
ax.set_xlabel("Demi-largeur de la zone des cas serrés (points de cote R)")
ax.set_ylabel("Accord avec l'étalon estimé (cote R + heures + bruit)")
ax.set_title("Cas serrés : s'écarter de la règle coûte de l'accord avec l'étalon estimé")
ax.legend(loc="lower left", fontsize=8)
save(fig, "a15_close_calls.png")
