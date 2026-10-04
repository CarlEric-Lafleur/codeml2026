"""Prototype de surveillance en production du modèle corrigé (voir gouvernance.md, section 3).

Takes batches of decisions with the applicants' columns, computes the monitoring indicators
with their uncertainty, compares them to the thresholds, prints a status report and saves
one dashboard figure to figures/monitoring.png.

Indicators (definitions, thresholds and actions: gouvernance.md section 3):
  budget        overall grant rate against the 36 % to 44 % band
  parity        grant-rate gap between the two groups
  regions       widest grant-rate difference between two regions
  penalty       what being remote costs in the decisions, in R points: the group term of a
                logit refitted on the batch (1.40 in the committee's history)
  blind_hours   the hours weight of that logit fitted without the group. A hidden penalty
                moves into this weight (0.145 down to 0.059 in the history), so a fall
                warns of bias coming back through a neutral-looking column
  audit         equal-opportunity gap on a sample re-reviewed independently
  drift         population stability index of the inputs against the reference population

The demo replays the retained model on the historical data split into simulated periods,
then adds one period where a half-corrected version is deployed by mistake: the monitor
must raise the alert. The audited sample is simulated with the "R score and hours"
standard; in production it comes from the blind review panel.

Usage : python monitoring.py
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from equialgo import evaluate, plots, references
from equialgo.data import GROUP, R_SCORE, REGION, REMOTE, load_history
from equialgo.pipeline import complete, decide
from model_corrige import RETAINED

GREEN, AMBER, RED = "green", "amber", "red"
STATUS_LABELS = {GREEN: "VERT", AMBER: "AMBRE", RED: "ROUGE"}
STATUS_COLORS = {GREEN: "#1baf7a", AMBER: "#eda100", RED: "#d03b3b"}

# name: (French label, amber threshold, red threshold, direction). "abs": alert when the
# absolute value reaches the threshold; "low": when the value falls to it; "band": when the
# value leaves the interval. Set in gouvernance.md section 3.
INDICATORS = {
    "budget": ("Taux d'octroi global", (0.38, 0.42), evaluate.BUDGET, "band"),
    "parity": ("Écart de taux d'octroi entre les groupes", 0.06, 0.10, "abs"),
    "regions": ("Écart maximal entre deux régions", 0.12, 0.18, "abs"),
    "penalty": ("Pénalité régionale dans les décisions (points de cote R)", 0.35, 0.70, "abs"),
    "blind_hours": ("Poids des heures appris sans le groupe", 0.115, 0.09, "low"),
    "audit": ("Écart d'égalité des chances, échantillon audité", 0.10, 0.20, "abs"),
    "drift": ("Dérive des entrées (indice de stabilité, max.)", 0.10, 0.25, "abs"),
}
DRIFT_COLUMNS = [R_SCORE, references.HOURS, references.INCOME]
N_PERIODS = 4
AUDIT_SIZE = 600
N_BOOTSTRAP = 100


def status(name, value):
    _, amber, red, direction = INDICATORS[name]
    if direction == "band":
        return RED if not red[0] <= value <= red[1] else AMBER if not amber[0] <= value <= amber[1] else GREEN
    if direction == "low":
        return RED if value <= red else AMBER if value <= amber else GREEN
    return RED if abs(value) >= red else AMBER if abs(value) >= amber else GREEN


def _logit_ratio(batch, decisions, with_group):
    """Weights of a logit of the decisions, divided by the R-score weight."""
    X = pd.DataFrame({"r": batch[R_SCORE], "hrs": batch[references.HOURS], "linc": np.log(batch[references.INCOME])})
    if with_group:
        X["rem"] = (batch[GROUP] == REMOTE).astype(float)
    # A deterministic rule separates perfectly: the mild penalty keeps the fit finite, and
    # only the ratios of the weights are read.
    fit = LogisticRegression(C=10.0, max_iter=2000).fit((X - X.mean()) / X.std(), decisions)
    weights = pd.Series(fit.coef_[0], index=X.columns) / X.std()
    return weights / weights["r"]


def stability_index(reference, current, bins=10):
    """Population stability index of one column against the reference population."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    observed = np.histogram(current, edges)[0] / len(current)
    expected, observed = np.clip(expected, 1e-4, None), np.clip(observed, 1e-4, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def indicators(batch, decisions, reference, audit_sample=None):
    """Every indicator on one batch. `audit_sample` is (row positions, independent labels)."""
    decisions = np.asarray(decisions)
    values = {
        "budget": decisions.mean(),
        "parity": evaluate.parity_gap(decisions, batch[GROUP]),
        "regions": np.ptp(evaluate.selection_rates(decisions, batch[REGION]).to_numpy()),
        "penalty": -_logit_ratio(batch, decisions, with_group=True)["rem"],
        "blind_hours": _logit_ratio(batch, decisions, with_group=False)["hrs"],
        "drift": max(stability_index(reference[c], batch[c]) for c in DRIFT_COLUMNS),
    }
    if audit_sample is not None:
        rows, label = audit_sample
        values["audit"] = evaluate.equal_opportunity_gap(decisions[rows], label, batch[GROUP].iloc[rows])
    return values


def monitor(batches, reference, seed=0):
    """One report row per batch and indicator: value, 95 % bootstrap interval, status.

    `batches` maps a period name to (applicant table, decisions, audited sample or None).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for period, (batch, decisions, audit_sample) in batches.items():
        decisions = np.asarray(decisions)
        point = indicators(batch, decisions, reference, audit_sample)
        draws = []
        for _ in range(N_BOOTSTRAP):
            i = rng.integers(0, len(batch), len(batch))
            draw = indicators(batch.iloc[i], decisions[i], reference)
            if audit_sample is not None:  # the audited rows are resampled among themselves
                positions, label = audit_sample
                j = rng.integers(0, len(positions), len(positions))
                draw["audit"] = evaluate.equal_opportunity_gap(decisions[positions[j]], np.asarray(label)[j],
                                                               batch[GROUP].iloc[positions[j]])
            draws.append(draw)
        draws = pd.DataFrame(draws)
        for name, value in point.items():
            low, high = np.nanpercentile(draws[name], [2.5, 97.5])
            rows.append({"periode": period, "n": len(batch), "indicateur": name, "valeur": value, "bas": low,
                         "haut": high, "statut": status(name, value)})
    return pd.DataFrame(rows)


def dashboard(report, name="monitoring.png"):
    """One small panel per indicator: value per period with its interval and the thresholds."""
    plots.apply_style()
    names = [n for n in INDICATORS if n in set(report["indicateur"])]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.0))
    for ax, key in zip(axes.flat, names):
        label, amber, red, direction = INDICATORS[key]
        rows = report[report["indicateur"] == key]
        ax.errorbar(rows["periode"], rows["valeur"], yerr=[(rows["valeur"] - rows["bas"]).clip(lower=0), (rows["haut"] - rows["valeur"]).clip(lower=0)],
                    fmt="-", color=plots.MUTED, ecolor=plots.MUTED, elinewidth=1.1, capsize=3, linewidth=1.2, zorder=2)
        ax.scatter(rows["periode"], rows["valeur"], s=70, zorder=3, edgecolor=plots.SURFACE, linewidth=1.3,
                   color=[STATUS_COLORS[s] for s in rows["statut"]])
        for level, color in ((amber, STATUS_COLORS[AMBER]), (red, STATUS_COLORS[RED])):
            for value in np.atleast_1d(level):
                ax.axhline(value, color=color, linestyle="--", linewidth=1)
        ax.set_title(label.replace(" (", "\n(").replace(", ", ",\n"), fontsize=10)
        ax.set_xticks(range(len(rows)), [p.replace(" (", "\n(") for p in rows["periode"]])
        ax.grid(axis="x", visible=False)
        ax.margins(x=0.12, y=0.25)
    summary = axes.flat[len(names)]
    summary.axis("off")
    worst = report.groupby("periode", sort=False)["statut"].agg(lambda s: RED if RED in set(s) else AMBER if AMBER in set(s) else GREEN)
    summary.set_title("État de chaque période", fontsize=10)
    for i, (period, state) in enumerate(worst.items()):
        y = 0.85 - i * 0.17
        summary.scatter([0.08], [y], s=180, color=STATUS_COLORS[state], transform=summary.transAxes, clip_on=False)
        summary.text(0.2, y, f"{period} : {STATUS_LABELS[state]}", transform=summary.transAxes, va="center", fontsize=11)
    for ax in axes.flat[len(names) + 1:]:
        ax.set_visible(False)
    fig.suptitle("Surveillance du modèle corrigé : indicateurs par période, seuils ambre et rouge", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = plots.save(fig, name)
    plt.close(fig)
    return path


def demo_batches(seed=0):
    """Simulated periods of the retained model, then one where a half-corrected version runs."""
    history = load_history()
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(history)), N_PERIODS + 1)
    specs = [complete(RETAINED)] * N_PERIODS + [complete({**RETAINED, "strength": 0.5})]
    batches = {}
    for i, (index, spec) in enumerate(zip(parts, specs), start=1):
        period = history.iloc[index].reset_index(drop=True)
        audited = np.sort(rng.choice(len(period), min(AUDIT_SIZE, len(period)), replace=False))
        # Simulated blind review: the "R score and hours" standard, labelled on the whole period.
        label = np.asarray(references.label("R3_merit_hours", period))[audited]
        name = f"P{i}" if i <= N_PERIODS else f"P{i} (dérive)"
        batches[name] = (period, decide(spec, history, period), (audited, label))
    return batches, history


def main():
    warnings.filterwarnings("ignore")
    batches, reference = demo_batches()
    report = monitor(batches, reference)
    table = report.pivot(index="indicateur", columns="periode", values="valeur").loc[list(INDICATORS)]
    states = report.pivot(index="indicateur", columns="periode", values="statut").loc[list(INDICATORS)]
    printable = table.round(3).astype(str) + " " + states.map(STATUS_LABELS.get)
    printable.index = [INDICATORS[k][0] for k in printable.index]
    print("Indicateurs par période (valeur et état) :")
    print(printable[list(batches)].to_string())
    alerts = report[report["statut"] != GREEN]
    print(f"\n{len(alerts)} alerte(s) :" if len(alerts) else "\nAucune alerte.")
    for row in alerts.itertuples():
        print(f"  {row.periode} : {INDICATORS[row.indicateur][0]} = {row.valeur:.3f} "
              f"[{row.bas:.3f} ; {row.haut:.3f}] -> {STATUS_LABELS[row.statut]}")
    print(f"Figure : {dashboard(report).name}")


if __name__ == "__main__":
    main()
