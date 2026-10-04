"""Shared setup for the analysis scripts: repo root on the path, short column names, output.

Run a script from the repo root: `.venv/Scripts/python.exe analysis/a04_committee.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from equialgo import plots, references
from equialgo.data import CENTRE, GROUP, GROUPS, REMOTE, load_candidates, load_history

pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200)

# Short aliases, added next to the original columns so formulas stay readable.
ALIASES = {
    "cote_r_equivalent": "r",
    "revenu_familial_estime": "inc",
    "heures_travail_semaine": "hrs",
    "distance_domicile_campus_km": "dist",
    "premiere_generation_universitaire": "fg",
    "programme_etudes": "prog",
    "region_administrative": "reg",
    "code_postal_3": "cp",
    "decision_octroi": "y",
}
NUMERIC = ["r", "inc", "hrs", "dist", "fg"]
# Income and distance are right-skewed; their logs are what behaves linearly.
MODEL_NUMERIC = ["r", "linc", "hrs", "ldist", "fg"]
COMMITTEE_FORMULA = "y ~ r + linc + hrs + rem"


def _prepare(df):
    for long, short in ALIASES.items():
        if long in df:
            df[short] = df[long]
    df["rem"] = (df[GROUP] == REMOTE).astype(int)
    df["linc"] = np.log(df["inc"])
    df["ldist"] = np.log(df["dist"])
    return df


def load():
    """(history, candidates) with the short aliases, `rem`, `linc` and `ldist` added."""
    return _prepare(load_history()), _prepare(load_candidates())


def section(title):
    print(f"\n=== {title} ===")


def save(fig, name):
    path = plots.save(fig, f"analysis/{name}")
    print(f"[figure] {path.relative_to(path.parents[2])}")
    return path


def logit(formula, data):
    return smf.logit(formula, data).fit(disp=0)


def coef_table(model):
    return pd.DataFrame({"coef": model.params, "se": model.bse, "z": model.tvalues}).round(3)


def committee_fit(history):
    """The committee as the data shows it: additive logit on R, log income, hours, group."""
    return logit(COMMITTEE_FORMULA, history)


def standards(df):
    """The reference panel of equialgo/references.py as priority scores, one column per hypothesis."""
    return pd.DataFrame({name: references.score(name, df) for name in references.SCORES})


def top_share(score, rate=0.40):
    """1 for the top `rate` share by score."""
    return (score >= score.quantile(1 - rate)).astype(int)


STANDARD_LABELS = references.LABELS


# The four continuous columns as histogram panels: (column, axis label, log scale, tick positions).
HIST_PANELS = [
    ("r", "Cote R", False, None),
    ("inc", "Revenu familial (k$, échelle log)", True, {20e3: "20", 50e3: "50", 100e3: "100", 200e3: "200"}),
    ("hrs", "Heures de travail par semaine", False, None),
    ("dist", "Distance au campus (km, échelle log)", True, {1: "1", 10: "10", 100: "100", 500: "500"}),
]


def share_hists(axes, samples, colors, **style):
    """One panel per HIST_PANELS entry; each sample is drawn as the share of its rows per bin."""
    pooled = pd.concat(samples.values())
    for ax, (col, label, log, ticks) in zip(axes.flat, HIST_PANELS):
        x = pooled[col]
        if col == "hrs":
            bins = np.arange(x.min() - 0.5, x.max() + 1.5)
        else:
            bins = np.geomspace(x.min(), x.max(), 40) if log else np.linspace(x.min(), x.max(), 40)
        for name, df in samples.items():
            ax.hist(df[col], bins=bins, weights=np.full(len(df), 1 / len(df)), color=colors[name], label=name, **style)
        if log:
            ax.set_xscale("log")
            ax.set_xticks(list(ticks), list(ticks.values()))
            ax.minorticks_off()
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.grid(visible=False)


def group_dots(ax, table, xlabel):
    """Dumbbell: one row per index value, one dot per group. `table` has the two group columns."""
    y = range(len(table))
    ax.hlines(y, table[CENTRE], table[REMOTE], color=plots.GRID, linewidth=3, zorder=1)
    for group in GROUPS:
        ax.scatter(table[group], y, s=70, color=plots.GROUP_COLORS[group], label=plots.GROUP_NAMES[group],
                   edgecolor=plots.SURFACE, linewidth=1.5, zorder=2)
    ax.set_yticks(list(y), [str(i) for i in table.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
