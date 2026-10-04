"""The panel of hypothesized reference standards.

The hidden reference is unknown. Each hypothesis here is a priority score on the applicant
columns; its label calls the top `rate` share of a table "deserving". Inside a group the
columns are independent, so a standard can only be a function of the columns themselves.

`PLAUSIBLE` lists the hypotheses still considered credible: compatible with the published
anchor (baseline equal-opportunity gap 0.270, `analysis/a08_anchor.py`) and with the
scorer's feedback (`analysis/a09_feedback.py`). The other entries stay in the panel so that
the reason they were dropped can be re-checked.
"""
import numpy as np
import pandas as pd

from .data import GROUP, R_SCORE

INCOME = "revenu_familial_estime"
HOURS = "heures_travail_semaine"
FIRST_GEN = "premiere_generation_universitaire"

# The committee's weights in R points, from its additive logit (analysis/a04_committee.py):
# per unit of log income and per hour worked. tests/test_references.py re-fits them.
COMMITTEE_LOG_INCOME = 1.29
COMMITTEE_HOURS = 0.145

DEFAULT_RATE = 0.40  # assumed: the reference labels about as many applicants as the budget funds


def _z(series):
    return (series - series.mean()) / series.std()


def _need(df):
    """Combined need index: hours worked, low income, first generation, equally weighted."""
    return (_z(df[HOURS]) - _z(np.log(df[INCOME])) + _z(df[FIRST_GEN])) / np.sqrt(3)


SCORES = {
    "R1_merit": lambda df: df[R_SCORE],
    "R2_no_penalty": lambda df: df[R_SCORE] + COMMITTEE_LOG_INCOME * np.log(df[INCOME]) + COMMITTEE_HOURS * df[HOURS],
    "R3_merit_hours": lambda df: df[R_SCORE] + COMMITTEE_HOURS * df[HOURS],
    "R4a_first_gen_1": lambda df: df[R_SCORE] + 1.0 * df[FIRST_GEN],
    "R4b_first_gen_3": lambda df: df[R_SCORE] + 3.0 * df[FIRST_GEN],
    "R4c_low_income": lambda df: _z(df[R_SCORE]) - 0.5 * _z(np.log(df[INCOME])),
    "R5_within_group": lambda df: df[R_SCORE].groupby(df[GROUP]).rank(pct=True),
    "X1_merit_income": lambda df: df[R_SCORE] + COMMITTEE_LOG_INCOME * np.log(df[INCOME]),
    "X2_need_half": lambda df: _z(df[R_SCORE]) + 0.5 * _need(df),
    "X3_need_full": lambda df: _z(df[R_SCORE]) + _need(df),
}

# French labels: these go on figures and into the report.
LABELS = {
    "R1_merit": "Mérite seul (cote R)",
    "R2_no_penalty": "Comité sans pénalité régionale",
    "R3_merit_hours": "Cote R + heures travaillées",
    "R4a_first_gen_1": "Cote R + 1 pt première génération",
    "R4b_first_gen_3": "Cote R + 3 pts première génération",
    "R4c_low_income": "Cote R, faible revenu favorisé",
    "R5_within_group": "Cote R classée dans chaque groupe",
    "X1_merit_income": "Cote R + revenu",
    "X2_need_half": "Mérite + besoin (poids 0,5)",
    "X3_need_full": "Mérite + besoin (poids 1)",
}

# Hypotheses the selection score takes its minimum over.
PLAUSIBLE = ["R1_merit", "R2_no_penalty", "R3_merit_hours", "R4a_first_gen_1", "R5_within_group"]
# Compatible with the anchor, then dropped: the rules built on them scored about 84% where
# the others scored 92% to 95% (first scorer feedback, 2026-10-03).
DROPPED_BY_FEEDBACK = ["R4b_first_gen_3", "R4c_low_income"]


def score(name, df):
    """Priority score of every row of `df` under hypothesis `name` (higher = more deserving)."""
    return pd.Series(np.asarray(SCORES[name](df), dtype=float), index=df.index)


def label(name, df, rate=DEFAULT_RATE):
    """0/1 array: 1 for the round(rate * n) rows hypothesis `name` ranks highest.

    Ties are broken by row order, so the count is exact, like the allocation.
    """
    values = score(name, df).to_numpy()
    out = np.zeros(len(values), dtype=int)
    out[np.argsort(-values, kind="stable")[: int(round(rate * len(values)))]] = 1
    return out


def labels(df, names=None, rate=DEFAULT_RATE):
    """Table of 0/1 labels, one column per hypothesis."""
    names = list(SCORES) if names is None else names
    return pd.DataFrame({name: label(name, df, rate) for name in names}, index=df.index)
