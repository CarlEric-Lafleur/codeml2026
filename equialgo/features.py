"""Named feature sets: what the scorer may see.

Each entry lists the raw columns kept. `encode` one-hot encodes the categorical ones and
aligns the columns of the target table on those of the training table.
"""
import pandas as pd

from .data import CATEGORICAL, R_SCORE, REGION

NUMERIC = [
    R_SCORE,
    "revenu_familial_estime",
    "heures_travail_semaine",
    "distance_domicile_campus_km",
    "premiere_generation_universitaire",
]
# Same column order as the CSV, so that `all` reproduces the baseline notebook exactly.
ALL = [
    R_SCORE,
    "programme_etudes",
    REGION,
    "code_postal_3",
    "revenu_familial_estime",
    "heures_travail_semaine",
    "distance_domicile_campus_km",
    "premiere_generation_universitaire",
]

FEATURE_SETS = {
    "all": ALL,
    "no_region": [c for c in ALL if c != REGION],
    "no_region_postal": [c for c in ALL if c not in (REGION, "code_postal_3")],
    "no_region_postal_distance": [c for c in ALL if c not in (REGION, "code_postal_3", "distance_domicile_campus_km")],
    "merit": [R_SCORE],
    "merit_hours": [R_SCORE, "heures_travail_semaine"],
    "merit_income": [R_SCORE, "revenu_familial_estime"],
    "merit_income_hours": [R_SCORE, "revenu_familial_estime", "heures_travail_semaine"],
    "merit_first_gen": [R_SCORE, "premiere_generation_universitaire"],
}


def uses_region(name):
    return REGION in FEATURE_SETS[name]


def encode(name, train, target):
    """Return (X_train, X_target) for the named feature set."""
    columns = FEATURE_SETS[name]
    categorical = [c for c in CATEGORICAL if c in columns]
    X = pd.get_dummies(train[columns], columns=categorical)
    Xt = pd.get_dummies(target[columns], columns=categorical).reindex(
        columns=X.columns, fill_value=0
    )
    return X, Xt
