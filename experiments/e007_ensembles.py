"""Family 2, continued: ensembles of the models trained without the biased decisions.

The `ensemble` scorer averages the ranks of the logistic regression, the gradient boosting
and the random forest, each trained on the same cleaned set and the same columns (never
region, postal code or distance). To compare with its members (e006) and with the closed
form (`committee_merit`, e003). Ensembles over cleaning methods do not fit one spec: they
are in analysis/a13_ensembles.py.
"""
from equialgo.targets import DEBIASED

SPECS = [
    {"target": DEBIASED, "features": ["merit_hours", "merit_income_hours"], "scorer": "ensemble",
     "fairness": "none", "strength": 0.0, "rate": 0.40},
]
