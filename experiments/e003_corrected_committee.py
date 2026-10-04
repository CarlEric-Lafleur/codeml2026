"""Family 2: models trained on the committee's decisions, with and without the correction.

`committee_merit` fits the committee with the group, then scores with the penalty (and the
income preference) switched off: the hours weight is learned from the history, not typed.
The other specs are what "just remove the columns" gives when the same label is learned
without the group: the comparison shows whether the estimator matters and what the proxies
do to the weights.
"""
TRAINED = {"target": "committee", "fairness": "none", "strength": 0.0, "rate": 0.40}

SPECS = [
    # The retained family: share of the penalty and of the income preference removed, 0 to 1.
    {"target": "committee", "scorer": "committee_merit", "features": "merit_hours",
     "fairness": "debias", "strength": [0.0, 0.25, 0.5, 0.75, 1.0], "rate": 0.40},
    {**TRAINED, "scorer": "committee_merit", "features": ["merit_hours", "merit_income_hours"]},
    {**TRAINED, "scorer": ["logit", "gradient_boosting", "random_forest"],
     "features": ["merit_hours", "merit_income_hours", "no_region_postal", "no_region", "all"]},
]
