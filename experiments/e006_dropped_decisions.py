"""Family 2, continued: train without the decisions that carry the bias.

The model never sees region, postal code or distance. What changes is the training set:

- `drop_flipped`, `drop_penalty_band`: the rows the group decided are removed;
- `drop_residual_5/10/20`: the rows a group-blind model finds most surprising are removed;
- `weight_flipped`: the soft version, rows down-weighted instead of removed;
- `relabel_flipped`, `relabel_penalty_band`: the same rows kept with the decision reversed.

To compare with the closed form (`committee_merit`, e003) and with the same models trained
on the committee's label as is (e003, target `committee`). Analysis: analysis/a11_dropped_decisions.py.
"""
from equialgo.targets import ROW_WEIGHTS

TARGETS = list(ROW_WEIGHTS) + ["relabel_flipped", "relabel_penalty_band"]

SPECS = [
    {"target": TARGETS + ["committee"], "features": "no_region_postal_distance",
     "scorer": ["logit", "gradient_boosting", "random_forest"],
     "fairness": "none", "strength": 0.0, "rate": 0.40},
    {"target": TARGETS, "features": ["merit_hours", "merit_income_hours"],
     "scorer": ["logit", "gradient_boosting", "random_forest"],
     "fairness": "none", "strength": 0.0, "rate": 0.40},
]
