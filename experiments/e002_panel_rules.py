"""Family 1: each hypothesis of the reference panel as a rule, judged on the other hypotheses.

No training. Each rule is run with one cutoff for everyone and with full group cutoffs.
`features` only records what the rule reads; R ranked within each group (R5) is e001 at
strength 1.
"""
from equialgo.scorers import RULE_FEATURES

SPECS = [
    {"target": "none", "features": features, "scorer": scorer,
     "fairness": "group_cutoffs", "strength": [0.0, 1.0], "rate": 0.40}
    for scorer, features in RULE_FEATURES.items() if scorer != "rule_r"
]
