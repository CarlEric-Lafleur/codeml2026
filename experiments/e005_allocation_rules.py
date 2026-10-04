"""Family 3: allocation rules on top of the corrected committee, and the rate sweep.

The corrected committee (`committee_merit`, penalty and income preference switched off) is
already at parity, so group cutoffs should move little. Per-region quotas are compared with
the two-group rule: the three remote regions are statistically identical, so
the two should agree up to the noise of five small counts. The rate sweep stays inside the
38% to 42% band we allow ourselves; the panel labels stay at 40%.
"""
CORRECTED = {"target": "committee", "scorer": "committee_merit"}
BY_GROUP_OR_REGION = {"fairness": ["group_cutoffs", "region_quotas"], "strength": [0.5, 1.0], "rate": 0.40}

SPECS = [
    # Cutoffs per group or per region on the corrected committee, with and without income.
    {**CORRECTED, **BY_GROUP_OR_REGION, "features": ["merit_hours", "merit_income_hours"]},
    # Region quotas on the R score alone (group cutoffs on R are e001).
    {"target": "none", "scorer": "rule_r", "features": "merit", **BY_GROUP_OR_REGION, "fairness": "region_quotas"},
    # The retained spec at other grant rates.
    {**CORRECTED, "features": "merit_hours", "fairness": "debias", "strength": 1.0,
     "rate": [0.38, 0.39, 0.41, 0.42]},
]
