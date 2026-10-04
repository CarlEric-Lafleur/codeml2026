"""Merit only: grant the top 40% by R score, with the group-cutoff rule swept off to full.

No training, so it does not depend on the committee's decisions. Strength 0 is the
skeleton's retained spec.
"""
SPEC = {
    "target": "none",
    "features": "merit",
    "scorer": "rule_r",
    "fairness": "group_cutoffs",
    "strength": [0.0, 0.25, 0.5, 0.75, 1.0],
    "rate": 0.40,
}
