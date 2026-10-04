"""Family 4: the two fairlearn methods the README names, as comparison points for the Pareto front.

Both take the committee's decision as the label, so their constraint is measured against
the biased label. `strength` sweeps the constraint from off to full.

- ThresholdOptimizer post-processes the production random forest with one threshold per
  group (the decision reads the group).
- ExponentiatedGradient retrains a logistic regression under the constraint, without region
  or postal code among the inputs (the group is used to train only).
"""
SWEEP = {"target": "committee", "fairness": "debias", "strength": [0.0, 0.25, 0.5, 0.75, 1.0], "rate": 0.40}

SPECS = [
    {**SWEEP, "scorer": ["threshold_optimizer_parity", "threshold_optimizer_tpr"], "features": "all"},
    {**SWEEP, "scorer": ["exp_gradient_parity", "exp_gradient_tpr"], "features": "no_region_postal"},
]
