"""R + 0.145 x hours with the hours counted up to a cap only.

Past the cap an extra hour of work earns nothing. The weight stays the committee's; only the
cap moves. No training. Evidence and reading: analysis/a17_capped_hours.py.
"""
CAPS = [8, 10, 12, 15, 20]

SPECS = [
    # `features` is a label here: rules ignore it.
    {"target": "none", "features": "merit_hours", "scorer": "rule_r_hours_capped", "hours_cap": CAPS},
]
