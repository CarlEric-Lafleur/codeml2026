"""R + 0.145 x hours with a program of study favoured.

The committee gave no program any weight, so nothing in the history says which program a
reference would favour or by how much: each program is tried in turn, with a bonus of 0.5 R
point (the noise of the fitted reference) and of 1. The last spec gives every program the
same grant rate. No training. Evidence and reading: analysis/a18_program.py.
"""
PROGRAMS = ["Arts et lettres", "Genie", "Sante", "Sciences", "Sciences sociales"]
BONUSES = [0.5, 1.0]

SPECS = [
    # `features` is a label here: rules ignore it.
    {"target": "none", "features": "merit_hours", "scorer": "rule_r_hours_program", "program": PROGRAMS, "program_bonus": BONUSES},
    {"target": "none", "features": "merit_hours", "scorer": "rule_r_hours_within_program"},
]
