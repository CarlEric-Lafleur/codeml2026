"""Two behaviours: the retained rule for the clear cases, something else for the close calls.

A close call is an applicant within `band` R points of the cutoff of R + 0.145 x hours.
Outside the band the decision is the retained rule's; inside it, a second behaviour ranks
the applicants, for the same number of grants. Bands: 0.25, 0.5 (the noise of the fitted
reference) and 1.0 R point. Evidence and reading: analysis/a15_close_calls.py.
"""
BANDS = [0.25, 0.5, 1.0]
R_FIRST_BANDS = [0.5, 1.0, 2.0, 3.0]

SPECS = [
    # A second rule inside the band. `features` is a label here: rules ignore it.
    {"target": "none", "features": "merit_hours", "band": BANDS,
     "scorer": ["close_r", "close_first_gen", "close_low_income", "close_remote"]},
    # A trained model inside the band: soft weights against the penalty, every column but the location.
    {"target": "weight_flipped", "features": "no_region_postal_distance", "band": BANDS,
     "scorer": ["close_boosting", "close_local_logit", "close_local_boosting"]},
    # The other way round: a cutoff on the R score alone, the hours adapt the close calls.
    # Wider bands: the hours are worth 0.6 R point between the groups.
    {"target": "none", "features": "merit_hours", "band": R_FIRST_BANDS,
     "scorer": ["r_then_hours", "r_then_hours_alone"]},
]
