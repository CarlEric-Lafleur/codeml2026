"""A spec names one choice per stage; this module turns it into decisions and metrics.

Shared by run.py (which logs to the ledger) and model_corrige.py (which does not).
"""
import itertools
import time

from . import evaluate, references
from .allocate import FAIRNESS, MODEL_CUTOFF, allocate
from .data import GROUP, LABEL, REGION, REGIONS, WEIGHT, load_candidates, load_history, split_history
from .features import FEATURE_SETS, uses_region
from .scorers import ALSO_EMBODIES, DECISION_READS_GROUP, EMBODIES, EMBODIES_BY_FEATURES, SCORE_READS_GROUP, SCORERS
from .targets import DEBIASED, TARGETS, training_set

DEFAULTS = {
    "target": "none",
    "features": "merit",
    "scorer": "rule_r",
    "fairness": "none",
    "strength": 0.0,
    "rate": 0.40,
    "seed": 42,
}
# Keys only some scorers read. They get no default, so the run ids of the other specs do not move.
OPTIONAL = {
    "band",  # half-width of the close band, in R points (the `close_*` scorers)
    "hours_cap",  # hours per week counted at most (`rule_r_hours_capped`)
    "program", "program_bonus",  # the program favoured and its bonus in R points (`rule_r_hours_program`)
}
REGISTRIES = {"target": TARGETS, "features": FEATURE_SETS, "scorer": SCORERS, "fairness": FAIRNESS}

# The production model, as in the last cell of baseline_model.ipynb. Never the retained spec.
BASELINE_SPEC = {
    "target": "committee",
    "features": "all",
    "scorer": "random_forest",
    "fairness": "none",
    "strength": 0.0,
    "rate": MODEL_CUTOFF,
    "seed": 42,
}


def complete(spec):
    """Fill defaults and reject unknown stages or component names."""
    unknown = set(spec) - set(DEFAULTS) - OPTIONAL
    if unknown:
        raise KeyError(f"unknown spec keys: {sorted(unknown)}")
    full = {**DEFAULTS, **spec}
    for stage, registry in REGISTRIES.items():
        if full[stage] not in registry:
            raise KeyError(f"unknown {stage} '{full[stage]}', choose from {sorted(registry)}")
    return full


def expand(spec):
    """Any value given as a list expands into a grid of specs."""
    keys = [k for k, v in spec.items() if isinstance(v, list)]
    grid = itertools.product(*(spec[k] for k in keys))
    return [complete({**spec, **dict(zip(keys, values))}) for values in grid]


def spec_uses_region(spec):
    """True if region enters the decision: as a feature, or through a group-aware rule."""
    if spec["scorer"] in SCORE_READS_GROUP:
        return True
    if spec["fairness"] == "debias":  # the correction is in the score: the scorer says
        reads_group = DECISION_READS_GROUP.get(spec["scorer"], lambda spec: False)
        return bool(uses_region(spec["features"]) or reads_group(spec))
    group_rule = spec["fairness"] != "none" and spec["strength"] > 0
    return bool(uses_region(spec["features"]) or group_rule)


def self_references(spec):
    """Hypotheses the spec scores well on by construction; left out of its selection score.

    A scorer that is a hypothesis embodies it, and so does a model trained on a target
    that removes the penalty. Group or region cutoffs on the R score move the rule from R1
    (strength 0) to R5, the R score ranked within each group (strength 1).
    """
    names = [EMBODIES[spec["scorer"]]] if spec["scorer"] in EMBODIES else []
    names += [ALSO_EMBODIES[spec["scorer"]]] if spec["scorer"] in ALSO_EMBODIES else []
    corrected = spec["fairness"] != "debias" or spec["strength"] == 1
    if (spec["scorer"], spec["features"]) in EMBODIES_BY_FEATURES and corrected:
        names.append(EMBODIES_BY_FEATURES[spec["scorer"], spec["features"]])
    trained = spec["scorer"] not in EMBODIES  # a rule stays its own hypothesis whatever its target
    if trained and spec["target"] in DEBIASED and spec["features"] in DEBIASED_EMBODIES:
        names.append(DEBIASED_EMBODIES[spec["features"]])
    if names == ["R1_merit"] and spec["fairness"] in ("group_cutoffs", "region_quotas") and spec["strength"] > 0:
        names.append("R5_within_group")
    return names


# A model trained on a target that removes the penalty (targets.DEBIASED) aims at the
# committee without it: which hypothesis that is depends on what the model may see.
DEBIASED_EMBODIES = {
    "merit": "R1_merit",
    "merit_hours": "R3_merit_hours",
    "merit_income_hours": "R2_no_penalty",
    "no_region_postal_distance": "R2_no_penalty",
}


def short(name):
    """Ledger column prefix of a hypothesis: 'R3_merit_hours' -> 'R3'."""
    return name.split("_")[0]


def panel_metrics(spec, decisions, candidates):
    """Ledger columns for the reference panel: per hypothesis, then the selection score."""
    panel = evaluate.against_panel(decisions, candidates)
    own = self_references(spec)
    points, worst = evaluate.selection_score(panel, own)
    worst_se = evaluate.bootstrap_se(decisions, references.label(worst, candidates), candidates[GROUP])
    judged = panel.drop(index=own, errors="ignore")
    metrics = {"min_points": points, "min_points_se": worst_se["points"], "worst_hypothesis": short(worst),
               "mean_points": float(judged["points"].mean()),
               "self_scored": " ".join(short(n) for n in own)}
    for name, row in panel.iterrows():
        metrics.update({f"{short(name)}_eo_gap": row["eo_gap"], f"{short(name)}_agreement": row["agreement"],
                        f"{short(name)}_points": row["points"]})
    return metrics


def decide(spec, train, target):
    """Train on `train` (if the scorer trains) and return 0/1 decisions for `target`."""
    train, y, weights = training_set(spec["target"], train)
    if weights is not None:
        train = train.assign(**{WEIGHT: weights})
    scores = SCORERS[spec["scorer"]](train, y, target, spec)
    by = target[REGION] if spec["fairness"] == "region_quotas" else target[GROUP]
    return allocate(scores, by, spec["rate"], spec["fairness"], spec["strength"])


def run_spec(spec, history=None, candidates=None):
    """Run one spec twice: on held-out history (committee metrics) and on the candidates.

    The reference panel is label-free, so its metrics are computed on the candidates themselves.

    Returns (decisions on the candidates, dict of metrics).
    """
    spec = complete(spec)
    history = load_history() if history is None else history
    candidates = load_candidates() if candidates is None else candidates
    start = time.perf_counter()

    train, test = split_history(history, seed=spec["seed"])
    held = decide(spec, train, test)
    decisions = decide(spec, history, candidates)

    metrics = {
        "grant_rate": float(decisions.mean()),
        "parity_gap": evaluate.parity_gap(decisions, candidates[GROUP]),
        "heldout_grant_rate": float(held.mean()),
        "heldout_parity_gap": evaluate.parity_gap(held, test[GROUP]),
        "heldout_committee_agreement": evaluate.agreement(held, test[LABEL]),
        "heldout_committee_eo_gap": evaluate.equal_opportunity_gap(held, test[LABEL], test[GROUP]),
        "uses_region": spec_uses_region(spec),
    }
    metrics.update(panel_metrics(spec, decisions, candidates))
    by_region = evaluate.selection_rates(decisions, candidates[REGION])
    metrics.update({f"rate_{r}": float(by_region[r]) for r in REGIONS})
    metrics["runtime_s"] = round(time.perf_counter() - start, 2)
    return decisions, metrics
