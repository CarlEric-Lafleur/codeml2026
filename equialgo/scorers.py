"""Named scoring functions.

Contract: `scorer(train, y, target, spec)` returns one priority score per row of `target`
(higher = granted first). `train` is historical data, `y` its training target (None for
rules). Any idea that fits no existing block becomes a new function here.
"""
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, ExponentiatedGradient, TruePositiveRateParity
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import references
from .data import GROUP, R_SCORE, REMOTE, WEIGHT
from .features import FEATURE_SETS, encode
from .targets import committee_fit, committee_terms


def rule_r(train, y, target, spec):
    """Merit only: the R score itself. No training."""
    return target[R_SCORE].to_numpy(dtype=float)


def rule_r_hours_018(train, y, target, spec):
    """R + 0.18 x hours: the hours weight the scorer's feedback fits best. No training.

    The history supports 0.145 (95% interval 0.130 to 0.161); 0.18 comes from the feedback
    alone, so this rule is a comparison file, not the retained rule.
    """
    return (target[R_SCORE] + 0.18 * target[references.HOURS]).to_numpy(dtype=float)


REMOTE_PENALTY = 1.40  # what being remote cost in the committee's decisions, in R points (a04)
BLIND_HOURS = 0.059  # the hours weight of the committee fitted without the group (a10)


def rule_r_remote_bonus(train, y, target, spec):
    """R score with the committee's penalty handed back: + 1.40 for the remote group. No training.

    The decision reads the group.
    """
    return (target[R_SCORE] + REMOTE_PENALTY * (target[GROUP] == REMOTE)).to_numpy(dtype=float)


def rule_r_hours_0059(train, y, target, spec):
    """R + 0.059 x hours: the weight a group-blind fit of the committee gives. No training.

    That weight has absorbed part of the penalty (0.145 with the group fitted), so this is
    a comparison file, not the retained rule.
    """
    return (target[R_SCORE] + BLIND_HOURS * target[references.HOURS]).to_numpy(dtype=float)


def rule_r_hours_capped(train, y, target, spec):
    """R + 0.145 x hours, the hours counted up to `hours_cap` only (spec key). No training.

    Past the cap an extra hour earns nothing: with a cap above the longest week this is
    R + hours, with a cap of 0 the R score alone.
    """
    counted = np.minimum(target[references.HOURS], spec["hours_cap"])
    return (target[R_SCORE] + references.COMMITTEE_HOURS * counted).to_numpy(dtype=float)


PROGRAM = "programme_etudes"


def rule_r_hours_program(train, y, target, spec):
    """R + 0.145 x hours + `program_bonus` R points for the applicants of `program` (spec keys). No training.

    The history gives no program any weight (a10): the bonus is a test of what the hidden
    reference rewards, not a finding.
    """
    if spec["program"] not in set(target[PROGRAM]):
        raise KeyError(f"unknown program '{spec['program']}', choose from {sorted(set(target[PROGRAM]))}")
    favoured = spec["program_bonus"] * (target[PROGRAM] == spec["program"])
    return (references.score("R3_merit_hours", target) + favoured).to_numpy(dtype=float)


def rule_r_hours_within_program(train, y, target, spec):
    """R + 0.145 x hours ranked inside each program: every program gets the same grant rate. No training."""
    return references.score("R3_merit_hours", target).groupby(target[PROGRAM]).rank(pct=True).to_numpy(dtype=float)


def random_forest(train, y, target, spec):
    """The production model of baseline_model.ipynb (same hyperparameters)."""
    X, Xt = encode(spec["features"], train, target)
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, random_state=spec.get("seed", 42)
    )
    model.fit(X, y, sample_weight=_weights(train))
    return model.predict_proba(Xt)[:, 1]


def _weights(train):
    """Row weights set by the target (targets.ROW_WEIGHTS), or None."""
    return train[WEIGHT].to_numpy() if WEIGHT in train else None


def committee_weights(train, y):
    """Coefficients of the committee's additive logit on R, log income, hours and group."""
    return committee_fit(train, y).drop("const")


def committee_merit(train, y, target, spec):
    """The committee's own judgment with the terms we refuse switched off.

    The committee is fitted with all four terms, group included, so that the weights of the
    other terms are not distorted by the penalty. The priority score then keeps only the
    terms named by the feature set: `merit_hours` drops the group penalty and the income
    preference, `merit_income_hours` drops the penalty only. Region is needed to fit, not
    to decide.

    With the `debias` fairness rule, `strength` is the share of the dropped terms that is
    removed: 0 is the committee as fitted, 1 the full correction. Below 1 the decision
    still reads the group.
    """
    weights = committee_weights(train, y)
    kept = [c for c in FEATURE_SETS[spec["features"]] if c in weights.index and c != GROUP]
    removed = _correction(spec)
    weights = weights.where(weights.index.isin(kept), (1 - removed) * weights)
    return (committee_terms(target) * weights).sum(axis=1).to_numpy()


def logit(train, y, target, spec):
    """Plain logistic regression on the feature set: what "just drop the column" gives."""
    X, Xt = encode(spec["features"], train, target)
    mean, std = X.mean(), X.std().replace(0, 1)
    model = LogisticRegression(C=1e4, max_iter=2000).fit((X - mean) / std, y, sample_weight=_weights(train))
    return model.decision_function((Xt - mean) / std)


def gradient_boosting(train, y, target, spec):
    """Gradient boosting on the feature set."""
    X, Xt = encode(spec["features"], train, target)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=spec.get("seed", 42))
    return model.fit(X, y, sample_weight=_weights(train)).predict_proba(Xt)[:, 1]


ENSEMBLE_MEMBERS = ("logit", "gradient_boosting", "random_forest")


def rank_average(scores):
    """Mean rank (0 to 1) over several score vectors: members on different scales count equally."""
    return np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in scores], axis=0)


def ensemble(train, y, target, spec):
    """Rank average of the logistic regression, the gradient boosting and the random forest."""
    return rank_average([SCORERS[name](train, y, target, spec) for name in ENSEMBLE_MEMBERS])


CLOSE_BAND = 0.5  # default half-width of the close band, in R points: the noise of the fitted reference (a09)


def close_calls(first, rate, band):
    """(margin to the cutoff of the top `rate` share, True where it is within `band`), in units of `first`."""
    first = np.asarray(first, dtype=float)
    margin = first - np.sort(first)[-int(round(rate * len(first)))]
    return margin, np.abs(margin) <= band


def _near_cutoff(train, spec):
    """Historical rows whose R + hours is within the band of the cutoff: the close calls of the past."""
    _, close = close_calls(references.score("R3_merit_hours", train), spec["rate"], spec.get("band", CLOSE_BAND))
    return close


def _local(name):
    """The named scorer trained on the historical close calls only: a specialist of the band."""
    def second(train, y, target, spec):
        near = _near_cutoff(train, spec)
        return SCORERS[name](train[near], y[near], target, spec)
    return second


# What decides inside the band: (priority among the close calls, what it is in words).
CLOSE_SECONDS = {
    "close_r": (lambda train, y, target, spec: target[R_SCORE], "the R score alone"),
    "close_first_gen": (lambda train, y, target, spec: target[references.FIRST_GEN], "first-generation students first"),
    "close_low_income": (lambda train, y, target, spec: -np.log(target[references.INCOME]), "the lowest incomes first"),
    "close_remote": (lambda train, y, target, spec: target[GROUP] == REMOTE, "the remote group first"),
    "close_boosting": (lambda train, y, target, spec: SCORERS["gradient_boosting"](train, y, target, spec),
                       "a gradient boosting trained on the whole history"),
    "close_local_logit": (_local("logit"), "a logistic regression trained on the past close calls only"),
    "close_local_boosting": (_local("gradient_boosting"), "a gradient boosting trained on the past close calls only"),
}
# The other way round: a cutoff on the R score alone, then
# the hours worked adapt the decision of the applicants near it.
R_FIRST_SECONDS = {
    "r_then_hours": (lambda train, y, target, spec: references.score("R3_merit_hours", target),
                     f"R + {references.COMMITTEE_HOURS} x hours"),
    "r_then_hours_alone": (lambda train, y, target, spec: target[references.HOURS], "the most hours worked first"),
}


def _close_call(second, words, first_rule="R3_merit_hours"):
    def scorer(train, y, target, spec):
        first = references.score(first_rule, target).to_numpy()
        margin, close = close_calls(first, spec["rate"], spec.get("band", CLOSE_BAND))
        inside = np.asarray(second(train, y, target, spec), dtype=float)
        # Three tiers: clear grants, close calls, clear refusals. Inside the band the second
        # behaviour ranks, the first breaks its ties; the one-cutoff allocation then grants
        # every clear grant and fills what is left of the budget from the band.
        rank = np.empty(len(first))
        rank[np.lexsort((first, np.where(close, inside, first)))] = np.arange(len(first)) / len(first)
        return np.where(close, 1.0, np.where(margin > 0, 2.0, 0.0)) + rank
    scorer.__doc__ = f"""Two behaviours: {references.LABELS[first_rule]} for the clear cases, {words} for the close calls.

    A close call is an applicant within `band` R points (spec key, default {CLOSE_BAND}) of
    the cutoff of the first rule. The band receives as many grants as that rule alone would
    give it, so only the applicants inside it can change decision.
    """
    return scorer


def _correction(spec):
    """Share of the correction applied: `strength` under the `debias` rule, else all of it."""
    return spec["strength"] if spec.get("fairness") == "debias" else 1.0


def _threshold_optimizer(constraint):
    def scorer(train, y, target, spec):
        X, Xt = encode(spec["features"], train, target)
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, random_state=spec.get("seed", 42))
        model.fit(X, y)
        post = ThresholdOptimizer(estimator=model, constraints=constraint, objective="accuracy_score",
                                  prefit=True, predict_method="predict_proba")
        post.fit(X, y, sensitive_features=train[GROUP])
        adjusted = post.predict(Xt, sensitive_features=target[GROUP], random_state=spec.get("seed", 42))
        share = _correction(spec)
        return share * np.asarray(adjusted, dtype=float) + (1 - share) * model.predict_proba(Xt)[:, 1]
    scorer.__doc__ = f"""fairlearn ThresholdOptimizer ({constraint}) on the production random forest.

    The constraint is measured against the committee's label. The priority blends the
    forest's probability (strength 0) with the post-processed decisions (strength 1); the
    allocation then trims or completes them to the budget. Group thresholds: the decision
    reads the group.
    """
    return scorer


def _exponentiated_gradient(constraint_class, widest=0.12, tightest=0.005):
    def scorer(train, y, target, spec):
        X, Xt = encode(spec["features"], train, target)
        mean, std = X.mean(), X.std().replace(0, 1)
        X, Xt = (X - mean) / std, (Xt - mean) / std
        bound = widest + _correction(spec) * (tightest - widest)
        reduction = ExponentiatedGradient(LogisticRegression(C=1e4, max_iter=2000),
                                          constraints=constraint_class(difference_bound=bound), max_iter=50)
        reduction.fit(X, y, sensitive_features=train[GROUP])
        probabilities = np.column_stack([m.predict_proba(Xt)[:, 1] for m in reduction.predictors_])
        return probabilities @ np.asarray(reduction.weights_, dtype=float)
    scorer.__doc__ = f"""fairlearn ExponentiatedGradient ({constraint_class.__name__}) around a logistic regression.

    The constraint is measured against the committee's label; `strength` tightens its bound
    from {widest} (inactive) to {tightest}. The priority is the weighted probability of the
    ensemble. The group is used to train, not to decide.
    """
    return scorer


def _panel_rule(name):
    """A rule that ranks applicants by the priority score of one panel hypothesis. No training."""
    def rule(train, y, target, spec):
        return references.score(name, target).to_numpy()
    rule.__doc__ = f"The priority score of panel hypothesis {name}. No training."
    return rule


# Scorers that are a hypothesis of the reference panel: a run built on one is not judged on it.
EMBODIES = {
    "rule_r": "R1_merit",
    "rule_no_penalty": "R2_no_penalty",
    "rule_r_hours": "R3_merit_hours",
    "rule_r_hours_018": "R3_merit_hours",
    "rule_r_hours_capped": "R3_merit_hours",  # R + hours below the cap, whatever the cap
    # R + hours inside a program, whatever the program term.
    "rule_r_hours_program": "R3_merit_hours",
    "rule_r_hours_within_program": "R3_merit_hours",
    "rule_first_gen_1": "R4a_first_gen_1",
    "rule_first_gen_3": "R4b_first_gen_3",
    "rule_low_income": "R4c_low_income",
    # Outside the band the two-behaviour models are R + hours, whatever decides inside it.
    **dict.fromkeys(CLOSE_SECONDS, "R3_merit_hours"),
    **dict.fromkeys(R_FIRST_SECONDS, "R1_merit"),
}
# Two-behaviour models whose band follows a second hypothesis: not judged on that one either.
ALSO_EMBODIES = {"close_r": "R1_merit", **dict.fromkeys(R_FIRST_SECONDS, "R3_merit_hours")}

SCORERS = {"rule_r": rule_r, "rule_r_hours_018": rule_r_hours_018, "rule_r_hours_0059": rule_r_hours_0059,
           "rule_r_remote_bonus": rule_r_remote_bonus, "rule_r_hours_capped": rule_r_hours_capped,
           "rule_r_hours_program": rule_r_hours_program, "rule_r_hours_within_program": rule_r_hours_within_program,
           "random_forest": random_forest, "committee_merit": committee_merit,
           "logit": logit, "gradient_boosting": gradient_boosting, "ensemble": ensemble,
           "threshold_optimizer_parity": _threshold_optimizer("demographic_parity"),
           "threshold_optimizer_tpr": _threshold_optimizer("true_positive_rate_parity"),
           "exp_gradient_parity": _exponentiated_gradient(DemographicParity),
           "exp_gradient_tpr": _exponentiated_gradient(TruePositiveRateParity)}
SCORERS.update({name: _close_call(second, words) for name, (second, words) in CLOSE_SECONDS.items()})
SCORERS.update({name: _close_call(second, words, "R1_merit") for name, (second, words) in R_FIRST_SECONDS.items()})

# Scorers whose score itself reads the group, whatever the fairness rule.
SCORE_READS_GROUP = {"rule_r_remote_bonus", "close_remote"}

# Under the `debias` rule the correction lives in the score: does the decision read the group?
DECISION_READS_GROUP = {
    "committee_merit": lambda spec: spec["strength"] < 1,  # the penalty is still in the score
    "threshold_optimizer_parity": lambda spec: spec["strength"] > 0,
    "threshold_optimizer_tpr": lambda spec: spec["strength"] > 0,
}
SCORERS.update({scorer: _panel_rule(name) for scorer, name in EMBODIES.items() if scorer not in SCORERS})

# The corrected committee is a panel hypothesis too, depending on the terms it keeps.
EMBODIES_BY_FEATURES = {
    ("committee_merit", "merit_hours"): "R3_merit_hours",
    ("committee_merit", "merit_income_hours"): "R2_no_penalty",
    ("committee_merit", "merit"): "R1_merit",
}

# What each rule reads, for the `features` stage of a spec (rules ignore it, the ledger shows it).
RULE_FEATURES = {
    "rule_r": "merit",
    "rule_no_penalty": "merit_income_hours",
    "rule_r_hours": "merit_hours",
    "rule_first_gen_1": "merit_first_gen",
    "rule_first_gen_3": "merit_first_gen",
    "rule_low_income": "merit_income",
}
