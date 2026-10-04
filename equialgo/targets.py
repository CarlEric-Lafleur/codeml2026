"""Named training targets: what a scorer is trained toward.

A target is the label given to the scorer. It may also come with one weight per historical
row (`ROW_WEIGHTS`): 0 drops the row from the training set, a value between 0 and 1
down-weights it. The filters need the group to choose rows; the trained model never sees it.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import GROUP, LABEL, R_SCORE, REMOTE
from .references import HOURS, INCOME


def committee(history):
    """The committee's decision as recorded. Biased; kept as the baseline's target."""
    return history[LABEL]


def none(history):
    """No training target: for rule-based scorers."""
    return None


def committee_terms(df):
    """The four terms of the committee's decision (analysis/a04_committee.py)."""
    return pd.DataFrame({
        R_SCORE: df[R_SCORE].astype(float),
        INCOME: np.log(df[INCOME]),
        HOURS: df[HOURS].astype(float),
        GROUP: (df[GROUP] == REMOTE).astype(float),
    }, index=df.index)


def committee_fit(train, y, terms=(R_SCORE, INCOME, HOURS, GROUP)):
    """Coefficients (constant included) of the committee's additive logit on the named terms."""
    X = sm.add_constant(committee_terms(train)[list(terms)])
    return sm.Logit(np.asarray(y, dtype=float), X).fit(disp=0).params


def _probability(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def _both_ways(history):
    """Probability of a grant under the fitted committee, read as Centre and as Eloignee."""
    params = committee_fit(history, history[LABEL])
    terms = committee_terms(history).drop(columns=GROUP)
    blind = params["const"] + (terms * params[terms.columns]).sum(axis=1)
    return _probability(blind), _probability(blind + params[GROUP])


def penalty_band(history):
    """Remote refusals the fitted committee would have granted to a Centre applicant."""
    as_centre, as_remote = _both_ways(history)
    remote, refused = (history[GROUP] == REMOTE).to_numpy(), (history[LABEL] == 0).to_numpy()
    return remote & refused & (as_centre > 0.5) & (as_remote <= 0.5)


def flipped(history):
    """Decisions the group decides: the penalty band, and its mirror among the centres.

    The mirror: Centre grants the fitted committee would have refused to a remote applicant.
    """
    as_centre, as_remote = _both_ways(history)
    centre, granted = (history[GROUP] != REMOTE).to_numpy(), (history[LABEL] == 1).to_numpy()
    return penalty_band(history) | (centre & granted & (as_centre > 0.5) & (as_remote <= 0.5))


def _keep(rows):
    return lambda history: (~rows(history)).astype(float)


def soft_weights(history):
    """Probability that a decision would have been the same in the other group.

    A remote refusal weighs P(refused as Centre) / P(refused as Eloignee); a Centre grant
    weighs P(granted as Eloignee) / P(granted as Centre). The other rows weigh 1.
    """
    as_centre, as_remote = _both_ways(history)
    remote, granted = (history[GROUP] == REMOTE).to_numpy(), (history[LABEL] == 1).to_numpy()
    weights = np.ones(len(history))
    weights[remote & ~granted] = ((1 - as_centre) / (1 - as_remote))[remote & ~granted]
    weights[~remote & granted] = (as_remote / as_centre)[~remote & granted]
    return weights


def _most_surprising(share):
    def weights(history):
        """Drop the rows a group-blind model of the committee finds most surprising."""
        terms = [R_SCORE, INCOME, HOURS]
        params = committee_fit(history, history[LABEL], terms=terms)
        z = params["const"] + (committee_terms(history)[terms] * params[terms]).sum(axis=1)
        surprise = np.abs(history[LABEL].to_numpy() - _probability(z))
        keep = np.ones(len(history))
        keep[np.argsort(-surprise, kind="stable")[: int(round(share * len(history)))]] = 0.0
        return keep
    return weights


def _relabel(rows):
    def target(history):
        """The committee's decision, reversed on the rows the group decided."""
        return history[LABEL].where(~rows(history), 1 - history[LABEL])
    return target


TARGETS = {"committee": committee, "none": none,
           "relabel_flipped": _relabel(flipped), "relabel_penalty_band": _relabel(penalty_band)}

# Targets that keep the committee's label and filter or weight the training rows instead.
ROW_WEIGHTS = {
    "drop_flipped": _keep(flipped),
    "drop_penalty_band": _keep(penalty_band),
    "drop_residual_5": _most_surprising(0.05),
    "drop_residual_10": _most_surprising(0.10),
    "drop_residual_20": _most_surprising(0.20),
    "weight_flipped": soft_weights,
}
TARGETS.update(dict.fromkeys(ROW_WEIGHTS, committee))

# Targets meant to learn the committee without its penalty: a model trained on one matches
# a panel hypothesis, depending on what it may see (pipeline.self_references).
DEBIASED = ["relabel_flipped", "relabel_penalty_band", "drop_flipped", "drop_penalty_band", "weight_flipped"]


def training_set(name, history):
    """(rows kept, their target, their weights or None) for the named target."""
    y = TARGETS[name](history)
    if name not in ROW_WEIGHTS:
        return history, y, None
    weights = np.asarray(ROW_WEIGHTS[name](history), dtype=float)
    kept = weights > 0
    return history[kept], y[kept], weights[kept]
