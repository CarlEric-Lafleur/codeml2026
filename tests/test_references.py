"""Reference panel and estimated points."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from equialgo import evaluate, references
from equialgo.data import CENTRE, GROUP, LABEL, R_SCORE, REMOTE, load_candidates, load_history

GROUPS = np.array([CENTRE] * 6 + [REMOTE] * 4)
LABELS = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])


def test_every_hypothesis_labels_an_exact_share():
    candidates = load_candidates()
    labels = references.labels(candidates)
    assert set(references.PLAUSIBLE + references.DROPPED_BY_FEEDBACK) <= set(labels.columns) == set(references.LABELS)
    assert (labels.sum() == 1600).all()
    assert references.label("R1_merit", candidates, rate=0.36).sum() == 1440


def test_merit_label_is_the_top_r_scores():
    candidates = load_candidates()
    label = references.label("R1_merit", candidates)
    assert candidates.loc[label == 1, R_SCORE].min() >= candidates.loc[label == 0, R_SCORE].max()


def test_within_group_label_is_at_parity():
    candidates = load_candidates()
    label = references.label("R5_within_group", candidates)
    assert abs(evaluate.parity_gap(label, candidates[GROUP])) < 0.002


def test_committee_weights_match_a_refit_of_the_committee():
    history = load_history()
    data = pd.DataFrame({"y": history[LABEL], "r": history[R_SCORE], "linc": np.log(history[references.INCOME]),
                         "hrs": history[references.HOURS], "rem": (history[GROUP] == REMOTE).astype(int)})
    fit = smf.logit("y ~ r + linc + hrs + rem", data).fit(disp=0).params
    assert fit["linc"] / fit["r"] == pytest.approx(references.COMMITTEE_LOG_INCOME, abs=0.01)
    assert fit["hrs"] / fit["r"] == pytest.approx(references.COMMITTEE_HOURS, abs=0.002)


def test_equity_points():
    assert evaluate.equity_points(0.0) == pytest.approx(20)
    assert evaluate.equity_points(0.135) == pytest.approx(10)
    assert evaluate.equity_points(-0.135) == pytest.approx(10)  # overshooting is a gap too
    assert evaluate.equity_points(0.30) == 0


def test_utility_points_between_random_and_perfect():
    assert evaluate.utility_points(LABELS, LABELS) == pytest.approx(15)  # clipped at the top
    # granted 50%, deserving 50%: a random draw agrees half the time, and the best in-budget
    # allocation (44%) misses 6% of the applicants
    half = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    assert evaluate.agreement(half, LABELS) == pytest.approx(0.6)
    assert evaluate.utility_points(half, LABELS) == pytest.approx(15 * (0.6 - 0.5) / (0.94 - 0.5))
    assert evaluate.utility_points(1 - LABELS, LABELS) == 0


def test_bootstrap_se_is_positive_and_small_on_real_sizes():
    candidates = load_candidates()
    label = references.label("R3_merit_hours", candidates)
    decisions = references.label("R1_merit", candidates)
    se = evaluate.bootstrap_se(decisions, label, candidates[GROUP])
    assert 0.005 < se["eo_gap"] < 0.04 and 0 < se["points"] < 3


def test_selection_score_is_the_worst_case_and_skips_self_scored():
    candidates = load_candidates()
    panel = evaluate.against_panel(references.label("R1_merit", candidates), candidates)
    assert panel.loc["R1_merit", "points"] == pytest.approx(35)
    points, worst = evaluate.selection_score(panel, self_scored=["R1_merit"])
    assert worst != "R1_merit" and points == pytest.approx(panel.drop("R1_merit")["points"].min())
    assert evaluate.selection_score(panel)[0] == points  # the minimum was not R1 anyway
