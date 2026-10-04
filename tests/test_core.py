"""Allocation, validation and metrics on toy cases."""
import numpy as np
import pandas as pd
import pytest

from equialgo import evaluate
from equialgo.allocate import MODEL_CUTOFF, allocate, top_k
from equialgo.data import CENTRE, ID, LABEL, REMOTE
from equialgo.pipeline import BASELINE_SPEC, complete, expand, self_references
from equialgo.validate import InvalidPredictions, check_predictions

GROUPS = np.array([CENTRE] * 6 + [REMOTE] * 4)
SCORES = np.array([9, 8, 7, 6, 5, 4, 3.5, 3, 2, 1], dtype=float)


def test_top_k_grants_exact_count_with_ties():
    decisions = top_k([1, 1, 1, 1, 1], 2)
    assert decisions.tolist() == [1, 1, 0, 0, 0]


def test_allocate_exact_count_whatever_the_strength():
    for strength in (0, 0.25, 0.5, 0.75, 1):
        decisions = allocate(SCORES, GROUPS, rate=0.4, fairness="group_cutoffs", strength=strength)
        assert decisions.sum() == 4


def test_group_cutoffs_strength_zero_is_one_cutoff():
    free = allocate(SCORES, GROUPS, rate=0.4, fairness="none")
    ruled = allocate(SCORES, GROUPS, rate=0.4, fairness="group_cutoffs", strength=0.0)
    assert free.tolist() == ruled.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]


def test_group_cutoffs_full_strength_equalizes_rates():
    decisions = allocate(SCORES, GROUPS, rate=0.5, fairness="group_cutoffs", strength=1.0)
    assert evaluate.parity_gap(decisions, GROUPS) == pytest.approx(0.0)
    # within a group, the best scores are granted
    assert decisions.tolist() == [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]


def test_model_cutoff_uses_the_half_threshold():
    assert allocate([0.2, 0.5, 0.51], GROUPS[:3], rate=MODEL_CUTOFF).tolist() == [0, 0, 1]


def test_strength_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        allocate(SCORES, GROUPS, rate=0.4, fairness="group_cutoffs", strength=1.5)


def test_metrics_on_a_toy_case():
    decisions = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0])
    label = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0])
    assert evaluate.parity_gap(decisions, GROUPS) == pytest.approx(3 / 6 - 1 / 4)
    assert evaluate.equal_opportunity_gap(decisions, label, GROUPS) == pytest.approx(2 / 3 - 1 / 2)
    assert evaluate.agreement(decisions, label) == pytest.approx(0.7)


def _toy(n=100, granted=40):
    candidates = pd.DataFrame({ID: [f"C{i:06d}" for i in range(n)]})
    predictions = pd.DataFrame({ID: candidates[ID], LABEL: [1] * granted + [0] * (n - granted)})
    return candidates, predictions


def test_valid_predictions_pass():
    candidates, predictions = _toy()
    assert check_predictions(predictions, candidates) == pytest.approx(0.40)


@pytest.mark.parametrize("granted", [0, 30, 50, 100])
def test_out_of_budget_is_blocked(granted):
    candidates, predictions = _toy(granted=granted)
    with pytest.raises(InvalidPredictions):
        check_predictions(predictions, candidates)


def test_bad_ids_rows_and_values_are_blocked():
    candidates, predictions = _toy()
    with pytest.raises(InvalidPredictions):
        check_predictions(predictions.iloc[:-1], candidates)
    with pytest.raises(InvalidPredictions):
        check_predictions(predictions.assign(**{ID: "C000000"}), candidates)
    with pytest.raises(InvalidPredictions):
        check_predictions(predictions.assign(**{LABEL: 2}), candidates)
    with pytest.raises(InvalidPredictions):
        check_predictions(predictions.rename(columns={LABEL: "decision"}), candidates)


def test_spec_lists_expand_into_a_grid():
    specs = expand({"strength": [0.0, 1.0], "rate": [0.39, 0.40, 0.41]})
    assert len(specs) == 6
    assert all(s["scorer"] == "rule_r" for s in specs)


def test_unknown_component_is_rejected():
    with pytest.raises(KeyError):
        complete({"scorer": "does_not_exist"})


def test_self_references_name_the_hypotheses_a_rule_embodies():
    merit = complete({"scorer": "rule_r", "fairness": "group_cutoffs"})
    assert self_references(merit) == ["R1_merit"]
    assert self_references({**merit, "strength": 0.5}) == ["R1_merit", "R5_within_group"]
    assert self_references(complete(BASELINE_SPEC)) == []


def test_committee_merit_learns_the_hours_weight_and_reproduces_the_rule():
    from equialgo.data import load_candidates, load_history
    from equialgo.pipeline import decide
    from equialgo.references import COMMITTEE_HOURS, HOURS
    from equialgo.scorers import committee_weights

    history, candidates = load_history(), load_candidates()
    weights = committee_weights(history, history[LABEL])
    assert weights[HOURS] / weights["cote_r_equivalent"] == pytest.approx(COMMITTEE_HOURS, abs=0.002)
    trained = decide(complete({"target": "committee", "scorer": "committee_merit", "features": "merit_hours"}), history, candidates)
    rule = decide(complete({"scorer": "rule_r_hours", "features": "merit_hours"}), history, candidates)
    assert (trained != rule).mean() < 0.005
    spec = complete({"target": "committee", "scorer": "committee_merit", "features": "merit_hours"})
    assert self_references(spec) == ["R3_merit_hours"]


def test_region_quotas_grant_the_same_rate_in_every_region():
    regions = np.array(["a"] * 4 + ["b"] * 2 + ["c"] * 4)
    decisions = allocate(SCORES, regions, rate=0.5, fairness="region_quotas", strength=1.0)
    assert decisions.tolist() == [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]


def test_row_filters_keep_the_label_and_drop_only_decisions_the_group_decided():
    from equialgo import targets
    from equialgo.data import GROUP, load_history

    history = load_history()
    band, both = targets.penalty_band(history), targets.flipped(history)
    assert band.sum() > 0 and (band <= both).all()
    assert (history.loc[band, GROUP] == REMOTE).all() and (history.loc[band, LABEL] == 0).all()
    kept, y, weights = targets.training_set("drop_flipped", history)
    assert len(kept) == len(y) == len(weights) == len(history) - both.sum()
    assert y.equals(history.loc[kept.index, LABEL])
    soft = targets.soft_weights(history)
    assert soft.min() > 0 and soft.max() == 1
    relabelled = targets.TARGETS["relabel_penalty_band"](history)
    assert (relabelled != history[LABEL]).sum() == band.sum()


def test_a_model_trained_on_a_debiased_target_is_not_judged_on_the_hypothesis_it_aims_at():
    spec = complete({"target": "drop_flipped", "scorer": "logit", "features": "merit_hours"})
    assert self_references(spec) == ["R3_merit_hours"]
    assert self_references({**spec, "target": "committee"}) == []


def test_a_submission_file_is_written_once_and_never_rewritten(tmp_path):
    import submit

    candidates, predictions = _toy()
    path = tmp_path / "01_try.csv"
    first = predictions[LABEL].to_numpy()
    assert submit.write_once(candidates, first, path)[0] == "written"
    assert submit.write_once(candidates, first, path)[0] == "kept"
    assert submit.write_once(candidates, first[::-1], path)[0] == "differs"
    assert pd.read_csv(path)[LABEL].tolist() == first.tolist()


def test_a_second_behaviour_only_moves_the_close_calls():
    from equialgo import references
    from equialgo.data import load_candidates, load_history
    from equialgo.pipeline import decide, spec_uses_region
    from equialgo.scorers import close_calls

    history, candidates = load_history(), load_candidates()
    rule = decide(complete({"scorer": "rule_r_hours"}), history, candidates)
    spec = complete({"scorer": "close_first_gen", "band": 0.5})
    two = decide(spec, history, candidates)
    _, close = close_calls(references.score("R3_merit_hours", candidates), 0.40, 0.5)
    assert two.sum() == rule.sum() and (two != rule).any()
    assert (two[~close] == rule[~close]).all()
    assert (decide({**spec, "band": 0.0}, history, candidates) == rule).all()
    assert self_references(spec) == ["R3_merit_hours"]
    r_first = complete({"scorer": "r_then_hours", "band": 3.0})  # every hour counted: R + hours again
    assert self_references(r_first) == ["R1_merit", "R3_merit_hours"]
    assert (decide(r_first, history, candidates) == rule).all()
    assert spec_uses_region(complete({"scorer": "close_remote"})) and not spec_uses_region(spec)


def test_capped_hours_run_from_the_r_score_to_r_plus_hours():
    from equialgo.data import load_candidates, load_history
    from equialgo.pipeline import decide

    history, candidates = load_history(), load_candidates()
    capped = lambda cap: decide(complete({"scorer": "rule_r_hours_capped", "hours_cap": cap}), history, candidates)
    assert (capped(0) == decide(complete({"scorer": "rule_r"}), history, candidates)).all()
    assert (capped(40) == decide(complete({"scorer": "rule_r_hours"}), history, candidates)).all()
    assert capped(10).sum() == 1600 and (capped(10) != capped(40)).any()
    assert self_references(complete({"scorer": "rule_r_hours_capped", "hours_cap": 10})) == ["R3_merit_hours"]


def test_a_program_bonus_only_moves_grants_toward_that_program():
    from equialgo.data import load_candidates, load_history
    from equialgo.pipeline import decide
    from equialgo.scorers import PROGRAM

    history, candidates = load_history(), load_candidates()
    rule = decide(complete({"scorer": "rule_r_hours"}), history, candidates)
    favour = lambda bonus: decide(complete({"scorer": "rule_r_hours_program", "program": "Sante", "program_bonus": bonus}),
                                  history, candidates)
    sante = (candidates[PROGRAM] == "Sante").to_numpy()
    assert (favour(0.0) == rule).all()
    favoured = favour(0.5)
    assert favoured.sum() == 1600 and (favoured[sante] >= rule[sante]).all() and (favoured[~sante] <= rule[~sante]).all()
    assert favoured[sante].sum() > rule[sante].sum()
    with pytest.raises(KeyError):
        decide(complete({"scorer": "rule_r_hours_program", "program": "Droit", "program_bonus": 0.5}), history, candidates)
    within = decide(complete({"scorer": "rule_r_hours_within_program"}), history, candidates)
    rates = [within[(candidates[PROGRAM] == p).to_numpy()].mean() for p in set(candidates[PROGRAM])]
    assert within.sum() == 1600 and max(rates) - min(rates) < 0.005
    assert self_references(complete({"scorer": "rule_r_hours_within_program"})) == ["R3_merit_hours"]
