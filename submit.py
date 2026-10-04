"""Prepare files for the subset scorer and read what comes back.

    python submit.py          write submissions/<name>.csv for each entry below, update log.csv
    python submit.py --fit    compare the returned scores with what each hypothesis predicted

Every file is a legitimate try: an in-budget rule we would keep as
final. The files come from different families, so their scores tell the hypotheses about
the hidden reference apart. The feedback chooses hypotheses and families; it is never used
to tune strength, rate or hyperparameters.

The scorer returns an accuracy and a macro F1 against the reference, in percent. After a
submission, type them into the `returned_*` columns of submissions/log.csv. Re-running
this script keeps them.

Every submission is kept. A file in submissions/ is written once and never rewritten: if
an entry below no longer produces the decisions of its file, the file and its log row stay
as they are and the script says so. A changed model is a new entry with a new name. An
entry removed from the list keeps its file and its log row.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from equialgo import evaluate, references
from equialgo.data import ID, LABEL, ROOT, load_candidates, load_history
from equialgo.ledger import run_id
from equialgo.pipeline import complete, run_spec, short
from equialgo.validate import check_file, write_predictions

SUBMISSIONS_DIR = ROOT / "submissions"
LOG_PATH = SUBMISSIONS_DIR / "log.csv"
RETURNED = ["returned_accuracy", "returned_f1_macro", "returned_notes"]

# The dropped hypotheses stay in the log: it records what each one predicted.
PANEL = references.PLAUSIBLE + references.DROPPED_BY_FEEDBACK

RULE = {"target": "none", "fairness": "group_cutoffs", "rate": 0.40}
TRAINED = {"features": "merit_hours", "fairness": "none", "strength": 0.0, "rate": 0.40}
CLOSE = {"target": "none", "features": "merit_hours", "fairness": "none", "strength": 0.0, "rate": 0.40, "band": 0.5}
FAIRLEARN = {"target": "committee", "fairness": "debias", "strength": 1.0, "rate": 0.40}
# In submission order: the first two separate the two leading hypotheses.
SUBMISSIONS = [
    ("01_merit_only", {**RULE, "scorer": "rule_r", "features": "merit", "strength": 0.0}),
    ("02_r_plus_hours", {**RULE, "scorer": "rule_r_hours", "features": "merit_hours", "strength": 0.0}),
    ("03_r_within_group", {**RULE, "scorer": "rule_r", "features": "merit", "strength": 1.0}),
    ("04_committee_no_penalty", {**RULE, "scorer": "rule_no_penalty", "features": "merit_income_hours", "strength": 0.0}),
    ("05_r_first_gen_3pts", {**RULE, "scorer": "rule_first_gen_3", "features": "merit_first_gen", "strength": 0.0}),
    ("06_r_low_income", {**RULE, "scorer": "rule_low_income", "features": "merit_income", "strength": 0.0}),
    # Second batch: other routes to the corrected committee. Each is close to file 02 by
    # design; the models never read region, postal code or distance unless said otherwise.
    ("07_weighted_logit", {**TRAINED, "target": "weight_flipped", "scorer": "logit"}),
    ("08_relabelled_logit", {**TRAINED, "target": "relabel_flipped", "scorer": "logit"}),
    ("09_dropped_forest", {**TRAINED, "target": "drop_flipped", "scorer": "random_forest"}),
    ("10_dropped_band_boosting", {**TRAINED, "target": "drop_penalty_band", "scorer": "gradient_boosting"}),
    # The corrected committee with one cutoff per region (the decision reads the region).
    ("11_corrected_region_quotas", {"target": "committee", "scorer": "committee_merit", "features": "merit_hours",
                                    "fairness": "region_quotas", "strength": 1.0, "rate": 0.40}),
    # The two fairlearn methods under a parity constraint, full strength (e004).
    ("12_exp_gradient_parity", {**FAIRLEARN, "scorer": "exp_gradient_parity", "features": "no_region_postal"}),
    ("13_threshold_optimizer_parity", {**FAIRLEARN, "scorer": "threshold_optimizer_parity", "features": "all"}),
    # The hours weight the feedback fits best. Not from the history.
    ("14_r_plus_018_hours", {**RULE, "scorer": "rule_r_hours_018", "features": "merit_hours", "strength": 0.0}),
    # The committee's penalty handed back to the remote group
    # (the decision reads the group), and the hours weight of the group-blind fit.
    ("15_r_plus_remote_bonus", {**RULE, "scorer": "rule_r_remote_bonus", "features": "merit", "strength": 0.0}),
    ("16_r_plus_0059_hours", {**RULE, "scorer": "rule_r_hours_0059", "features": "merit_hours", "strength": 0.0}),
    # Two behaviours (e008): one rule for the clear cases, another
    # for the applicants within 0.5 R point of its cutoff. 17 to 19: R + hours, then first
    # generation, the R score alone, or a logit trained on the past close calls.
    ("17_close_first_gen", {**CLOSE, "scorer": "close_first_gen"}),
    ("18_close_r", {**CLOSE, "scorer": "close_r"}),
    ("19_close_local_logit", {**CLOSE, "scorer": "close_local_logit", "target": "weight_flipped",
                              "features": "no_region_postal_distance"}),
    # 20, 21: a cutoff on the R score alone, then the hours adapt the close calls
    # (R + 0.145 x hours inside the band, or the most hours first).
    ("20_r_then_hours", {**CLOSE, "scorer": "r_then_hours"}),
    ("21_r_then_hours_alone", {**CLOSE, "scorer": "r_then_hours_alone"}),
    # R + 0.145 x hours with the hours counted up to a cap (e009).
    # No file for a cap of 20: it differs from file 02 on four applicants only, so its score
    # would be information about those four rows, not about the rule.
    ("22_r_plus_hours_cap_10", {**RULE, "scorer": "rule_r_hours_capped", "features": "merit_hours", "strength": 0.0, "hours_cap": 10}),
    ("23_r_plus_hours_cap_15", {**RULE, "scorer": "rule_r_hours_capped", "features": "merit_hours", "strength": 0.0, "hours_cap": 15}),
    # R + 0.145 x hours with 0.5 R point for one program of study, each in turn
    # (e010, a18). They are read against file 02: a program the reference favours
    # returns more than it, any other less. No file for "the same rate in every program": 46
    # applicants away from file 02, and its score would not tell the two readings apart.
    *[(f"{number}_r_plus_hours_{slug}", {**RULE, "scorer": "rule_r_hours_program", "features": "merit_hours", "strength": 0.0,
                                         "program": program, "program_bonus": 0.5})
      for number, slug, program in [(24, "arts", "Arts et lettres"), (25, "genie", "Genie"), (26, "sante", "Sante"),
                                    (27, "sciences", "Sciences"), (28, "sciences_sociales", "Sciences sociales")]],
]


def write_once(candidates, decisions, path):
    """Write a submission file unless it exists. Returns (status, grant rate of the file).

    Status: "written" (new file), "kept" (the file already holds these decisions) or
    "differs" (the file holds other decisions and is left untouched).
    """
    if not path.exists():
        return "written", write_predictions(candidates, decisions, path)
    saved = pd.read_csv(path).set_index(ID)[LABEL].reindex(candidates[ID]).to_numpy()
    same = np.array_equal(saved, np.asarray(decisions))
    return ("kept" if same else "differs"), check_file(path, candidates)


def prepare():
    history, candidates = load_history(), load_candidates()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    names = [name for name, _ in SUBMISSIONS]
    if len(set(names)) != len(names):
        raise SystemExit("two entries of SUBMISSIONS share a name")
    rows, differs = [], []
    for name, spec in SUBMISSIONS:
        spec = complete(spec)
        decisions, metrics = run_spec(spec, history, candidates)
        status, rate = write_once(candidates, decisions, SUBMISSIONS_DIR / f"{name}.csv")
        if status == "differs":  # the file and its log row stay as they were submitted
            differs.append(f"{name}.csv")
            continue
        row = {"file": f"{name}.csv", "run_id": run_id(spec), "spec": json.dumps(spec, sort_keys=True),
               "grant_rate": rate, "parity_gap": metrics["parity_gap"], "uses_region": metrics["uses_region"]}
        panel = evaluate.against_panel(decisions, candidates, PANEL)
        for hypothesis, measured in panel.iterrows():
            label = references.label(hypothesis, candidates)
            row[f"{short(hypothesis)}_accuracy"] = round(100 * measured["agreement"], 2)
            row[f"{short(hypothesis)}_f1_macro"] = round(100 * f1_score(label, decisions, average="macro"), 2)
            row[f"{short(hypothesis)}_eo_gap"] = round(measured["eo_gap"], 3)
            row[f"{short(hypothesis)}_points"] = round(measured["points"], 1)
        rows.append(row)
    log = pd.DataFrame(rows, columns=None if rows else ["file"])
    previous = pd.read_csv(LOG_PATH) if LOG_PATH.exists() else pd.DataFrame(columns=["file"] + RETURNED)
    log = log.merge(previous[["file"] + RETURNED], on="file", how="left")
    # Rows of files this run did not regenerate (changed or removed entries) are carried over.
    log = pd.concat([log, previous[~previous["file"].isin(log["file"])]], ignore_index=True).sort_values("file")
    log = log[["file"] + RETURNED + [c for c in log.columns if c not in ["file"] + RETURNED]]
    log.to_csv(LOG_PATH, index=False)
    untracked = sorted(p.name for p in SUBMISSIONS_DIR.glob("*.csv") if p.name not in set(log["file"]) | {LOG_PATH.name})

    accuracy = [c for c in log.columns if c.endswith("_accuracy") and not c.startswith("returned")]
    print("Files in submissions/ (existing files are never rewritten). Predicted accuracy (%) if the hypothesis is the reference:")
    print(log[["file", "returned_accuracy"] + accuracy].to_string(index=False, na_rep=""))
    print(f"\nLog: {LOG_PATH.relative_to(ROOT)}. Fill in the returned_* columns after each submission.")
    for name in differs:
        print(f"NOT REWRITTEN: {name} holds other decisions than its entry now produces. The file and its "
              "log row are kept as submitted; give the changed model a new name.")
    if untracked:
        print("In submissions/ without a log row (left untouched):", ", ".join(untracked))


def fit():
    """Rank the hypotheses by how well they predicted the returned scores."""
    log = pd.read_csv(LOG_PATH)
    kinds = {"accuracy": "returned_accuracy", "f1_macro": "returned_f1_macro"}
    scored = log[log[list(kinds.values())].notna().any(axis=1)]
    if scored.empty:
        print("No returned score in submissions/log.csv yet.")
        return
    rows = {}
    for hypothesis in PANEL:
        errors = pd.concat([scored[f"{short(hypothesis)}_{kind}"] - scored[column] for kind, column in kinds.items()]).dropna()
        rows[hypothesis] = {"rms error": float(np.sqrt((errors ** 2).mean())),
                            **{file[:2]: value for file, value in zip(scored["file"], scored[f"{short(hypothesis)}_accuracy"])}}
    table = pd.DataFrame(rows).T.sort_values("rms error")
    table.loc["returned"] = {"rms error": np.nan, **{file[:2]: value for file, value in zip(scored["file"], scored["returned_accuracy"])}}
    print("Predicted accuracy (%) of each file under each hypothesis, and the rms error over accuracy and F1")
    print("(percentage points; smaller = more credible):")
    print(table.round(2).to_string(na_rep=""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit", action="store_true")
    fit() if parser.parse_args().fit else prepare()
