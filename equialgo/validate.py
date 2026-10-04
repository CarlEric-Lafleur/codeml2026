"""The checks on predictions.csv: row count, id set, value set, grant rate. A failed check blocks the write of predictions.csv."""
import pandas as pd

from .data import ID, LABEL, ROOT

PREDICTIONS_PATH = ROOT / "predictions.csv"
RULE_BAND = (0.36, 0.44)  # outside: the technical section scores 0
OUR_BAND = (0.38, 0.42)  # the margin we keep inside the rule


class InvalidPredictions(ValueError):
    pass


def check_predictions(predictions, candidates, band=OUR_BAND):
    """Raise InvalidPredictions unless the table is a valid, in-budget submission."""
    if list(predictions.columns) != [ID, LABEL]:
        raise InvalidPredictions(f"columns must be [{ID}, {LABEL}], got {list(predictions.columns)}")
    if len(predictions) != len(candidates):
        raise InvalidPredictions(f"expected {len(candidates)} rows, got {len(predictions)}")
    if predictions[ID].duplicated().any():
        raise InvalidPredictions("duplicated ids")
    if set(predictions[ID]) != set(candidates[ID]):
        raise InvalidPredictions("ids differ from the candidate file")
    if not set(predictions[LABEL].unique()) <= {0, 1}:
        raise InvalidPredictions("decisions must be 0 or 1")
    rate = predictions[LABEL].mean()
    if not band[0] <= rate <= band[1]:
        raise InvalidPredictions(f"grant rate {rate:.4f} outside {band}")
    return rate


def write_predictions(candidates, decisions, path=PREDICTIONS_PATH):
    """Check, then write. Only model_corrige.py calls this for the root predictions.csv."""
    predictions = pd.DataFrame({ID: candidates[ID].to_numpy(), LABEL: pd.Series(decisions).astype(int).to_numpy()})
    rate = check_predictions(predictions, candidates)
    predictions.to_csv(path, index=False)
    return rate


def check_file(path=PREDICTIONS_PATH, candidates=None, band=OUR_BAND):
    """Re-read a written file and run the same checks (used by build.py)."""
    from .data import load_candidates

    candidates = load_candidates() if candidates is None else candidates
    return check_predictions(pd.read_csv(path), candidates, band)
