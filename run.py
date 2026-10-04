"""Run experiments and show the ledger.

    python run.py e001            run every spec of experiments/e001_*.py
    python run.py e000 e001       several experiment files
    python run.py --all           every experiment file (specs already in the ledger are skipped)
    python run.py --board         show the ledger, best worst-case estimated points first
    python run.py e001 --force    re-run even if already in the ledger

Experiments never touch the root predictions.csv; only model_corrige.py writes it.
"""
import argparse
import importlib.util
import sys

import pandas as pd

from equialgo.data import ROOT, load_candidates, load_history
from equialgo.ledger import read_ledger, record
from equialgo.pipeline import OPTIONAL, expand, short
from equialgo.references import PLAUSIBLE

EXPERIMENTS_DIR = ROOT / "experiments"
BOARD_COLUMNS = [
    "run_id", "experiment", "target", "features", "scorer", "fairness", "strength", "rate",
    "min_points", "min_points_se", "worst_hypothesis", "mean_points", "self_scored",
    "grant_rate", "parity_gap", "heldout_committee_agreement", "differs_from_baseline", "uses_region",
]
POINTS_COLUMNS = [f"{short(name)}_points" for name in PLAUSIBLE]


def load_experiment(prefix):
    """Return (name, list of specs) for the experiment file starting with `prefix`."""
    matches = sorted(EXPERIMENTS_DIR.glob(f"{prefix}*.py"))
    if len(matches) != 1:
        sys.exit(f"expected one file experiments/{prefix}*.py, found {len(matches)}")
    module_spec = importlib.util.spec_from_file_location(matches[0].stem, matches[0])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # A file holds one spec (SPEC) or several (SPECS); lists inside a spec expand into a grid.
    specs = getattr(module, "SPECS", None) or [module.SPEC]
    return matches[0].stem, [full for spec in specs for full in expand(spec)]


def board():
    ledger = read_ledger()
    if ledger.empty:
        print("ledger is empty")
        return
    # Best worst case first: the selection score. A run's points under a
    # hypothesis it embodies (self_scored) are shown but do not enter min_points or mean_points.
    ledger = ledger.sort_values("min_points", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        columns = BOARD_COLUMNS + [c for c in OPTIONAL if c in ledger]
        print(ledger[columns].round(3).to_string(index=False, na_rep=""))
        print("\nEstimated automated points (of 35) under each plausible hypothesis:")
        print(ledger[["run_id", "experiment", "strength"] + POINTS_COLUMNS].round(dict.fromkeys(POINTS_COLUMNS, 1)).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", help="experiment prefixes, e.g. e001")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--board", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prefixes = list(args.experiments)
    if args.all:
        prefixes = sorted(p.stem.split("_")[0] for p in EXPERIMENTS_DIR.glob("e*.py"))
    if prefixes:
        history, candidates = load_history(), load_candidates()
        for prefix in prefixes:
            name, specs = load_experiment(prefix)
            for spec in specs:
                row, was_run = record(name, spec, history, candidates, force=args.force)
                status = "ran" if was_run else "skipped (in ledger)"
                print(f"{name} {row['run_id']} {status}: rate {row['grant_rate']:.4f}, parity gap {row['parity_gap']:+.4f}, "
                      f"worst case {row['min_points']:.1f} pts ({row['worst_hypothesis']})")
    if args.board or not prefixes:
        board()


if __name__ == "__main__":
    main()
