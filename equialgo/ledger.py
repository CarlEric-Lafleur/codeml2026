"""Run ids, ledger rows and the per-run cache.

The run id is a hash of the full spec, so a spec already in the ledger is skipped. Runs are
deterministic (the seed is in the spec): the ledger is committed, per-run files are not.
"""
import hashlib
import json
import subprocess

import pandas as pd

from .data import ID, LABEL, ROOT, RUNS_DIR
from .pipeline import BASELINE_SPEC, complete, run_spec

LEDGER_PATH = RUNS_DIR / "ledger.csv"


def run_id(spec):
    canonical = json.dumps(complete(spec), sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()[:10]


def git_commit():
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return out.stdout.strip() or "unknown"


def read_ledger():
    if not LEDGER_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LEDGER_PATH)


def cached_decisions(spec, candidates):
    """Decisions of a run on the candidates, from runs/<id>/ or recomputed."""
    path = RUNS_DIR / run_id(spec) / "predictions.csv"
    if path.exists():
        saved = pd.read_csv(path)
        if list(saved[ID]) == list(candidates[ID]):
            return saved[LABEL].to_numpy()
    decisions, _ = run_spec(spec, candidates=candidates)
    _save_decisions(spec, candidates, decisions)
    return decisions


def _save_decisions(spec, candidates, decisions):
    path = RUNS_DIR / run_id(spec) / "predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: candidates[ID], LABEL: decisions}).to_csv(path, index=False)


def record(experiment, spec, history, candidates, force=False):
    """Run a spec and append its row to the ledger. Returns (row, was_run)."""
    spec = complete(spec)
    rid = run_id(spec)
    ledger = read_ledger()
    if not force and len(ledger) and rid in set(ledger["run_id"]):
        return ledger[ledger["run_id"] == rid].iloc[-1].to_dict(), False

    decisions, metrics = run_spec(spec, history, candidates)
    _save_decisions(spec, candidates, decisions)
    baseline = cached_decisions(BASELINE_SPEC, candidates)
    row = {
        "run_id": rid,
        "experiment": experiment,
        **spec,
        "git_commit": git_commit(),
        **metrics,
        "differs_from_baseline": float((decisions != baseline).mean()),
    }
    ledger = pd.concat([ledger[ledger["run_id"] != rid] if len(ledger) else ledger, pd.DataFrame([row])])
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    return row, True
