"""Metrics. Gaps are signed, Centre minus Eloignee: positive means the centres are favoured.

The first half measures decisions against any 0/1 label. The second half measures them
against the reference panel and turns the result into estimated automated points.
The point formulas are our reading of the README, not published code.
"""
import numpy as np
import pandas as pd

from . import references
from .data import CENTRE, GROUP, REMOTE

PUBLISHED_BASELINE_GAP = 0.270  # README: the baseline's equal-opportunity gap against the reference
BUDGET = (0.36, 0.44)
EQUITY_POINTS = 20.0
UTILITY_POINTS = 15.0


def selection_rates(decisions, by):
    """Grant rate per value of `by` (group or region)."""
    return pd.Series(np.asarray(decisions)).groupby(np.asarray(by)).mean()


def parity_gap(decisions, groups):
    rates = selection_rates(decisions, groups)
    return float(rates[CENTRE] - rates[REMOTE])


def true_positive_rates(decisions, label, groups):
    """Among the rows the label calls deserving, the share granted, per group."""
    decisions, label, groups = map(np.asarray, (decisions, label, groups))
    return selection_rates(decisions[label == 1], groups[label == 1])


def equal_opportunity_gap(decisions, label, groups):
    rates = true_positive_rates(decisions, label, groups)
    return float(rates[CENTRE] - rates[REMOTE])


def agreement(decisions, label):
    return float((np.asarray(decisions) == np.asarray(label)).mean())


def equity_points(eo_gap):
    """20 x the share of the published baseline gap closed. Overshooting counts as a gap."""
    return EQUITY_POINTS * float(np.clip(1 - abs(eo_gap) / PUBLISHED_BASELINE_GAP, 0, 1))


def utility_points(decisions, label):
    """15 x agreement, rescaled between a random in-budget draw and a perfect allocation.

    The random draw grants at the same rate as `decisions`. The perfect allocation is the
    label itself, or the closest in-budget allocation when the label's rate is outside it.
    """
    granted, deserving = float(np.mean(decisions)), float(np.mean(label))
    random = granted * deserving + (1 - granted) * (1 - deserving)
    perfect = 1 - max(0.0, deserving - BUDGET[1], BUDGET[0] - deserving)
    share = (agreement(decisions, label) - random) / (perfect - random)
    return UTILITY_POINTS * float(np.clip(share, 0, 1))


def against_label(decisions, label, groups):
    """Gap, agreement and estimated points of `decisions` against one 0/1 label."""
    gap = equal_opportunity_gap(decisions, label, groups)
    equity, utility = equity_points(gap), utility_points(decisions, label)
    return {"eo_gap": gap, "agreement": agreement(decisions, label),
            "equity": equity, "utility": utility, "points": equity + utility}


def against_panel(decisions, table, names=None, rate=references.DEFAULT_RATE):
    """One row per hypothesis: `decisions` on `table` measured against that hypothesis."""
    names = references.PLAUSIBLE if names is None else names
    return pd.DataFrame({name: against_label(decisions, references.label(name, table, rate), table[GROUP])
                         for name in names}).T


def bootstrap_se(decisions, label, groups, n=300, seed=0):
    """Standard errors of eo_gap, agreement and points, resampling applicants (all draws at once)."""
    decisions, label, groups = map(np.asarray, (decisions, label, groups))
    i = np.random.default_rng(seed).integers(0, len(label), (n, len(label)))
    d, y, centre = decisions[i], label[i], groups[i] == CENTRE
    tpr = lambda member: (d * y * member).sum(1) / (y * member).sum(1)
    gap = tpr(centre) - tpr(~centre)
    agree, granted, deserving = (d == y).mean(1), d.mean(1), y.mean(1)
    random = granted * deserving + (1 - granted) * (1 - deserving)
    perfect = 1 - np.maximum(0, np.maximum(deserving - BUDGET[1], BUDGET[0] - deserving))
    points = (EQUITY_POINTS * np.clip(1 - np.abs(gap) / PUBLISHED_BASELINE_GAP, 0, 1)
              + UTILITY_POINTS * np.clip((agree - random) / (perfect - random), 0, 1))
    return {"eo_gap": float(gap.std()), "agreement": float(agree.std()), "points": float(points.std())}


def selection_score(panel, self_scored=()):
    """(points, hypothesis): the worst case over the plausible hypotheses.

    Hypotheses the run embodies are left out: it scores near perfect on them by construction.
    """
    judged = panel.loc[[n for n in panel.index if n in references.PLAUSIBLE and n not in self_scored], "points"]
    return float(judged.min()), judged.idxmin()
