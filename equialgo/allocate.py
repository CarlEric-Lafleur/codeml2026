"""From a ranking to 0/1 decisions under the budget.

The allocation grants an exact count, so the budget cannot be missed by accident whatever
the scorer does. Every fairness rule takes the same `strength` knob: 0 is off (one cutoff
for everyone), 1 is the rule fully applied.
"""
import numpy as np

MODEL_CUTOFF = "model_cutoff"  # rate value meaning "the scorer's own 0.5 threshold"


def top_k(scores, k):
    """Grant the k highest scores. Ties are broken by row order, deterministically."""
    scores = np.asarray(scores, dtype=float)
    decisions = np.zeros(len(scores), dtype=int)
    if k > 0:
        order = np.argsort(-scores, kind="stable")
        decisions[order[:k]] = 1
    return decisions


def _split_count(total, weights):
    """Split an integer total proportionally to weights (largest remainder)."""
    raw = total * np.asarray(weights, dtype=float) / np.sum(weights)
    counts = np.floor(raw).astype(int)
    remainder = total - counts.sum()
    for i in np.argsort(-(raw - counts), kind="stable")[:remainder]:
        counts[i] += 1
    return counts


def none(scores, groups, k, strength):
    """One cutoff for everyone."""
    return top_k(scores, k)


def group_cutoffs(scores, groups, k, strength):
    """One cutoff per group.

    At strength 0 each group gets the count a single cutoff would give it; at strength 1
    each group gets the same grant rate; in between, the counts are interpolated. Within
    a group the highest scores are granted.
    """
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    labels = sorted(set(groups))
    free = top_k(scores, k)
    free_counts = np.array([free[groups == g].sum() for g in labels], dtype=float)
    sizes = np.array([(groups == g).sum() for g in labels], dtype=float)
    equal_counts = k * sizes / sizes.sum()
    target = (1 - strength) * free_counts + strength * equal_counts
    counts = np.minimum(_split_count(k, target), sizes.astype(int))

    decisions = np.zeros(len(scores), dtype=int)
    for g, n in zip(labels, counts):
        idx = np.flatnonzero(groups == g)
        decisions[idx[top_k(scores[idx], n) == 1]] = 1
    return decisions


def region_quotas(scores, regions, k, strength):
    """One cutoff per region: the group-cutoff rule applied to the five regions.

    The pipeline hands this rule the region of each applicant instead of the group.
    """
    return group_cutoffs(scores, regions, k, strength)


def debias(scores, groups, k, strength):
    """One cutoff for everyone; the correction happens in the score.

    Used with scorers that read `strength` themselves: `committee_merit` (share of the
    group penalty and income preference removed) and the fairlearn scorers (how far the
    constraint is applied).
    """
    return top_k(scores, k)


FAIRNESS = {"none": none, "group_cutoffs": group_cutoffs, "region_quotas": region_quotas, "debias": debias}


def allocate(scores, groups, rate=0.40, fairness="none", strength=0.0):
    """Return 0/1 decisions: exactly round(rate * n) grants, shared out by the rule."""
    if rate == MODEL_CUTOFF:
        return (np.asarray(scores, dtype=float) > 0.5).astype(int)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    k = int(round(rate * len(scores)))
    return FAIRNESS[fairness](scores, groups, k, strength)
