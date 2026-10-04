"""Does training without the biased decisions (T4) do anything the closed-form correction does not (T3)?
Do the estimators differ (T2), does a weight grid cover them (T1), do region quotas differ from group cutoffs (T5)?

Reads the decisions of experiments e005 and e006 from runs/ and recomputes the missing ones:
run `run.py e005 e006` first, otherwise this script takes several minutes.
"""
import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression

from common import CENTRE, GROUP, REMOTE, load, pd, section
from equialgo import evaluate, references, targets
from equialgo.data import LABEL, REGION, REGIONS
from equialgo.ledger import cached_decisions
from equialgo.pipeline import complete

history, candidates = load()
groups = candidates[GROUP]
SCORERS = ["logit", "gradient_boosting", "random_forest"]
FEATURES = ["merit_hours", "merit_income_hours", "no_region_postal_distance"]
TARGETS = ["committee"] + list(targets.ROW_WEIGHTS) + ["relabel_flipped", "relabel_penalty_band"]
# The closed form each feature set aims at: the committee without penalty, with or without income.
CLOSED_FORM = {"merit_hours": "R3_merit_hours", "merit_income_hours": "R2_no_penalty", "no_region_postal_distance": "R2_no_penalty"}


def decisions_of(**spec):
    return cached_decisions(complete({"fairness": "none", "strength": 0.0, "rate": 0.40, **spec}), candidates)


section("T4. What each filter removes from the 10,000 historical rows")
rows = {}
for name in targets.ROW_WEIGHTS:
    weight = np.asarray(targets.ROW_WEIGHTS[name](history))
    lost = 1 - weight
    remote, granted = (history[GROUP] == REMOTE).to_numpy(), (history[LABEL] == 1).to_numpy()
    rows[name] = {"rows touched": int((weight < 1).sum()), "weight removed": lost.sum(),
                  "remote refusals": lost[remote & ~granted].sum(), "remote grants": lost[remote & granted].sum(),
                  "centre refusals": lost[~remote & ~granted].sum(), "centre grants": lost[~remote & granted].sum(),
                  "grant rate left, Centre": np.average(granted[~remote], weights=weight[~remote]),
                  "grant rate left, Eloignee": np.average(granted[remote], weights=weight[remote])}
print(pd.DataFrame(rows).T.round(3).to_string())
print("The residual filters are group-blind: they remove noise on both sides and leave the gap in place.")

section("T4. Hours weight (R points per hour) a logit on R and hours learns from each training set")
X = history[["r", "hrs"]]
rows = {}
for name in TARGETS:
    kept, y, weight = targets.training_set(name, history)
    coef = LogisticRegression(C=1e4, max_iter=2000).fit(X.loc[kept.index], y, sample_weight=weight).coef_[0]
    rows[name] = coef[1] / coef[0]
print(pd.Series(rows).round(3).to_string())
print(f"Closed form (committee fitted with the group): {references.COMMITTEE_HOURS}.")

section("T3/T4. Decisions on the 4,000: parity gap, and agreement with the closed form the feature set aims at")
decisions, rows = {}, []
for target, features, scorer in itertools.product(TARGETS, FEATURES, SCORERS):
    d = decisions[target, features, scorer] = decisions_of(target=target, features=features, scorer=scorer)
    rows.append({"target": target, "features": features, "scorer": scorer, "parity gap": evaluate.parity_gap(d, groups),
                 "agreement": evaluate.agreement(d, references.label(CLOSED_FORM[features], candidates))})
table = pd.DataFrame(rows)
for value in ("parity gap", "agreement"):
    print(f"\n{value}:")
    print(table.pivot(index="target", columns=["features", "scorer"], values=value).loc[TARGETS, FEATURES].round(3).to_string())

section("T2. Do the three estimators decide differently? Share of the 4,000 on which two of them differ (largest pair)")
rows = {target: {features: max((decisions[target, features, a] != decisions[target, features, b]).mean()
                               for a, b in itertools.combinations(SCORERS, 2)) for features in FEATURES}
        for target in TARGETS}
print(pd.DataFrame(rows).T.round(3).to_string())

section("T1. Is every trained model close to a rule R + a x log income + b x hours + c x first generation?")
grid = list(itertools.product(np.arange(0, 2.01, 0.25), np.arange(0, 0.3001, 0.02), (0.0, 0.5, 1.0)))
rule_decisions = np.array([references.label("R1_merit", candidates.assign(cote_r_equivalent=candidates["r"] + a * candidates["linc"]
                                                                          + b * candidates["hrs"] + c * candidates["fg"]))
                           for a, b, c in grid])
rows = []
for key, d in decisions.items():
    agree = (rule_decisions == d).mean(axis=1)
    a, b, c = grid[int(agree.argmax())]
    rows.append({"target": key[0], "features": key[1], "scorer": key[2], "best rule agreement": agree.max(),
                 "log income": a, "hours": round(b, 2), "first gen": c})
cover = pd.DataFrame(rows)
print(cover.groupby("scorer")["best rule agreement"].describe()[["min", "mean", "max"]].round(3).to_string())
print("\nLeast well covered:")
print(cover.sort_values("best rule agreement").head(8).round(3).to_string(index=False))

section("T5. Region quotas against group cutoffs, full strength, on the 4,000")
rows = {}
for label, base in {"R score": {"target": "none", "scorer": "rule_r", "features": "merit"},
                    "corrected committee (R + hours)": {"target": "committee", "scorer": "committee_merit", "features": "merit_hours"}}.items():
    by_group = decisions_of(**base, fairness="group_cutoffs", strength=1.0)
    by_region = decisions_of(**base, fairness="region_quotas", strength=1.0)
    rows[label] = {"decisions that differ": (by_group != by_region).mean(),
                   **{f"group cutoffs, {r}": by_group[(candidates[REGION] == r).to_numpy()].mean() for r in REGIONS}}
print(pd.DataFrame(rows).round(4).to_string())
print(f"(Region quotas grant 40% in each region by construction; {CENTRE} and {REMOTE} hold 2 and 3 regions.)")
