"""Modèle corrigé ÉquiAlgo : méthode retenue, front de Pareto et prédictions.

Méthode retenue : le « comité corrigé ». L'audit montre que le comité décidait selon
quatre éléments : la cote R, les heures travaillées, le revenu familial (favorable aux
revenus élevés) et une pénalité constante pour les régions éloignées. Le modèle :

1. réapprend ces quatre poids sur les décisions passées, région comprise. Ajuster avec la
   région est nécessaire : sans elle, les heures travaillées, qui sont un proxy des
   régions éloignées, absorbent la pénalité et leur poids est sous-estimé ;
2. retire ensuite la pénalité régionale et la préférence pour le revenu, et classe les
   candidats selon la cote R et les heures travaillées seulement.

La région sert donc à apprendre, pas à décider : deux candidats de même cote R et de
mêmes heures reçoivent la même réponse, où qu'ils habitent. Les bourses vont aux 40 % de
candidats les mieux classés.

Le script :
1. construit le modèle retenu à partir de la spécification `RETAINED` ci-dessous ;
2. fait varier la force de la correction de 0 (le comité tel quel) à 1 (pénalité et
   préférence pour le revenu entièrement retirées) et trace le front de Pareto dans
   `figures/pareto.png`, avec pour repères le modèle en production, le même modèle sans
   la région ni le code postal, et les deux méthodes de fairlearn (`ThresholdOptimizer`,
   `ExponentiatedGradient`) balayées de la même façon.
   L'étalon de référence du jury étant inconnu, le front est tracé une fois contre les
   décisions du comité, puis une fois par hypothèse plausible sur cet étalon
   (`equialgo/references.py`) ;
3. écrit `predictions.csv` pour le point retenu, après vérification du format et du budget.

Usage : python model_corrige.py
"""
import pandas as pd

from equialgo import plots, references
from equialgo.data import load_candidates, load_history
from equialgo.pipeline import BASELINE_SPEC, complete, run_spec, self_references, short
from equialgo.validate import write_predictions

# The retained spec. Changing it is a human decision: copy the winning spec from its
# experiment file here, then run this script. Currently experiment e003, full correction
# (retained on 2026-10-03).
RETAINED = {
    "target": "committee",        # the committee's weights are learned from its decisions
    "features": "merit_hours",    # terms kept in the decision: R score and hours worked
    "scorer": "committee_merit",  # committee logit fitted with the group, refused terms removed
    "fairness": "debias",         # one cutoff for everyone; the correction is in the score
    "strength": 1.0,              # share of the penalty and income preference removed
    "rate": 0.40,                 # budget: 36 % to 44 %, we aim for 40 %
}
SWEEP_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Comparison methods, swept over the same strengths. Both learn the committee's decision and
# apply a demographic-parity constraint (experiment e004).
COMPARISONS = [
    ("ThresholdOptimizer (seuils par groupe)",
     {"target": "committee", "features": "all", "scorer": "threshold_optimizer_parity", "fairness": "debias", "rate": 0.40}),
    ("ExponentiatedGradient (réentraînement contraint)",
     {"target": "committee", "features": "no_region_postal", "scorer": "exp_gradient_parity", "fairness": "debias", "rate": 0.40}),
]
# The reflex the README warns about: the production model without region or postal code.
WITHOUT_REGION = {**BASELINE_SPEC, "features": "no_region_postal", "rate": 0.40}


def sweep(spec, history, candidates):
    """Metrics of the spec at each fairness strength, one row per strength."""
    rows = []
    for strength in SWEEP_STRENGTHS:
        _, metrics = run_spec({**spec, "strength": strength}, history, candidates)
        rows.append({"strength": strength, **metrics})
    return pd.DataFrame(rows)


def main():
    history, candidates = load_history(), load_candidates()
    retained = complete(RETAINED)

    front = sweep(retained, history, candidates)
    _, baseline = run_spec(BASELINE_SPEC, history, candidates)
    _, without_region = run_spec(WITHOUT_REGION, history, candidates)
    comparisons = [(label, sweep(spec, history, candidates)) for label, spec in COMPARISONS]
    print("Front de Pareto contre le comité (historique mis de côté, 30 %) :")
    print(front[["strength", "heldout_parity_gap", "heldout_committee_agreement",
                 "grant_rate", "parity_gap"]].round(4).to_string(index=False))
    print("\nPoints automatiques estimés (sur 35) selon l'hypothèse sur l'étalon, 4 000 candidats :")
    points = [f"{short(name)}_points" for name in references.PLAUSIBLE]
    print(front[["strength", "min_points", "worst_hypothesis"] + points]
          .round(dict.fromkeys(["min_points"] + points, 1)).to_string(index=False))

    # A hypothesis the rule embodies at some strength scores perfectly there by construction.
    own = {name for strength in SWEEP_STRENGTHS for name in self_references({**retained, "strength": strength})}
    panels = [("heldout_parity_gap", "heldout_committee_agreement", "Comité (écart de taux d'octroi)")]
    for name in references.PLAUSIBLE:
        note = "\n(par construction)" if name in own else ""
        panels.append((f"{short(name)}_eo_gap", f"{short(name)}_agreement", references.LABELS[name] + note))
    path = plots.pareto(
        front, baseline, retained["strength"], panels,
        title="Compromis équité–utilité selon l'étalon de référence supposé",
        xlabel="Écart d'égalité des chances entre les groupes (valeur absolue)",
        ylabel="Accord avec l'étalon",
        comparisons=comparisons, points=[("Production sans région ni code postal", without_region)],
    )
    print(f"Figure : {path.name}")
    # The same front on three yardsticks only, for the slides: the committee, the standard
    # least favourable to the retained model, and the one it embodies.
    summary = [panels[0]] + [panel for panel in panels[1:] if panel[0].startswith(("R1_", "R3_"))]
    path = plots.pareto(
        front, baseline, retained["strength"], summary,
        title="Compromis équité–utilité quand on retire la pénalité (0 = le comité, 1 = corrigé)",
        xlabel="Écart entre les groupes (valeur absolue)", ylabel="Accord avec l'étalon",
        comparisons=comparisons, points=[("Production sans région ni code postal", without_region)],
        name="pareto_resume.png",
    )
    print(f"Figure : {path.name}")

    print("\nMéthodes de comparaison, contrainte complète (force 1), 4 000 candidats :")
    rows = {"Modèle retenu": front.iloc[-1], "Modèle en production": pd.Series(baseline),
            "Production sans région ni code postal": pd.Series(without_region)}
    rows.update({label: table.iloc[-1] for label, table in comparisons})
    columns = {"parity_gap": "écart de parité", "heldout_committee_agreement": "accord avec le comité",
               "min_points": "points estimés, pire cas", "uses_region": "la décision lit la région"}
    print(pd.DataFrame(rows).T[list(columns)].rename(columns=columns).to_string(float_format=lambda v: f"{v:.3f}"))

    decisions, metrics = run_spec(retained, history, candidates)
    rate = write_predictions(candidates, decisions)
    print(f"\npredictions.csv écrit : {len(decisions)} lignes, taux d'octroi {rate:.2%}")
    print(f"Écart de parité (Centre - Eloignee) : {metrics['parity_gap']:+.4f}"
          f"   (modèle en production : {baseline['parity_gap']:+.4f})")
    print(f"Points estimés dans le pire cas : {metrics['min_points']:.1f} sur 35, sous l'hypothèse "
          f"{metrics['worst_hypothesis']} (modèle en production : {baseline['min_points']:.1f})")
    print("Taux d'octroi par région :")
    for key, value in metrics.items():
        if key.startswith("rate_"):
            print(f"  {key[5:]:<32} {value:.2%}   (production : {baseline[key]:.2%})")


if __name__ == "__main__":
    main()
