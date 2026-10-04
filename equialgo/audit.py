"""Tables and figures for the audit report. Each figure is also saved to figures/ for the slides.

Output labels are in French: they appear as is in the notebook and on the slides.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from . import evaluate, plots, references
from .data import CENTRE, GROUP, GROUPS, ID, LABEL, R_SCORE, REGION, REMOTE, ROOT, split_history
from .pipeline import BASELINE_SPEC, complete, decide

INCOME, HOURS, FIRST_GEN = references.INCOME, references.HOURS, references.FIRST_GEN
DISTANCE, PROGRAM, POSTAL = "distance_domicile_campus_km", "programme_etudes", "code_postal_3"

R_BAND_EDGES = [0, 24, 26, 28, 30, 32, 100]
R_BAND_LABELS = ["< 24", "24–26", "26–28", "28–30", "30–32", "≥ 32"]

COLUMN_NAMES = {
    R_SCORE: "Cote R", INCOME: "Revenu familial", HOURS: "Heures de travail", DISTANCE: "Distance au campus",
    FIRST_GEN: "Première génération", PROGRAM: "Programme d'études", POSTAL: "Code postal (3 car.)",
}
TERM_NAMES = {"r": "Cote R (1 point)", "linc": "Revenu familial (log)", "hrs": "Heures de travail (1 h)",
              "rem": "Région éloignée"}


# --- Section 1 and 2: the data and the raw gap ------------------------------------------------

def file_overview(history, candidates):
    """Rows, group mix and committee grant rate of each file."""
    rows = {}
    for name, df in (("donnees_demandes.csv (historique)", history), ("candidats_evaluation.csv (à évaluer)", candidates)):
        rows[name] = {"lignes": len(df), "part grands centres": (df[GROUP] == CENTRE).mean(),
                      "part régions éloignées": (df[GROUP] == REMOTE).mean(),
                      "taux d'octroi du comité": df[LABEL].mean() if LABEL in df else np.nan}
    return pd.DataFrame(rows).T.round(3)


def group_profiles(history):
    """Mean profile of each group: the columns on which the two groups differ."""
    table = history.groupby(GROUP).agg(**{
        "cote R moyenne": (R_SCORE, "mean"), "revenu médian": (INCOME, "median"),
        "heures de travail": (HOURS, "mean"), "distance médiane (km)": (DISTANCE, "median"),
        "première génération": (FIRST_GEN, "mean"), "taux d'octroi du comité": (LABEL, "mean")})
    return table.round(2).T


def grant_rate_table(history, by=GROUP):
    """Applications, committee grant rate and mean R score per group (or per region)."""
    table = history.groupby(by).agg(
        demandes=(LABEL, "size"),
        taux_octroi=(LABEL, "mean"),
        cote_r_moyenne=(R_SCORE, "mean"),
    )
    return table.sort_values("taux_octroi", ascending=False).round(3)


def grant_rate_by_r_band(history):
    """Committee grant rate per R-score band and group, with the number of applications."""
    band = pd.cut(history[R_SCORE], R_BAND_EDGES, labels=R_BAND_LABELS, right=False)
    table = history.groupby([band, GROUP], observed=True)[LABEL].agg(["mean", "size"]).unstack(GROUP)
    table.columns = [f"{'taux' if stat == 'mean' else 'n'} {group}" for stat, group in table.columns]
    table.index.name = "cote R"
    table["écart"] = table[f"taux {GROUPS[0]}"] - table[f"taux {GROUPS[1]}"]
    return table.round(3)


def raw_gap(history, draws=500, seed=0):
    """The raw gap in grant rate, its bootstrap interval, and what it is worth in grants."""
    rng = np.random.default_rng(seed)
    y, remote = history[LABEL].to_numpy(), (history[GROUP] == REMOTE).to_numpy()
    gap = y[~remote].mean() - y[remote].mean()
    boot = []
    for _ in range(draws):
        i = rng.integers(0, len(y), len(y))
        boot.append(y[i][~remote[i]].mean() - y[i][remote[i]].mean())
    low, high = np.percentile(boot, [2.5, 97.5])
    return pd.DataFrame({"écart de taux d'octroi (Centre - Éloignée)": [gap], "IC 95 % bas": [low], "IC 95 % haut": [high],
                         "bourses en moins pour les régions éloignées": [round(gap * remote.sum())]}).round(3)


# --- Section 2 and 3: the committee as a model -------------------------------------------------

def _terms(df):
    return pd.DataFrame({"r": df[R_SCORE].astype(float), "linc": np.log(df[INCOME]), "hrs": df[HOURS].astype(float),
                         "rem": (df[GROUP] == REMOTE).astype(float)}, index=df.index)


def committee_fit(history, terms=("r", "linc", "hrs", "rem")):
    """Additive logit of the committee's decision on the given terms."""
    return sm.Logit(history[LABEL].astype(float), sm.add_constant(_terms(history)[list(terms)])).fit(disp=0)


def committee_table(history):
    """What the committee weighed: each term's coefficient, and its value in R points."""
    fit = committee_fit(history)
    table = pd.DataFrame({"coefficient": fit.params, "erreur type": fit.bse, "z": fit.tvalues,
                          "en points de cote R": fit.params / fit.params["r"]}).drop("const")
    return table.rename(index=TERM_NAMES).round(3)


def ignored_columns(history):
    """Columns the committee did not use: p-value of each one added to the four-term model."""
    from scipy import stats

    base = committee_fit(history)
    X = sm.add_constant(_terms(history))
    extras = {"Première génération": history[[FIRST_GEN]].astype(float),
              "Programme d'études": pd.get_dummies(history[PROGRAM], drop_first=True).astype(float),
              "Région à l'intérieur du groupe": pd.get_dummies(history[REGION]).astype(float)[
                  ["Capitale-Nationale", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]],
              "Cote R x groupe (pénalité variable)": (_terms(history)["r"] * _terms(history)["rem"]).to_frame("r_rem")}
    rows = {}
    for name, extra in extras.items():
        fit = sm.Logit(history[LABEL].astype(float), pd.concat([X, extra], axis=1)).fit(disp=0)
        stat = 2 * (fit.llf - base.llf)
        rows[name] = {"termes ajoutés": extra.shape[1], "valeur p": stats.chi2.sf(stat, extra.shape[1])}
    return pd.DataFrame(rows).T.round(3)


def penalty_scenarios(history, draws=200, seed=0):
    """Remote grant rate as recorded, then with the penalty removed, then with income ignored too."""
    def scenarios(sample):
        fit, no_income = committee_fit(sample), committee_fit(sample, ("r", "hrs", "rem"))
        remote = sample[sample[GROUP] == REMOTE]
        X = sm.add_constant(_terms(remote).assign(rem=0.0), has_constant="add")
        return np.array([remote[LABEL].mean(), fit.predict(X).mean(),
                         no_income.predict(X[["const", "r", "hrs", "rem"]]).mean(),
                         sample.loc[sample[GROUP] == CENTRE, LABEL].mean(), fit.params["rem"] / fit.params["r"]])
    point = scenarios(history)
    rng = np.random.default_rng(seed)
    boot = [scenarios(history.iloc[rng.integers(0, len(history), len(history))]) for _ in range(draws)]
    low, high = np.percentile(boot, [2.5, 97.5], axis=0) if draws else (np.full(5, np.nan), np.full(5, np.nan))
    n_remote = (history[GROUP] == REMOTE).sum()
    names = ["Régions éloignées, décisions du comité", "… sans la pénalité régionale",
             "… sans la pénalité ni la préférence pour le revenu", "Grands centres, décisions du comité"]
    table = pd.DataFrame({"taux d'octroi": point[:4], "IC 95 % bas": low[:4], "IC 95 % haut": high[:4]}, index=names)
    table["bourses (sur les demandes des régions éloignées)"] = (table["taux d'octroi"] * n_remote).round()
    table.loc[names[3], "bourses (sur les demandes des régions éloignées)"] = np.nan
    penalty = pd.DataFrame({"pénalité en points de cote R": [point[4]], "IC 95 % bas": [low[4]], "IC 95 % haut": [high[4]]})
    return table.round(3), penalty.round(2)


def gap_decomposition(history):
    """The raw gap split into the penalty, the income preference and the rest."""
    table, _ = penalty_scenarios(history, draws=0)
    rate = table["taux d'octroi"]
    recorded, no_penalty, no_income, centre = rate.iloc[0], rate.iloc[1], rate.iloc[2], rate.iloc[3]
    parts = pd.Series({"Pénalité régionale": no_penalty - recorded, "Préférence pour le revenu élevé": no_income - no_penalty,
                       "Reste (cote R, heures travaillées)": centre - no_income, "Écart total": centre - recorded})
    return pd.DataFrame({"points de taux d'octroi": 100 * parts, "part de l'écart": parts / parts["Écart total"]}).round(3)


def fig_committee_weights(history, name="poids_comite.png"):
    """What the committee weighed, one standard deviation of each column expressed in R points."""
    fit, terms = committee_fit(history), _terms(history)
    sd = terms[["r", "linc", "hrs"]].std()
    effects = (fit.params[sd.index] * sd / fit.params["r"]).rename(
        {"r": "Cote R (+1 écart-type)", "linc": "Revenu familial (+1 écart-type)", "hrs": "Heures de travail (+1 écart-type)"})
    effects["Région éloignée"] = fit.params["rem"] / fit.params["r"]
    effects = effects.sort_values()
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bars = ax.barh(effects.index, effects.values, height=0.6,
                   color=[plots.GROUP_COLORS[REMOTE] if v < 0 else plots.ACCENT for v in effects.values])
    ax.bar_label(bars, labels=[f"{v:+.2f}".replace(".", ",") for v in effects.values], padding=4, color=plots.INK)
    ax.axvline(0, color=plots.MUTED, linewidth=1)
    ax.set_xlim(effects.min() - 0.6, effects.max() + 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effet sur la décision, en points de cote R")
    ax.set_title("Ce que le comité a pesé")
    plots.save(fig, name)
    plt.close(fig)
    return fig


# --- Section 4: proxies -------------------------------------------------------------------------

def _design(history, columns):
    parts = []
    for column in columns:
        if column in (PROGRAM, POSTAL):
            parts.append(pd.get_dummies(history[column], prefix=column).astype(float))
        elif column in (INCOME, DISTANCE):
            parts.append(np.log(history[[column]]))
        else:
            parts.append(history[[column]].astype(float))
    X = pd.concat(parts, axis=1)
    return (X - X.mean()) / X.std().replace(0, 1)


def _group_auc(history, columns):
    remote = (history[GROUP] == REMOTE).astype(int)
    p = cross_val_predict(LogisticRegression(max_iter=2000), _design(history, columns), remote, cv=5,
                          method="predict_proba")[:, 1]
    return roc_auc_score(remote, p)


def proxy_table(history):
    """How well each column, and sets of columns, give the group away (cross-validated AUC)."""
    rows = {COLUMN_NAMES[c]: _group_auc(history, [c]) for c in COLUMN_NAMES}
    numeric = [R_SCORE, INCOME, HOURS, DISTANCE, FIRST_GEN]
    rows["Toutes les colonnes sauf la région"] = _group_auc(history, list(COLUMN_NAMES))
    rows["Sans région, code postal ni distance"] = _group_auc(history, [c for c in numeric if c != DISTANCE] + [PROGRAM])
    table = pd.DataFrame({"AUC pour deviner le groupe": rows})
    return table.sort_values("AUC pour deviner le groupe", ascending=False).round(3)


def fig_proxies(history, name="proxys.png"):
    table = proxy_table(history)["AUC pour deviner le groupe"]
    table = table[[c for c in table.index if c in COLUMN_NAMES.values()]].sort_values()
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.barh(table.index, table.values - 0.5, left=0.5, height=0.6, color=plots.ACCENT)
    ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in table.values], padding=4, color=plots.INK)
    ax.set_xlim(0.5, 1.06)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Capacité à deviner le groupe (AUC : 0,5 = aucune, 1 = parfaite)")
    ax.set_title("Chaque colonne trahit la région, certaines presque parfaitement")
    plots.save(fig, name)
    plt.close(fig)
    return fig


def column_removal_table(history):
    """The production model retrained with fewer and fewer columns: gap and accuracy on held-out history."""
    train, test = split_history(history)
    sets = {"all": "Toutes les colonnes (modèle en production)", "no_region": "Sans la région",
            "no_region_postal": "Sans la région ni le code postal",
            "merit_income_hours": "Cote R, revenu et heures seulement", "merit_hours": "Cote R et heures seulement"}
    rows = {}
    for features, name in sets.items():
        decisions = decide({**complete(BASELINE_SPEC), "features": features}, train, test)
        rows[name] = {"écart de parité": evaluate.parity_gap(decisions, test[GROUP]),
                      "accord avec le comité": evaluate.agreement(decisions, test[LABEL])}
    return pd.DataFrame(rows).T.round(3)


def hours_weight_table(history):
    """The hours weight (R points per hour) when the group is fitted, and when it is left out."""
    specs = {"Comité ajusté avec le groupe": ("r", "linc", "hrs", "rem"),
             "Groupe retiré du modèle": ("r", "linc", "hrs"), "Groupe et revenu retirés": ("r", "hrs")}
    rows = {}
    for name, terms in specs.items():
        fit = committee_fit(history, terms)
        ratio = fit.params["hrs"] / fit.params["r"]
        v = fit.cov_params()
        se = abs(ratio) * np.sqrt(v.loc["hrs", "hrs"] / fit.params["hrs"] ** 2 + v.loc["r", "r"] / fit.params["r"] ** 2
                                  - 2 * v.loc["hrs", "r"] / (fit.params["hrs"] * fit.params["r"]))
        rows[name] = {"poids d'une heure, en points de cote R": ratio, "erreur type": se}
    return pd.DataFrame(rows).T.round(3)


def hours_weight_interval():
    """95 % interval of the hours weight of the committee fitted with the group, from the history."""
    from .data import load_history

    row = hours_weight_table(load_history()).iloc[0]
    return row.iloc[0] - 1.96 * row.iloc[1], row.iloc[0] + 1.96 * row.iloc[1]


# --- Section 5: the production model ------------------------------------------------------------

def baseline_decisions(history, candidates):
    """(held-out test table, baseline decisions on it, baseline decisions on the candidates)."""
    spec = complete(BASELINE_SPEC)
    train, test = split_history(history)
    return test, np.asarray(decide(spec, train, test)), np.asarray(decide(spec, history, candidates))


def production_audit(test, decisions):
    """Selection, true-positive and false-positive rates of the production model against the committee, by group."""
    rows = {}
    for group in GROUPS:
        member = (test[GROUP] == group).to_numpy()
        y, d = test[LABEL].to_numpy()[member], decisions[member]
        rows[group] = {"taux d'octroi du comité": y.mean(), "taux d'octroi du modèle": d.mean(),
                       "accord avec le comité": (y == d).mean(), "vrais positifs (TPR)": d[y == 1].mean(),
                       "faux positifs (FPR)": d[y == 0].mean()}
    table = pd.DataFrame(rows)
    table["écart (Centre - Éloignée)"] = table[CENTRE] - table[REMOTE]
    return table.round(3)


def fig_baseline_by_region(candidates, decisions, name="production_par_region.png"):
    """Grant rate of the production model on the 4,000, per region."""
    by_region = evaluate.selection_rates(decisions, candidates[REGION]).sort_values()
    group_of = candidates.groupby(REGION)[GROUP].first()
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.barh(by_region.index, by_region.values, height=0.6,
                   color=[plots.GROUP_COLORS[group_of[r]] for r in by_region.index])
    ax.bar_label(bars, labels=[f"{v:.1%}".replace(".", ",").replace("%", " %") for v in by_region.values],
                 padding=4, color=plots.INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, by_region.max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Taux d'octroi du modèle en production, 4 000 candidats")
    ax.set_title("Le modèle reproduit l'écart du comité sur les nouveaux candidats")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plots.GROUP_COLORS[g]) for g in GROUPS]
    ax.legend(handles, [plots.GROUP_NAMES[g] for g in GROUPS], loc="lower right")
    plots.save(fig, name)
    plt.close(fig)
    return fig


# --- Section 6: which standard, which metric ----------------------------------------------------

def standards_table(candidates, baseline, retained):
    """Per standard of merit: who it calls deserving, and the equal-opportunity gap of both models."""
    names = references.PLAUSIBLE
    rows = {}
    for name in names:
        label = references.label(name, candidates)
        share = evaluate.selection_rates(label, candidates[GROUP])
        rows[references.LABELS[name]] = {
            "méritants, grands centres": share[CENTRE], "méritants, régions éloignées": share[REMOTE],
            "écart d'égalité des chances, production": evaluate.equal_opportunity_gap(baseline, label, candidates[GROUP]),
            "écart d'égalité des chances, modèle corrigé": evaluate.equal_opportunity_gap(retained, label, candidates[GROUP])}
    return pd.DataFrame(rows).T.round(3)


def committee_label_gap(test, decisions):
    """The equal-opportunity gap of the production model when the committee's own label is the yardstick."""
    rates = evaluate.true_positive_rates(decisions, test[LABEL], test[GROUP])
    return pd.DataFrame({"TPR grands centres": [rates[CENTRE]], "TPR régions éloignées": [rates[REMOTE]],
                         "écart mesuré contre le comité": [rates[CENTRE] - rates[REMOTE]],
                         "écart publié contre l'étalon": [evaluate.PUBLISHED_BASELINE_GAP]}).round(3)


def retained_decisions(candidates):
    """Decisions of the retained model, read from predictions.csv (written by model_corrige.py)."""
    saved = pd.read_csv(ROOT / "predictions.csv").set_index(ID)
    return saved.loc[candidates[ID], LABEL].to_numpy()


# --- Section 7: subgroups -----------------------------------------------------------------------

def subgroup_table(candidates, baseline, retained):
    """Grant rate before (production) and after (corrected), per subgroup crossed with the group."""
    income = pd.qcut(candidates[INCOME], 3, labels=["revenu bas", "revenu moyen", "revenu élevé"])
    hours = pd.qcut(candidates[HOURS], 3, labels=["peu d'heures", "heures moyennes", "beaucoup d'heures"])
    keys = {"Première génération": candidates[FIRST_GEN].map({1: "oui", 0: "non"}), "Revenu familial": income,
            "Heures de travail": hours, "Programme": candidates[PROGRAM], "Région": candidates[REGION]}
    frame = pd.DataFrame({"groupe": candidates[GROUP].to_numpy(), "production": baseline, "corrigé": retained})
    parts = []
    for name, key in keys.items():
        part = frame.assign(valeur=np.asarray(key)).groupby(["valeur", "groupe"], observed=True).agg(
            candidats=("production", "size"), production=("production", "mean"), corrigé=("corrigé", "mean"))
        part["variation"] = part["corrigé"] - part["production"]
        parts.append(pd.concat({name: part}, names=["sous-groupe"]))
    return pd.concat(parts).round(3)


def fig_before_after(candidates, baseline, retained, name="avant_apres.png"):
    """Grant rate per region, production model against corrected model."""
    before = evaluate.selection_rates(baseline, candidates[REGION])
    after = evaluate.selection_rates(retained, candidates[REGION])
    order = before.sort_values().index
    group_of = candidates.groupby(REGION)[GROUP].first()
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(order))
    ax.hlines(y, before[order], after[order], color=plots.GRID, linewidth=3, zorder=1)
    ax.scatter(before[order], y, s=70, marker="s", color=plots.BASELINE_COLOR, label="Modèle en production", zorder=2)
    ax.scatter(after[order], y, s=80, color=[plots.GROUP_COLORS[group_of[r]] for r in order],
               edgecolor=plots.SURFACE, linewidth=1.5, zorder=3)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=plots.BASELINE_COLOR, markersize=8)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=plots.GROUP_COLORS[g], markersize=8) for g in GROUPS]
    ax.legend(handles, ["Modèle en production"] + [f"Modèle corrigé, {plots.GROUP_NAMES[g].lower()}" for g in GROUPS],
              loc="lower right")
    ax.axvline(0.40, color=plots.MUTED, linewidth=1, linestyle=":")
    ax.set_yticks(y, order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Taux d'octroi sur les 4 000 candidats")
    ax.set_title("Après correction, les cinq régions se rejoignent")
    plots.save(fig, name)
    plt.close(fig)
    return fig


# --- Section 0: the summary ---------------------------------------------------------------------

def summary(history, candidates, baseline, retained):
    """The finding in a few numbers, each recomputed from the data."""
    scenarios, penalty = penalty_scenarios(history, draws=0)
    rate = history.groupby(GROUP)[LABEL].mean()
    rows = {
        "Taux d'octroi du comité, grands centres": rate[CENTRE],
        "Taux d'octroi du comité, régions éloignées": rate[REMOTE],
        "Pénalité régionale, en points de cote R": penalty["pénalité en points de cote R"].iloc[0],
        "Taux des régions éloignées sans la pénalité": scenarios["taux d'octroi"].iloc[1],
        "Écart de parité du modèle en production (4 000 candidats)": evaluate.parity_gap(baseline, candidates[GROUP]),
        "Écart de parité du modèle corrigé (4 000 candidats)": evaluate.parity_gap(retained, candidates[GROUP]),
        "Taux d'octroi du modèle corrigé (budget : 36 % à 44 %)": float(np.mean(retained)),
    }
    return pd.DataFrame({"valeur": rows}).round(3)


# --- Earlier figures ------------------------------------------------------------------------------

def fig_grant_rate_by_r_band(history, name="taux_par_cote_r.png"):
    """At equal R score, the committee still grants less to the remote regions."""
    table = grant_rate_by_r_band(history)
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for group in GROUPS:
        rates = table[f"taux {group}"]
        ax.plot(table.index.astype(str), rates, "-o", color=plots.GROUP_COLORS[group],
                label=plots.GROUP_NAMES[group], markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Cote R")
    ax.set_ylabel("Taux d'octroi du comité")
    ax.set_title("À cote R égale, les régions éloignées reçoivent moins")
    ax.legend(loc="upper left")
    plots.save(fig, name)
    plt.close(fig)
    return fig


def fig_grant_rate_by_region(history, name="taux_par_region.png"):
    """Committee grant rate per region, coloured by group."""
    table = grant_rate_table(history, by=[REGION, GROUP]).reset_index().sort_values("taux_octroi")
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bars = ax.barh(table[REGION], table["taux_octroi"], height=0.6,
                   color=[plots.GROUP_COLORS[g] for g in table[GROUP]])
    ax.bar_label(bars, labels=[f"{v:.1%}".replace(".", ",").replace("%", " %") for v in table["taux_octroi"]],
                 padding=4, color=plots.INK)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, table["taux_octroi"].max() * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Taux d'octroi du comité")
    ax.set_title("Taux d'octroi par région")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plots.GROUP_COLORS[g]) for g in GROUPS]
    ax.legend(handles, [plots.GROUP_NAMES[g] for g in GROUPS], loc="lower right")
    plots.save(fig, name)
    plt.close(fig)
    return fig
