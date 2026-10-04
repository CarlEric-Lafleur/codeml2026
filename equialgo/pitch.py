"""Figures and tables that tell how the corrected model was found. Used by the notebook and the slides.

Output labels are in French: they appear as is in the notebook and on the slides.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from . import audit, plots, references, scorers
from .data import CENTRE, REMOTE, ROOT

LOG_PATH = ROOT / "submissions" / "log.csv"
RETAINED_FILE = "02_r_plus_hours.csv"

RULES, NEED, TRAINED, FAIRLEARN = "Règles simples", "Règles avec besoin", "Modèles entraînés", "fairlearn (parité)"
FAMILY_COLORS = {RULES: plots.ACCENT, NEED: plots.MUTED, TRAINED: "#1baf7a", FAIRLEARN: plots.GROUP_COLORS[REMOTE]}
# What each scored file is, in the words of the slides, and its family.
FILES = {
    "01_merit_only.csv": ("Cote R seule", RULES),
    "02_r_plus_hours.csv": ("Cote R + heures travaillées (retenu)", RULES),
    "03_r_within_group.csv": ("Cote R classée dans chaque groupe", RULES),
    "04_committee_no_penalty.csv": ("Comité sans la pénalité, revenu gardé", RULES),
    "05_r_first_gen_3pts.csv": ("Cote R + bonus première génération", NEED),
    "06_r_low_income.csv": ("Cote R, faibles revenus favorisés", NEED),
    "07_weighted_logit.csv": ("Logit, décisions biaisées sous-pondérées", TRAINED),
    "08_relabelled_logit.csv": ("Logit, décisions biaisées réétiquetées", TRAINED),
    "09_dropped_forest.csv": ("Forêt, décisions biaisées retirées", TRAINED),
    "10_dropped_band_boosting.csv": ("Boosting, décisions biaisées retirées", TRAINED),
    "11_corrected_region_quotas.csv": ("Comité corrigé, quotas par région", TRAINED),
    "12_exp_gradient_parity.csv": ("ExponentiatedGradient", FAIRLEARN),
    "13_threshold_optimizer_parity.csv": ("ThresholdOptimizer", FAIRLEARN),
    "14_r_plus_018_hours.csv": ("Cote R + heures, poids plus fort", RULES),
    "15_r_plus_remote_bonus.csv": ("Cote R + bonus aux régions éloignées", RULES),
    "16_r_plus_0059_hours.csv": ("Cote R + heures, poids appris sans le groupe", RULES),
}
# Files that are R + weight x hours: the weight of each, in R points per hour.
HOURS_WEIGHTS = {"01_merit_only.csv": 0.0, "16_r_plus_0059_hours.csv": scorers.BLIND_HOURS,
                 "02_r_plus_hours.csv": references.COMMITTEE_HOURS, "14_r_plus_018_hours.csv": 0.18}
SAMPLING_NOISE = 1.0  # points of accuracy: the feedback is on a subset of unknown size


def _fr(value, digits=2, sign=False):
    return f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",").replace("-", "−")


def _comma(digits):
    """Axis formatter with a decimal comma."""
    return FuncFormatter(lambda value, _: _fr(value, digits))


# --- The diagnosis ------------------------------------------------------------------------------

def fig_decomposition(history, name="decomposition.png"):
    """The raw gap in grant rate, from the remote rate up to the centres' rate, cause by cause."""
    parts = audit.gap_decomposition(history)["points de taux d'octroi"]
    rate = audit.grant_rate_table(history)["taux_octroi"] * 100
    steps = [("Pénalité\nrégionale", parts["Pénalité régionale"], plots.GROUP_COLORS[REMOTE]),
             ("Préférence\npour le revenu", parts["Préférence pour le revenu élevé"], "#eda100"),
             ("Cote R\net heures", parts["Reste (cote R, heures travaillées)"], plots.MUTED)]
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    labels = ["Régions\néloignées"] + [s[0] for s in steps] + ["Grands\ncentres"]
    ax.bar(0, rate[REMOTE], width=0.62, color=plots.GROUP_COLORS[REMOTE])
    ax.annotate(f"{_fr(rate[REMOTE], 1)} %", (0, rate[REMOTE]), xytext=(0, 5), textcoords="offset points", ha="center")
    level = rate[REMOTE]
    for i, (_, value, color) in enumerate(steps, start=1):
        ax.bar(i, value, bottom=level, width=0.62, color=color)
        ax.plot([i - 0.69, i - 0.31], [level, level], color=plots.MUTED, linewidth=1)
        level += value
        ax.annotate(f"+{_fr(value, 1)} pts", (i, level), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontweight="bold")
    ax.plot([len(steps) + 0.31, len(steps) + 0.69], [level, level], color=plots.MUTED, linewidth=1)
    ax.bar(len(steps) + 1, rate[CENTRE], width=0.62, color=plots.GROUP_COLORS[CENTRE])
    ax.annotate(f"{_fr(rate[CENTRE], 1)} %", (len(steps) + 1, rate[CENTRE]), xytext=(0, 5), textcoords="offset points",
                ha="center")
    ax.set_xticks(range(len(labels)), labels)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.set_ylim(0, rate[CENTRE] * 1.16)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Taux d'octroi du comité")
    ax.set_title("D'où viennent les 21 points d'écart".replace("21", f"{parts['Écart total']:.0f}"))
    plots.save(fig, name)
    plt.close(fig)
    return fig


def fig_without_region(history, name="sans_region.png"):
    """Hiding columns barely moves the gap, and why: the penalty moves into the hours weight."""
    removal = audit.column_removal_table(history)["écart de parité"]
    weights = audit.hours_weight_table(history)["poids d'une heure, en points de cote R"].iloc[:2]
    plots.apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    names = [n.replace(" (modèle en production)", "") for n in removal.index][::-1]
    bars = left.barh(names, removal.values[::-1], height=0.6, color=plots.BASELINE_COLOR)
    left.bar_label(bars, labels=[_fr(v, 3) for v in removal.values[::-1]], padding=4, color=plots.INK)
    left.set_xlim(0, removal.max() * 1.18)
    left.xaxis.set_major_locator(MaxNLocator(5))
    left.xaxis.set_major_formatter(_comma(2))
    left.grid(axis="y", visible=False)
    left.set_xlabel("Écart de taux d'octroi du modèle réentraîné")
    left.set_title("Cacher des colonnes : l'écart reste")
    bars = right.bar(["Appris avec\nla région", "Appris sans\nla région"], weights.values, width=0.55,
                     color=[plots.ACCENT, plots.GROUP_COLORS[REMOTE]])
    right.bar_label(bars, labels=[_fr(v, 3) for v in weights.values], padding=4, color=plots.INK)
    right.set_ylim(0, weights.max() * 1.2)
    right.yaxis.set_major_formatter(_comma(2))
    right.grid(axis="x", visible=False)
    right.set_ylabel("Poids d'une heure travaillée\n(points de cote R)")
    right.set_title("La pénalité se cache dans les heures")
    fig.tight_layout(w_pad=3)
    plots.save(fig, name)
    plt.close(fig)
    return fig


# --- The method ---------------------------------------------------------------------------------

def fig_method(history, name="methode.png"):
    """The correction in three boxes, with the weights learned from the history (in R points)."""
    fit = audit.committee_fit(history)
    weight = fit.params / fit.params["r"]
    terms = [("Cote R", "1 point", True), ("Heures travaillées", f"{_fr(weight['hrs'], 3, sign=True)} / h", True),
             ("Revenu familial (log)", _fr(weight["linc"], sign=True), False),
             ("Région éloignée", _fr(weight["rem"], sign=True), False)]
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    ax.set_xlim(0, 3.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    titles = ["1. Apprendre le comité", "2. Retirer les deux biais", "3. Décider"]
    subtitles = ["sur ses décisions, région comprise", "la pénalité régionale et le revenu",
                 "la région n'est pas lue"]
    for i, (title, subtitle) in enumerate(zip(titles, subtitles)):
        x = 0.02 + i * 1.12
        ax.add_patch(FancyBboxPatch((x, 0.04), 1.0, 0.92, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor="white", edgecolor=plots.GRID, linewidth=1.5))
        ax.text(x + 0.05, 0.86, title, fontsize=13, fontweight="bold", va="center")
        ax.text(x + 0.05, 0.76, subtitle, fontsize=9.5, color=plots.MUTED, va="center")
        if i < 2:
            ax.annotate("", (x + 1.11, 0.5), (x + 1.01, 0.5), arrowprops={"arrowstyle": "-|>", "color": plots.MUTED, "lw": 1.6})
    for column in (0, 1):
        x = 0.07 + column * 1.12
        for row, (label, value, kept) in enumerate(terms):
            y = 0.60 - row * 0.14
            removed = column == 1 and not kept
            color = plots.GRID if removed else plots.GROUP_COLORS[REMOTE] if not kept else plots.INK
            ax.text(x, y, label, fontsize=11, va="center", color="#b5b4b0" if removed else plots.INK)
            ax.text(x + 0.90, y, value, fontsize=11, va="center", ha="right", fontweight="bold",
                    color="#b5b4b0" if removed else color)
            if removed:
                ax.plot([x, x + 0.90], [y, y], color=plots.GROUP_COLORS[REMOTE], linewidth=1.3)
    x = 0.07 + 2 * 1.12
    ax.text(x, 0.58, "Score =", fontsize=11, va="center", color=plots.MUTED)
    ax.text(x, 0.45, f"cote R + {_fr(weight['hrs'], 3)} × heures", fontsize=12.5, va="center", fontweight="bold",
            color=plots.ACCENT)
    ax.text(x, 0.28, "Bourse aux 40 % les mieux classés,", fontsize=10.5, va="center")
    ax.text(x, 0.18, "un seul seuil pour tout le monde.", fontsize=10.5, va="center")
    plots.save(fig, name)
    plt.close(fig)
    return fig


# --- The search ---------------------------------------------------------------------------------

def search_table():
    """Every file the scorer has marked: what it is, its family, what came back, its parity gap."""
    log = pd.read_csv(LOG_PATH).dropna(subset=["returned_accuracy"])
    log = log[log["file"].isin(FILES)]
    table = pd.DataFrame({
        "fichier": log["file"].str[:2].to_numpy(),
        "modèle": [FILES[f][0] for f in log["file"]],
        "famille": [FILES[f][1] for f in log["file"]],
        "accord avec l'étalon (%)": log["returned_accuracy"].to_numpy(),
        "F1 macro (%)": log["returned_f1_macro"].to_numpy(),
        "écart de parité": log["parity_gap"].round(3).to_numpy(),
        "la décision lit la région": log["uses_region"].to_numpy(),
    }, index=log["file"].to_numpy())
    return table.sort_values("accord avec l'étalon (%)", ascending=False)


def fig_search(name="recherche.png"):
    """What the scorer returned for each family of models, and for each hours weight."""
    table = search_table().iloc[::-1]
    best = table["accord avec l'étalon (%)"].max()
    plots.apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.45, 1]})
    y = range(len(table))
    left.axvspan(best - SAMPLING_NOISE, best, color=plots.GRID, alpha=0.7, zorder=0)
    left.annotate("à moins d'un point du meilleur :\nindiscernables (sous-ensemble)", (best - SAMPLING_NOISE - 0.15, 0.6), ha="right",
                  fontsize=8.5, color=plots.MUTED)
    left.scatter(table["accord avec l'étalon (%)"], y, s=75, zorder=3, edgecolor=plots.SURFACE, linewidth=1.2,
                 color=[FAMILY_COLORS[f] for f in table["famille"]])
    retained = list(table.index).index(RETAINED_FILE)
    left.scatter([best], [retained], s=260, facecolors="none", edgecolors=plots.INK, linewidths=1.8, zorder=4)
    left.set_yticks(y, table["modèle"])
    left.get_yticklabels()[retained].set_fontweight("bold")
    left.get_yticklabels()[retained].set_color(plots.INK)
    left.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.grid(axis="y", visible=False)
    left.set_xlabel("Accord avec l'étalon caché (retour du correcteur)")
    left.set_title(f"{len(table)} modèles soumis, quatre familles")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=8) for c in FAMILY_COLORS.values()]
    left.legend(handles, FAMILY_COLORS, loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=9.5)

    accuracy = table["accord avec l'étalon (%)"]
    weights = pd.Series(HOURS_WEIGHTS).sort_values()
    right.plot(weights.values, accuracy[weights.index].values, "-o", color=plots.ACCENT,
               markeredgecolor=plots.SURFACE, markeredgewidth=1.5)
    low, high = audit.hours_weight_interval()
    right.axvspan(low, high, color=plots.ACCENT, alpha=0.15)
    right.annotate("poids appris\ndu comité,\nrégion comprise", ((low + high) / 2, accuracy.loc[list(weights.index)].min()),
                   ha="center", va="bottom", fontsize=9, color=plots.ACCENT)
    right.scatter([references.COMMITTEE_HOURS], [best], s=260, facecolors="none", edgecolors=plots.INK,
                  linewidths=1.8, zorder=4)
    right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{_fr(value, 1)} %"))
    right.xaxis.set_major_formatter(_comma(2))
    right.set_xlabel("Poids d'une heure travaillée (points de cote R)")
    right.set_ylabel("Accord avec l'étalon caché")
    right.set_title("Le poids vient de l'historique, pas du correcteur")
    right.margins(x=0.12, y=0.2)
    fig.tight_layout(w_pad=2.5)
    plots.save(fig, name)
    plt.close(fig)
    return fig
