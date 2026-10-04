"""Columns and region: what differs between groups and regions, and how well each column gives the group away."""
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import GROUP, GROUPS, MODEL_NUMERIC, NUMERIC, load, pd, plots, save, section

history, _ = load()
LABELS = {"r": "Cote R", "linc": "Revenu familial", "hrs": "Heures de travail", "ldist": "Distance au campus",
          "fg": "Première génération", "prog": "Programme", "cp": "Code postal (3 car.)"}

section("Mean per group, and the difference in pooled standard deviations")
by_group = history.groupby(GROUP)[NUMERIC + ["linc", "ldist"]].mean().T
by_group["diff in sd"] = (by_group[GROUPS[1]] - by_group[GROUPS[0]]) / history[by_group.index].std()
print(by_group.round(3).to_string())

section("Mean per region")
print(history.groupby([GROUP, "reg"])[NUMERIC + ["y"]].mean().round(2).to_string())

section("Do regions differ inside a group? One-way ANOVA p-values across the regions of each group")
rows = {}
for group, part in history.groupby(GROUP):
    samples = [g for _, g in part.groupby("reg")]
    rows[group] = {col: stats.f_oneway(*[s[col] for s in samples]).pvalue for col in MODEL_NUMERIC + ["y"]}
print(pd.DataFrame(rows).round(3).to_string())

section("Does the postal prefix determine the region?")
per_prefix = history.groupby("cp")["reg"].nunique()
print(f"{len(per_prefix)} prefixes, regions per prefix: max {per_prefix.max()} -> the prefix determines the region")
print(history.groupby("reg")["cp"].agg(lambda s: " ".join(sorted(s.unique()))).to_string())

section("Do prefixes differ inside a region? ANOVA p-values across the prefixes of each region")
rows = {}
for region, part in history.groupby("reg"):
    samples = [g for _, g in part.groupby("cp")]
    rows[region] = {col: stats.f_oneway(*[s[col] for s in samples]).pvalue for col in MODEL_NUMERIC + ["y"]}
print(pd.DataFrame(rows).round(3).to_string())

section("How well does each column predict the group? AUC (0.5 = not at all, 1 = perfectly)")
remote = history["rem"]
auc = {}
for col in MODEL_NUMERIC:
    a = roc_auc_score(remote, history[col])
    auc[col] = max(a, 1 - a)
for col in ["prog", "cp"]:  # categorical: rank the categories by their share of remote applicants
    auc[col] = roc_auc_score(remote, history.groupby(col)["rem"].transform("mean"))
auc = pd.Series(auc).sort_values()
print(auc.round(3).to_string())

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
for name, cols in {"all numeric columns (no region, no postal code)": MODEL_NUMERIC,
                   "same without distance": [c for c in MODEL_NUMERIC if c != "ldist"]}.items():
    p = cross_val_predict(model, history[cols], remote, cv=5, method="predict_proba")[:, 1]
    print(f"{name}: cross-validated AUC {roc_auc_score(remote, p):.3f}, accuracy {((p > 0.5) == remote).mean():.3f}")

section("Overlap of the distance distributions")
print(history.groupby(GROUP)["dist"].quantile([0, 0.01, 0.5, 0.99, 1]).unstack().round(1).to_string())
cut = 60
print(f"rule 'distance > {cut} km' classifies the group with accuracy {((history['dist'] > cut) == remote).mean():.3f}")

plots.apply_style()
fig, ax = plt.subplots(figsize=(7.5, 4))
bars = ax.barh([LABELS[c] for c in auc.index], auc.values - 0.5, left=0.5, height=0.6, color=plots.ACCENT)
ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in auc.values], padding=4, color=plots.INK)
ax.set_xlim(0.5, 1.06)
ax.grid(axis="y", visible=False)
ax.set_xlabel("AUC pour deviner le groupe (0,5 = hasard, 1 = certitude)")
ax.set_title("Chaque colonne trahit la région, à des degrés divers")
save(fig, "a02_proxys.png")
