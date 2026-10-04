"""Each column: distribution, range, rounding, clipping, outliers; same in both files?"""
import matplotlib.pyplot as plt
from scipy import stats

from common import NUMERIC, load, pd, plots, save, section, share_hists

history, candidates = load()
files = {"history": history, "candidates": candidates}

section("Shape, missing values, ids")
for name, df in files.items():
    print(f"{name}: {len(df)} rows, {int(df.isna().sum().sum())} missing, ids unique: {df['id_candidat'].is_unique}")
ids = pd.concat([history["id_candidat"], candidates["id_candidat"]]).str[1:].astype(int)
print(f"ids shared by both files: {len(set(history['id_candidat']) & set(candidates['id_candidat']))}; "
      f"together they cover {ids.min()}..{ids.max()} with {ids.nunique()} distinct values")
print("history rows duplicated on every column but the id:",
      int(history.drop(columns="id_candidat").duplicated().sum()))

section("Numeric columns, per file")
print(pd.concat({n: df[NUMERIC].describe().T for n, df in files.items()}).round(2).to_string())

section("Rounding, clipping, tails (both files pooled)")
both = pd.concat(files.values())
rows = []
for col in NUMERIC[:-1]:
    x = both[col]
    decimals = max(len(f"{v:g}".partition(".")[2]) for v in x.unique())
    z = (x - x.mean()) / x.std()
    rows.append({"column": col, "decimals": decimals, "min": x.min(), "n at min": int((x == x.min()).sum()),
                 "max": x.max(), "n at max": int((x == x.max()).sum()), "n beyond 4 sd": int((z.abs() > 4).sum()),
                 "most frequent value share": round(x.value_counts(normalize=True).iloc[0], 4)})
print(pd.DataFrame(rows).set_index("column").to_string())
print("R score outside the documented 15-40:", int(((both["r"] < 15) | (both["r"] > 40)).sum()),
      "| R >= 39:", sorted(both.loc[both["r"] >= 39, "r"]))

section("Same distribution in both files? Kolmogorov-Smirnov p-values")
print(pd.Series({col: stats.ks_2samp(history[col], candidates[col]).pvalue for col in NUMERIC}).round(3).to_string())

section("Categorical columns: share per file, chi-square p-value")
for col in ["prog", "reg", "cp", "fg"]:
    counts = pd.concat({n: df[col].value_counts() for n, df in files.items()}, axis=1).fillna(0)
    p = stats.chi2_contingency(counts)[1]
    print(f"\n{col}: {len(counts)} values, chi-square p = {p:.3f}")
    print((counts / counts.sum()).round(3).to_string())

section("Committee grant rate on the history")
print(f"{history['y'].mean():.4f} ({int(history['y'].sum())} of {len(history)})")

# Figure: the two files side by side, column by column.
plots.apply_style()
samples = {"10 000 demandes historiques": history, "4 000 candidats à évaluer": candidates}
colors = dict(zip(samples, [plots.MUTED, plots.ACCENT]))
fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
share_hists(axes, samples, colors, histtype="step", linewidth=1.8)
axes[0, 0].legend(loc="upper left", fontsize=9)
fig.suptitle("Les deux fichiers suivent les mêmes distributions", x=0.02, ha="left", fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "a01_fichiers.png")
