#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``Untitled.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

# paired_violin_sigma_QTN_GAF_MTF.py
import os, re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- input files ----------
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ABIDE.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ABIDE.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ABIDE.csv",
#}
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ADHD.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ADHD.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ADHD.csv",
#}
#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_SCZ-SCZ.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_SCZ-SCZ.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_SCZ-SCZ.csv",
#}

#FILES = {
 #   "QTN": "CSV-Small-world/metrics_QTN_ABIDE-ASD.csv",
  #  "GAF": "CSV-Small-world/metrics_GAF_ABIDE-ASD.csv",
   # "MTF": "CSV-Small-world/metrics_MTF_ABIDE-ASD.csv",
#}
FILES = {
    "QTN": "CSV-Small-world/metrics_QTN_MEArecs.csv",
    "GAF": "CSV-Small-world/metrics_GAF_MEArecs.csv",
    "MTF": "CSV-Small-world/metrics_MTF_MEArecs.csv",
}

DATASET_ORDER = ["QTN", "GAF", "MTF"]

# Your palette for the violins
PALETTE_DATASET = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# Which column holds sigma? (auto-detect among these)
SIGMA_ALIASES = ["small_world_sigma", "sigma_small_world", "sigma", "sigma_sw"]

# Output
#OUT_DIR         = "ABIDE-RESULTS/paired_sigma_plots-ABIDE-TD"
#OUT_DIR         = "ADHD-RESULTS-Control/paired_sigma_plots-ADHD-TD"
#OUT_DIR         = "SCZ-RESULTS-Control/paired_sigma_plots-SCZ-TD"
OUT_DIR         = "OTHER-RESULTS//MEA-DATA/"
OUT_PLOT_PNG    = "sigma_violin_box_points_lines.png"
OUT_PLOT_PDF    = "sigma_violin_box_points_lines.pdf"
OUT_WIDE_CSV    = "sigma_wide_matched.csv"
OUT_LONG_CSV    = "sigma_long_matched.csv"

# Plot tuning
SIGMA_THRESH   = 1.0
POINT_SIZE     = 34
LINE_WIDTH     = 1.2
JITTER_SD      = 0.05
RNG_SEED       = 2025

# Style
sns.set_theme(context="notebook", style="white")
plt.rcParams.update({"axes.grid": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white"})

# ------------- helpers -------------
def _norm(s: str) -> str:
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")

def _find_sigma_col(df: pd.DataFrame) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for a in SIGMA_ALIASES:
        if _norm(a) in norm_map: return norm_map[_norm(a)]
    return None

def _normalize_key(s: str) -> str:
    s = str(s).strip().lower()
    m = re.search(r'(?:^|[^a-z0-9])(patient|td|hc|sz)[\s_-]*([0-9]+)\b', s)
    if m: return f"num_{int(m.group(2))}"
    m = re.search(r'([0-9]+)', s)
    if m: return f"num_{int(m.group(1))}"
    return re.sub(r'[^a-z0-9]+', '_', s)

def _load_sigma(path: str, dataset: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] missing {dataset}: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "patient_id" in df.columns:
        ids = df["patient_id"].astype(str)
    else:
        ids = pd.Series([f"{dataset}_{i}" for i in range(len(df))], name="patient_id")

    c_sigma = _find_sigma_col(df)
    if c_sigma is None:
        print(f"[WARN] {dataset}: no sigma column in {path}. Columns: {list(df.columns)[:10]} …")
        return pd.DataFrame()

    out = pd.DataFrame({
        "patient_id": ids,
        "key": [ _normalize_key(x) for x in ids ],
        dataset: pd.to_numeric(df[c_sigma], errors="coerce")
    })
    # average duplicates by key
    out = out.groupby(["key"], as_index=False)[dataset].mean(numeric_only=True)
    return out

# ------------- main -------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load each dataset's sigma
    tables = []
    for ds, path in FILES.items():
        d = _load_sigma(path, ds)
        if not d.empty:
            tables.append(d)
    if len(tables) < 2:
        raise RuntimeError("Could not load at least two datasets with sigma.")

    # Inner-join on key to keep subjects present in ALL sets
    wide = tables[0]
    for t in tables[1:]:
        wide = wide.merge(t, on="key", how="inner")

    keep_cols = ["key"] + DATASET_ORDER
    wide = wide[keep_cols].dropna()
    if wide.empty:
        raise RuntimeError("No overlapping subjects across QTN/GAF/MTF after matching keys.")

    # Save matched tables
    wide.to_csv(os.path.join(OUT_DIR, OUT_WIDE_CSV), index=False)
    long = wide.melt(id_vars=["key"], value_vars=DATASET_ORDER,
                     var_name="dataset", value_name="sigma")
    long["dataset"] = pd.Categorical(long["dataset"], categories=DATASET_ORDER, ordered=True)
    long.to_csv(os.path.join(OUT_DIR, OUT_LONG_CSV), index=False)

    # ---------- Plot ----------
    rng = np.random.default_rng(RNG_SEED)
    xcats = {ds: i for i, ds in enumerate(DATASET_ORDER)}
    fig, ax = plt.subplots(figsize=(9, 5.4))

    # Violin per dataset with your palette
    sns.violinplot(
        data=long, x="dataset", y="sigma", order=DATASET_ORDER,
        inner=None, cut=0, palette=PALETTE_DATASET, alpha=0.7, linewidth=1, saturation=1, ax=ax
    )

    # Box overlay (white fill)
    sns.boxplot(
        data=long, x="dataset", y="sigma", order=DATASET_ORDER,
        width=0.28, showcaps=True,
        boxprops={"facecolor":"white", "zorder":3},
        showfliers=False, whiskerprops={"linewidth":1}, medianprops={"linewidth":1.5},
        ax=ax
    )

    # Threshold line
    ax.axhline(SIGMA_THRESH, ls="--", lw=1.2, color="black", alpha=0.9)
    ax.text(len(DATASET_ORDER)-0.05, SIGMA_THRESH+0.01, f"σ = {SIGMA_THRESH}",
            ha="right", va="bottom", fontsize=10)

    # Grey, semi-transparent lines/points connecting same subject across methods
    line_color = "#9e9e9e"
    point_face = "#bdbdbd"
    point_edge = "#f2f2f2"

    for _, row in wide.iterrows():
        jitters = rng.normal(0, JITTER_SD, size=len(DATASET_ORDER))
        xs = [xcats[ds] + jitters[i] for i, ds in enumerate(DATASET_ORDER)]
        ys = [row[ds] for ds in DATASET_ORDER]
        # line
        ax.plot(xs, ys, color=line_color, alpha=0.3, lw=LINE_WIDTH, zorder=3.5)
        # points
        ax.scatter(xs, ys, s=POINT_SIZE, facecolor=point_face, edgecolor=point_edge,
                   linewidths=0.5, alpha=0.7, zorder=4)

    ax.set_xlabel("")
    ax.set_ylabel(r"Small-worldness $\sigma$")
   # ax.set_title("Paired σ across methods (QTN / GAF / MTF)\n"
            #     )
    sns.despine(trim=True)

    out_png = os.path.join(OUT_DIR, OUT_PLOT_PNG)
    out_pdf = os.path.join(OUT_DIR, OUT_PLOT_PDF)
    plt.tight_layout()
    plt.savefig(out_png, dpi=1000, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=1000, bbox_inches="tight")
    plt.close()
    print(f"[OK] wrote: {out_png}\n[OK] wrote: {out_pdf}")

if __name__ == "__main__":
    main()

# In[ ]:
