#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``heatmap-surrogate analysis.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[3]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
# compatibility for older packages
np.float = float
np.int = int
np.object = object
np.bool = bool
# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")

SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"
GROUP_CSV   = BASE_DIR / "td_group_summary_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "comparison_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QG", "GAF", "MTF"]

METRICS = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# ==========================================================
# HELPERS
# ==========================================================
def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return stat, p
    except Exception:
        return np.nan, np.nan


def rank_biserial_paired(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    return (n_pos - n_neg) / (n_pos + n_neg)


def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ==========================================================
# LOAD
# ==========================================================
subject_df = pd.read_csv(SUBJECT_CSV)
group_df = pd.read_csv(GROUP_CSV)

# ==========================================================
# SUBJECT-LEVEL STATISTICS
# ==========================================================
stat_rows = []

for rep in REP_ORDER:
    sub = subject_df[subject_df["method"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        stat, p = safe_wilcoxon(x, y)
        rbc = rank_biserial_paired(x, y)

        stat_rows.append({
            "representation": rep,
            "metric": metric,
            "n": np.sum(np.isfinite(x) & np.isfinite(y)),
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "rank_biserial": rbc,
            "mean_obs_subject": np.nanmean(x),
            "mean_surr_subject": np.nanmean(y),
            "mean_diff_subject": np.nanmean(x - y),
        })

stats_df = pd.DataFrame(stat_rows)

if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_fdr"] = p_corr
    stats_df["significant_fdr"] = reject

stats_df.to_csv(OUT_DIR / "subject_level_stats.csv", index=False)

# ==========================================================
# GROUP-LEVEL COMPARISON TABLE
# ==========================================================
group_rows = []

for _, row in group_df.iterrows():
    rep = row["method"]

    for metric in METRICS:
        obs_col = f"mean_obs_{metric}"
        surr_col = f"mean_surr_{metric}"

        if obs_col not in row.index or surr_col not in row.index:
            continue

        group_rows.append({
            "representation": rep,
            "metric": metric,
            "mean_obs_group": row[obs_col],
            "mean_surr_group": row[surr_col],
            "mean_diff_group": row[obs_col] - row[surr_col],
        })

group_long = pd.DataFrame(group_rows)

# merge group summary with subject-level stats
merged = group_long.merge(
    stats_df,
    on=["representation", "metric"],
    how="left"
)

merged.to_csv(OUT_DIR / "merged_group_subject_comparison.csv", index=False)

# ==========================================================
# HEATMAP 1: group mean difference + subject significance
# ==========================================================
pivot_diff = merged.pivot(
    index="metric",
    columns="representation",
    values="mean_diff_group"
).reindex(index=METRICS, columns=REP_ORDER)

pivot_p = merged.pivot(
    index="metric",
    columns="representation",
    values="wilcoxon_p_fdr"
).reindex(index=METRICS, columns=REP_ORDER)

annot = pivot_diff.copy().astype(object)
for i in annot.index:
    for j in annot.columns:
        val = pivot_diff.loc[i, j]
        p = pivot_p.loc[i, j]
        if pd.isna(val):
            annot.loc[i, j] = ""
        else:
            annot.loc[i, j] = f"{val:.3f}{p_to_stars(p)}"

plt.figure(figsize=(6.8, 6.2))
vlim = np.nanmax(np.abs(pivot_diff.to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

ax = sns.heatmap(
    pivot_diff,
    cmap="coolwarm",
    center=0,
    vmin=-vlim,
    vmax=vlim,
    linewidths=0.5,
    linecolor="white",
    annot=annot,
    fmt="",
    cbar_kws={"label": "Group mean difference (real - surrogate)"}
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("TD: Real vs surrogate by representation and metric", pad=12)
ax.set_xticklabels(REP_ORDER, rotation=0, fontsize=12)
ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in pivot_diff.index], rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_groupdiff_with_subjectstats.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "heatmap_groupdiff_with_subjectstats.pdf", dpi=600, bbox_inches="tight")
plt.close()

# ==========================================================
# HEATMAP 2: subject-level effect size
# ==========================================================
pivot_rbc = merged.pivot(
    index="metric",
    columns="representation",
    values="rank_biserial"
).reindex(index=METRICS, columns=REP_ORDER)

plt.figure(figsize=(6.8, 6.2))
vlim = np.nanmax(np.abs(pivot_rbc.to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

ax = sns.heatmap(
    pivot_rbc,
    cmap="coolwarm",
    center=0,
    vmin=-vlim,
    vmax=vlim,
    linewidths=0.5,
    linecolor="white",
    annot=True,
    fmt=".2f",
    cbar_kws={"label": "Rank-biserial effect size"}
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("TD: Subject-level effect sizes", pad=12)
ax.set_xticklabels(REP_ORDER, rotation=0, fontsize=12)
ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in pivot_rbc.index], rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_effectsize.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "heatmap_effectsize.pdf", dpi=600, bbox_inches="tight")
plt.close()

# ==========================================================
# HEATMAP 3: significance heatmap
# ==========================================================
plot_p = merged.copy()
plot_p["minuslog10_fdr_p"] = -np.log10(plot_p["wilcoxon_p_fdr"].clip(lower=1e-300))

pivot_sig = plot_p.pivot(
    index="metric",
    columns="representation",
    values="minuslog10_fdr_p"
).reindex(index=METRICS, columns=REP_ORDER)

annot_sig = pivot_sig.copy().astype(object)
for i in annot_sig.index:
    for j in annot_sig.columns:
        p = pivot_p.loc[i, j]
        if pd.isna(p):
            annot_sig.loc[i, j] = ""
        else:
            annot_sig.loc[i, j] = p_to_stars(p)

plt.figure(figsize=(6.8, 6.2))
ax = sns.heatmap(
    pivot_sig,
    cmap="magma",
    linewidths=0.5,
    linecolor="white",
    annot=annot_sig,
    fmt="",
    cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"}
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("TD: Subject-level significance", pad=12)
ax.set_xticklabels(REP_ORDER, rotation=0, fontsize=12)
ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in pivot_sig.index], rotation=0, fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_significance.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "heatmap_significance.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved to:", OUT_DIR)
print("Main heatmap:", OUT_DIR / "heatmap_groupdiff_with_subjectstats.png")

# In[5]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")

SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"
GROUP_CSV   = BASE_DIR / "td_group_summary_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "comparison_plots_pretty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QG", "GAF", "MTF"]

METRICS = [
    "sigma_small_world",
    "transitivity",
    "global_efficiency",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "char_path_len_gcc",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Transitivity",
    "char_path_len_gcc": "Path length",
    "global_efficiency": "Global efficiency",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

sns.set(style="white", context="talk")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12
})

# ==========================================================
# HELPERS
# ==========================================================
def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan
    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return stat, p
    except Exception:
        return np.nan, np.nan


def rank_biserial_paired(x, y):
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    return (n_pos - n_neg) / (n_pos + n_neg)


def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ==========================================================
# LOAD
# ==========================================================
subject_df = pd.read_csv(SUBJECT_CSV)
group_df = pd.read_csv(GROUP_CSV)

# ==========================================================
# STATS
# ==========================================================
stat_rows = []

for rep in REP_ORDER:
    sub = subject_df[subject_df["method"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        stat, p = safe_wilcoxon(x, y)
        rbc = rank_biserial_paired(x, y)

        stat_rows.append({
            "representation": rep,
            "metric": metric,
            "n": np.sum(np.isfinite(x) & np.isfinite(y)),
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
            "rank_biserial": rbc,
            "mean_obs_subject": np.nanmean(x),
            "mean_surr_subject": np.nanmean(y),
            "mean_diff_subject": np.nanmean(x - y),
        })

stats_df = pd.DataFrame(stat_rows)

if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_fdr"] = p_corr
    stats_df["significant_fdr"] = reject

# ==========================================================
# GROUP TABLE
# ==========================================================
group_rows = []

for _, row in group_df.iterrows():
    rep = row["method"]

    for metric in METRICS:
        obs_col = f"mean_obs_{metric}"
        surr_col = f"mean_surr_{metric}"

        if obs_col not in row.index or surr_col not in row.index:
            continue

        group_rows.append({
            "representation": rep,
            "metric": metric,
            "mean_obs_group": row[obs_col],
            "mean_surr_group": row[surr_col],
            "mean_diff_group": row[obs_col] - row[surr_col],
        })

group_long = pd.DataFrame(group_rows)

merged = group_long.merge(
    stats_df,
    on=["representation", "metric"],
    how="left"
)

merged.to_csv(OUT_DIR / "merged_group_subject_comparison.csv", index=False)

# ==========================================================
# KEEP ONLY INFORMATIVE METRICS
# ==========================================================
pivot_diff_full = merged.pivot(
    index="metric",
    columns="representation",
    values="mean_diff_group"
).reindex(index=METRICS, columns=REP_ORDER)

valid_rows = ~pivot_diff_full.isna().all(axis=1)
pivot_diff = pivot_diff_full.loc[valid_rows].copy()

pivot_p = merged.pivot(
    index="metric",
    columns="representation",
    values="wilcoxon_p_fdr"
).reindex(index=pivot_diff.index, columns=REP_ORDER)

pivot_rbc = merged.pivot(
    index="metric",
    columns="representation",
    values="rank_biserial"
).reindex(index=pivot_diff.index, columns=REP_ORDER)

# ==========================================================
# MAIN HEATMAP
# ==========================================================
annot = pivot_diff.copy().astype(object)

for i in annot.index:
    for j in annot.columns:
        val = pivot_diff.loc[i, j]
        p = pivot_p.loc[i, j]
        if pd.isna(val):
            annot.loc[i, j] = ""
        else:
            stars = p_to_stars(p)
            annot.loc[i, j] = f"{val:.3f}\n{stars}" if stars else f"{val:.3f}"

height = max(2.8, 0.9 * len(pivot_diff.index) + 1.2)

plt.figure(figsize=(6.0, height))
vlim = np.nanmax(np.abs(pivot_diff.to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

ax = sns.heatmap(
    pivot_diff,
    cmap="RdBu_r",
    center=0,
    vmin=-vlim,
    vmax=vlim,
    linewidths=1.2,
    linecolor="white",
    annot=annot,
    fmt="",
    square=False,
    cbar_kws={
        "label": "Mean difference\n(real - surrogate)",
        "shrink": 0.9
    }
)

ax.set_title("TD: real vs surrogate", pad=14, fontsize=20, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(REP_ORDER, rotation=0, fontsize=14, weight="bold")
ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in pivot_diff.index], rotation=0, fontsize=14)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_main_pretty.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "heatmap_main_pretty.pdf", dpi=600, bbox_inches="tight")
plt.close()

# ==========================================================
# EFFECT SIZE HEATMAP
# ==========================================================
plt.figure(figsize=(6.0, height))
vlim = np.nanmax(np.abs(pivot_rbc.to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

ax = sns.heatmap(
    pivot_rbc,
    cmap="RdBu_r",
    center=0,
    vmin=-vlim,
    vmax=vlim,
    linewidths=1.2,
    linecolor="white",
    annot=True,
    fmt=".2f",
    cbar_kws={
        "label": "Rank-biserial\neffect size",
        "shrink": 0.9
    }
)

ax.set_title("TD: effect sizes", pad=14, fontsize=20, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(REP_ORDER, rotation=0, fontsize=14, weight="bold")
ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in pivot_rbc.index], rotation=0, fontsize=14)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_effectsize_pretty.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "heatmap_effectsize_pretty.pdf", dpi=600, bbox_inches="tight")
plt.close()

# ==========================================================
# SIGNIFICANCE HEATMAP
# ==========================================================
plot_sig = -np.log10(pivot_p.clip(lower=1e-300))

annot_sig = plot_sig.copy().astype(object)
for i in annot_sig.index:
    for j in annot_sig.columns:
        p = pivot_p.loc[i, j]
        annot_sig.loc[i, j] = p_to_stars(p) if not pd.isna(p) else ""

plt.figure(figsize=(6.0, height))
ax = sns.heatmap(
    plot_sig,
    cmap="magma",
    linewidths=1.2,
    linecolor="white",
    annot=annot_sig,
    fmt="",
    cbar_kws={
        "label": r"$-\log_{10}(\mathrm{FDR}\ p)$",
        "shrink": 0.9
    }
)

ax.set_title("TD: significance", pad=14, fontsize=20, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(REP_ORDER, rotation=0, fontsize=14, weight="bold")
ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in plot_sig.index], rotation=0, fontsize=14)

plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_significance_pretty.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "heatmap_significance_pretty.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved pretty plots to:", OUT_DIR)

# In[6]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "barplots_pretty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_MAP_IN_FILE = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}   # if needed
REP_FILE_NAMES = ["QG", "GAF", "MTF"]  # names already in your surrogate csv

REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

PALETTE_REP = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

METRICS = [
    "sigma_small_world",
    "transitivity",
    "global_efficiency",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "char_path_len_gcc",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "transitivity": "Trans.",
    "global_efficiency": "GE",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "char_path_len_gcc": "Path",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 14
})


# ==========================================================
# HELPERS
# ==========================================================
def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return stat, p
    except Exception:
        return np.nan, np.nan


# ==========================================================
# LOAD
# ==========================================================
df = pd.read_csv(SUBJECT_CSV)

# map file representation names to desired display names if necessary
df["method_plot"] = df["method"].replace({
    "QG": "QTN",
    "GAF": "GAF",
    "MTF": "MTF"
})

# ==========================================================
# STATISTICS TABLE
# ==========================================================
rows = []

for rep in REP_ORDER:
    sub = df[df["method_plot"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        stat, p = safe_wilcoxon(x, y)

        rows.append({
            "representation": rep,
            "metric": metric,
            "n": np.sum(np.isfinite(x) & np.isfinite(y)),
            "mean_obs": np.nanmean(x),
            "mean_surr": np.nanmean(y),
            "mean_diff": np.nanmean(x - y),
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
        })

stats_df = pd.DataFrame(rows)

if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_fdr"] = p_corr
    stats_df["significant_fdr"] = reject

stats_df["metric_label"] = stats_df["metric"].map(METRIC_LABELS).fillna(stats_df["metric"])
stats_df["rep_label"] = stats_df["representation"].map(REP_DISPLAY)

stats_df.to_csv(OUT_DIR / "stats_barplot.csv", index=False)

# ==========================================================
# OPTIONAL: keep only metrics that have at least one finite value
# ==========================================================
metric_keep = []
for metric in METRICS:
    subm = stats_df[stats_df["metric"] == metric]
    if not subm.empty and np.isfinite(subm["mean_diff"]).any():
        metric_keep.append(metric)

stats_df = stats_df[stats_df["metric"].isin(metric_keep)].copy()

# ==========================================================
# BARPLOT
# ==========================================================
metric_positions = np.arange(len(metric_keep))
bar_width = 0.24
offsets = {
    "QTN": -bar_width,
    "GAF": 0.0,
    "MTF": +bar_width,
}

fig_width = max(10, 1.25 * len(metric_keep) + 2)
fig, ax = plt.subplots(figsize=(fig_width, 6.5))

all_y = []

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    x = metric_positions + offsets[rep]
    y = sub["mean_diff"].to_numpy()
    all_y.extend(y[np.isfinite(y)])

    ax.bar(
        x,
        y,
        width=bar_width * 0.95,
        color=PALETTE_REP[rep],
        edgecolor="black",
        linewidth=1.0,
        label=REP_DISPLAY[rep],
        alpha=0.95,
        zorder=3
    )

# zero line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=2)

# stars
if len(all_y) > 0:
    ymin = min(all_y)
    ymax = max(all_y)
    yrange = ymax - ymin if ymax != ymin else 1.0
else:
    ymin, ymax, yrange = -1, 1, 2

star_offset = 0.04 * yrange

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    xvals = metric_positions + offsets[rep]
    yvals = sub["mean_diff"].to_numpy()
    pvals = sub["wilcoxon_p_fdr"].to_numpy()

    for x, y, p in zip(xvals, yvals, pvals):
        if not np.isfinite(y):
            continue
        stars = p_to_stars(p)
        if stars:
            y_text = y + star_offset if y >= 0 else y - star_offset
            va = "bottom" if y >= 0 else "top"
            ax.text(
                x, y_text, stars,
                ha="center", va=va,
                fontsize=16, fontweight="bold"
            )

# axes formatting
ax.set_xticks(metric_positions)
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keep], rotation=0, fontsize=16)
ax.set_ylabel("Mean difference (real - surrogate)", fontsize=16)
ax.set_title("TD: surrogate comparison across representations", fontsize=24, fontweight="bold", pad=14)

# legend
leg = ax.legend(
    title="Representation",
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)

# spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved:", OUT_DIR / "barplot_all_metrics_real_minus_surrogate.png")
print("Saved:", OUT_DIR / "stats_barplot.csv")

# In[7]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "barplots_pretty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QTN", "GAF", "MTF"]

REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

PALETTE_REP = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

METRICS = [
    "sigma_small_world",
    "transitivity",
    "global_efficiency",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "char_path_len_gcc",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "transitivity": "Trans.",
    "global_efficiency": "GE",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "char_path_len_gcc": "Path",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 14
})


# ==========================================================
# HELPERS
# ==========================================================
def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return stat, p
    except Exception:
        return np.nan, np.nan


# ==========================================================
# LOAD
# ==========================================================
df = pd.read_csv(SUBJECT_CSV)

# map file names in the surrogate CSV to plotting groups
df["method_plot"] = df["method"].replace({
    "QG": "QTN",
    "GAF": "GAF",
    "MTF": "MTF"
})

# ==========================================================
# STATISTICS TABLE
# ==========================================================
rows = []

for rep in REP_ORDER:
    sub = df[df["method_plot"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]
        diff = x_valid - y_valid

        stat, p = safe_wilcoxon(x_valid, y_valid)

        rows.append({
            "representation": rep,
            "metric": metric,
            "n": len(diff),
            "mean_obs": np.nanmean(x_valid) if len(x_valid) else np.nan,
            "mean_surr": np.nanmean(y_valid) if len(y_valid) else np.nan,
            "mean_diff": np.nanmean(diff) if len(diff) else np.nan,
            "std_diff": np.nanstd(diff, ddof=1) if len(diff) > 1 else np.nan,
            "sem_diff": (np.nanstd(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
        })

stats_df = pd.DataFrame(rows)

if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_fdr"] = p_corr
    stats_df["significant_fdr"] = reject

stats_df["metric_label"] = stats_df["metric"].map(METRIC_LABELS).fillna(stats_df["metric"])
stats_df["rep_label"] = stats_df["representation"].map(REP_DISPLAY)

stats_df.to_csv(OUT_DIR / "stats_barplot.csv", index=False)

# ==========================================================
# KEEP ONLY METRICS WITH AT LEAST ONE FINITE MEAN
# ==========================================================
metric_keep = []
for metric in METRICS:
    subm = stats_df[stats_df["metric"] == metric]
    if not subm.empty and np.isfinite(subm["mean_diff"]).any():
        metric_keep.append(metric)

stats_df = stats_df[stats_df["metric"].isin(metric_keep)].copy()

# ==========================================================
# BARPLOT
# ==========================================================
metric_positions = np.arange(len(metric_keep))
bar_width = 0.24
offsets = {
    "QTN": -bar_width,
    "GAF": 0.0,
    "MTF": +bar_width,
}

fig_width = max(10, 1.25 * len(metric_keep) + 2)
fig, ax = plt.subplots(figsize=(fig_width, 6.8))

all_y = []
all_err = []

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    x = metric_positions + offsets[rep]
    y = sub["mean_diff"].to_numpy(dtype=float)
   # yerr = sub["std_diff"].to_numpy(dtype=float)   # change to sem_diff if you prefer SEM
    yerr = sub["sem_diff"].to_numpy(dtype=float)
    all_y.extend(y[np.isfinite(y)])
    all_err.extend(yerr[np.isfinite(yerr)])

    ax.bar(
        x,
        y,
        width=bar_width * 0.95,
        color=PALETTE_REP[rep],
        edgecolor="black",
        linewidth=1.0,
        label=REP_DISPLAY[rep],
        alpha=0.95,
        zorder=3,
        yerr=yerr,
        error_kw={
            "elinewidth": 1.5,
            "ecolor": "black",
            "capsize": 4,
            "capthick": 1.5
        }
    )

# zero line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=2)

# stars
finite_y = np.asarray(all_y, dtype=float)
finite_y = finite_y[np.isfinite(finite_y)]
finite_err = np.asarray(all_err, dtype=float)
finite_err = finite_err[np.isfinite(finite_err)]

if len(finite_y) > 0:
    ymin = float(np.min(finite_y))
    ymax = float(np.max(finite_y))
else:
    ymin, ymax = -1.0, 1.0

max_err = float(np.max(finite_err)) if len(finite_err) > 0 else 0.0
yrange = (ymax - ymin) if ymax != ymin else 1.0
star_offset = max(0.04 * yrange, 0.25 * max_err if max_err > 0 else 0.04 * yrange)

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    xvals = metric_positions + offsets[rep]
    yvals = sub["mean_diff"].to_numpy(dtype=float)
    errvals = sub["std_diff"].to_numpy(dtype=float)
    pvals = sub["wilcoxon_p_fdr"].to_numpy(dtype=float)

    for x, y, err, p in zip(xvals, yvals, errvals, pvals):
        if not np.isfinite(y):
            continue

        err = 0.0 if not np.isfinite(err) else err
        stars = p_to_stars(p)
        if stars:
            if y >= 0:
                y_text = y + err + star_offset
                va = "bottom"
            else:
                y_text = y - err - star_offset
                va = "top"

            ax.text(
                x, y_text, stars,
                ha="center", va=va,
                fontsize=16, fontweight="bold"
            )

# axes formatting
ax.set_xticks(metric_positions)
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keep], rotation=0, fontsize=16)
ax.set_ylabel("Mean difference (real - surrogate)", fontsize=16)
#ax.set_title("TD: surrogate comparison across representations", fontsize=24, fontweight="bold", pad=14)

# legend
ax.legend(
    title="Representation",
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)

# spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_sd.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_sd.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved:", OUT_DIR / "barplot_all_metrics_real_minus_surrogate_sd.png")
print("Saved:", OUT_DIR / "stats_barplot.csv")

# In[8]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "barplots_pretty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QTN", "GAF", "MTF"]

REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

PALETTE_REP = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# Keep all metrics here, or reduce later if you want a cleaner figure
METRICS = [
    "sigma_small_world",
    "transitivity",
    "global_efficiency",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "char_path_len_gcc",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "transitivity": "Trans.",
    "global_efficiency": "GE",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "char_path_len_gcc": "Path",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

# More compact / publication-friendly defaults
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

USE_SEM = True  # SEM is more common for this kind of bar plot


# ==========================================================
# HELPERS
# ==========================================================
def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return stat, p
    except Exception:
        return np.nan, np.nan


# ==========================================================
# LOAD
# ==========================================================
df = pd.read_csv(SUBJECT_CSV)

# map the names in the surrogate CSV to the plot representation groups
df["method_plot"] = df["method"].replace({
    "QG": "QTN",
    "GAF": "GAF",
    "MTF": "MTF"
})

# ==========================================================
# BUILD SUBJECT-LEVEL SUMMARY TABLE
# ==========================================================
rows = []

for rep in REP_ORDER:
    sub = df[df["method_plot"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]
        diff = x_valid - y_valid

        stat, p = safe_wilcoxon(x_valid, y_valid)

        rows.append({
            "representation": rep,
            "metric": metric,
            "n": len(diff),
            "mean_obs": np.nanmean(x_valid) if len(x_valid) else np.nan,
            "mean_surr": np.nanmean(y_valid) if len(y_valid) else np.nan,
            "mean_diff": np.nanmean(diff) if len(diff) else np.nan,
            "std_diff": np.nanstd(diff, ddof=1) if len(diff) > 1 else np.nan,
            "sem_diff": (np.nanstd(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
            "wilcoxon_stat": stat,
            "wilcoxon_p": p,
        })

stats_df = pd.DataFrame(rows)

# FDR correction across all rep x metric tests
if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_fdr"] = p_corr
    stats_df["significant_fdr"] = reject

stats_df["metric_label"] = stats_df["metric"].map(METRIC_LABELS).fillna(stats_df["metric"])
stats_df["rep_label"] = stats_df["representation"].map(REP_DISPLAY)

stats_df.to_csv(OUT_DIR / "stats_barplot.csv", index=False)

# ==========================================================
# KEEP ONLY METRICS WITH AT LEAST ONE FINITE VALUE
# ==========================================================
metric_keep = []
for metric in METRICS:
    subm = stats_df[stats_df["metric"] == metric]
    if not subm.empty and np.isfinite(subm["mean_diff"]).any():
        metric_keep.append(metric)

stats_df = stats_df[stats_df["metric"].isin(metric_keep)].copy()

# If you want a cleaner final figure, uncomment this block:
# preferred_metrics = [
#     "sigma_small_world",
#     "transitivity",
#     "global_efficiency",
#     "gamma_C_over_Crand",
#     "lambda_L_over_Lrand",
#     "char_path_len_gcc",
# ]
# metric_keep = [m for m in preferred_metrics if m in stats_df["metric"].unique()]
# stats_df = stats_df[stats_df["metric"].isin(metric_keep)].copy()

# ==========================================================
# BARPLOT
# ==========================================================
metric_positions = np.arange(len(metric_keep))
bar_width = 0.24
offsets = {
    "QTN": -bar_width,
    "GAF": 0.0,
    "MTF": +bar_width,
}

fig_width = max(10, 1.25 * len(metric_keep) + 2)
fig, ax = plt.subplots(figsize=(fig_width, 6.5))

all_bar_tops = []
all_bar_bottoms = []

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    x = metric_positions + offsets[rep]
    y = sub["mean_diff"].to_numpy(dtype=float)

    if USE_SEM:
        yerr = sub["sem_diff"].to_numpy(dtype=float)
    else:
        yerr = sub["std_diff"].to_numpy(dtype=float)

    yerr = np.nan_to_num(yerr, nan=0.0)

    ax.bar(
        x,
        y,
        width=bar_width * 0.95,
        color=PALETTE_REP[rep],
        edgecolor="black",
        linewidth=1.0,
        label=REP_DISPLAY[rep],
        alpha=0.95,
        zorder=3,
        yerr=yerr,
        error_kw={
            "elinewidth": 1.4,
            "ecolor": "black",
            "capsize": 3,
            "capthick": 1.4
        }
    )

    all_bar_tops.extend((y + yerr)[np.isfinite(y + yerr)])
    all_bar_bottoms.extend((y - yerr)[np.isfinite(y - yerr)])

# zero line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=2)

# ==========================================================
# STARS
# ==========================================================
if len(all_bar_tops) > 0 and len(all_bar_bottoms) > 0:
    ymax_data = float(np.max(all_bar_tops))
    ymin_data = float(np.min(all_bar_bottoms))
else:
    ymax_data, ymin_data = 1.0, -1.0

yrange = ymax_data - ymin_data
if yrange == 0:
    yrange = 1.0

star_offset = 0.03 * yrange

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    xvals = metric_positions + offsets[rep]
    yvals = sub["mean_diff"].to_numpy(dtype=float)

    if USE_SEM:
        errvals = sub["sem_diff"].to_numpy(dtype=float)
    else:
        errvals = sub["std_diff"].to_numpy(dtype=float)

    errvals = np.nan_to_num(errvals, nan=0.0)
    pvals = sub["wilcoxon_p_fdr"].to_numpy(dtype=float)

    for x, y, err, p in zip(xvals, yvals, errvals, pvals):
        if not np.isfinite(y):
            continue

        stars = p_to_stars(p)
        if not stars:
            continue

        if y >= 0:
            y_text = y + err + star_offset
            va = "bottom"
        else:
            y_text = y - err - star_offset
            va = "top"

        ax.text(
            x, y_text, stars,
            ha="center", va=va,
            fontsize=15, fontweight="bold"
        )

# ==========================================================
# AXES / LAYOUT
# ==========================================================
top_margin = 0.10 * yrange
bottom_margin = 0.10 * yrange
ax.set_ylim(ymin_data - bottom_margin, ymax_data + top_margin + 2 * star_offset)

ax.set_xticks(metric_positions)
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keep], rotation=0)
ax.set_ylabel("Mean difference (real - surrogate)")

ax.legend(
    title="Representation",
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_sem.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_sem.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved:", OUT_DIR / "barplot_all_metrics_real_minus_surrogate_sem.png")
print("Saved:", OUT_DIR / "stats_barplot.csv")

# In[9]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "barplots_pretty_onesided"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QTN", "GAF", "MTF"]

REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

PALETTE_REP = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

METRICS = [
    "sigma_small_world",
    "transitivity",
    "global_efficiency",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "char_path_len_gcc",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "transitivity": "Trans.",
    "global_efficiency": "GE",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "char_path_len_gcc": "Path",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

USE_SEM = True  # more common for bar plots


# ==========================================================
# HELPERS
# ==========================================================
def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def safe_wilcoxon_greater(x, y):
    """
    One-sided paired Wilcoxon:
    H1 = x > y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="greater")
        return stat, p
    except Exception:
        return np.nan, np.nan


# ==========================================================
# LOAD
# ==========================================================
df = pd.read_csv(SUBJECT_CSV)

# map names from surrogate csv to plot groups
df["method_plot"] = df["method"].replace({
    "QG": "QTN",
    "GAF": "GAF",
    "MTF": "MTF"
})

# ==========================================================
# SUBJECT-LEVEL SUMMARY TABLE
# ==========================================================
rows = []

for rep in REP_ORDER:
    sub = df[df["method_plot"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]
        diff = x_valid - y_valid

        stat, p = safe_wilcoxon_greater(x_valid, y_valid)

        rows.append({
            "representation": rep,
            "metric": metric,
            "n": len(diff),
            "mean_obs": np.nanmean(x_valid) if len(x_valid) else np.nan,
            "mean_surr": np.nanmean(y_valid) if len(y_valid) else np.nan,
            "mean_diff": np.nanmean(diff) if len(diff) else np.nan,
            "std_diff": np.nanstd(diff, ddof=1) if len(diff) > 1 else np.nan,
            "sem_diff": (np.nanstd(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
            "wilcoxon_stat": stat,
            "wilcoxon_p_greater": p,
        })

stats_df = pd.DataFrame(rows)

# FDR correction across all rep x metric tests
if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p_greater"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_greater_fdr"] = p_corr
    stats_df["significant_greater_fdr"] = reject

stats_df["metric_label"] = stats_df["metric"].map(METRIC_LABELS).fillna(stats_df["metric"])
stats_df["rep_label"] = stats_df["representation"].map(REP_DISPLAY)

stats_df.to_csv(OUT_DIR / "stats_barplot_onesided_greater.csv", index=False)

# ==========================================================
# KEEP ONLY METRICS WITH AT LEAST ONE FINITE VALUE
# ==========================================================
metric_keep = []
for metric in METRICS:
    subm = stats_df[stats_df["metric"] == metric]
    if not subm.empty and np.isfinite(subm["mean_diff"]).any():
        metric_keep.append(metric)

stats_df = stats_df[stats_df["metric"].isin(metric_keep)].copy()

# ==========================================================
# BARPLOT
# ==========================================================
metric_positions = np.arange(len(metric_keep))
bar_width = 0.24
offsets = {
    "QTN": -bar_width,
    "GAF": 0.0,
    "MTF": +bar_width,
}

fig_width = max(10, 1.25 * len(metric_keep) + 2)
fig, ax = plt.subplots(figsize=(fig_width, 6.5))

all_bar_tops = []
all_bar_bottoms = []

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    x = metric_positions + offsets[rep]
    y = sub["mean_diff"].to_numpy(dtype=float)

    if USE_SEM:
        yerr = sub["sem_diff"].to_numpy(dtype=float)
    else:
        yerr = sub["std_diff"].to_numpy(dtype=float)

    yerr = np.nan_to_num(yerr, nan=0.0)

    ax.bar(
        x,
        y,
        width=bar_width * 0.95,
        color=PALETTE_REP[rep],
        edgecolor="black",
        linewidth=1.0,
        label=REP_DISPLAY[rep],
        alpha=0.95,
        zorder=3,
        yerr=yerr,
        error_kw={
            "elinewidth": 1.4,
            "ecolor": "black",
            "capsize": 3,
            "capthick": 1.4
        }
    )

    all_bar_tops.extend((y + yerr)[np.isfinite(y + yerr)])
    all_bar_bottoms.extend((y - yerr)[np.isfinite(y - yerr)])

# zero line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=2)

# ==========================================================
# STARS FOR ONE-SIDED TEST
# Only positive and significant bars get stars
# ==========================================================
if len(all_bar_tops) > 0 and len(all_bar_bottoms) > 0:
    ymax_data = float(np.max(all_bar_tops))
    ymin_data = float(np.min(all_bar_bottoms))
else:
    ymax_data, ymin_data = 1.0, -1.0

yrange = ymax_data - ymin_data
if yrange == 0:
    yrange = 1.0

star_offset = 0.03 * yrange

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    xvals = metric_positions + offsets[rep]
    yvals = sub["mean_diff"].to_numpy(dtype=float)

    if USE_SEM:
        errvals = sub["sem_diff"].to_numpy(dtype=float)
    else:
        errvals = sub["std_diff"].to_numpy(dtype=float)

    errvals = np.nan_to_num(errvals, nan=0.0)
    pvals = sub["wilcoxon_p_greater_fdr"].to_numpy(dtype=float)

    for x, y, err, p in zip(xvals, yvals, errvals, pvals):
        if not np.isfinite(y):
            continue

        # One-sided interpretation: only positive bars can support the claim
        if y <= 0:
            continue

        stars = p_to_stars(p)
        if not stars:
            continue

        y_text = y + err + star_offset

        ax.text(
            x, y_text, stars,
            ha="center", va="bottom",
            fontsize=15, fontweight="bold"
        )

# ==========================================================
# AXES / LAYOUT
# ==========================================================
top_margin = 0.12 * yrange
bottom_margin = 0.10 * yrange
ax.set_ylim(ymin_data - bottom_margin, ymax_data + top_margin + 2 * star_offset)

ax.set_xticks(metric_positions)
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keep], rotation=0)
ax.set_ylabel("Mean difference (real - surrogate)")

ax.legend(
    title="Representation",
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_onesided_greater_sem.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_onesided_greater_sem.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved:", OUT_DIR / "barplot_all_metrics_real_minus_surrogate_onesided_greater_sem.png")
print("Saved:", OUT_DIR / "stats_barplot_onesided_greater.csv")

# In[10]:

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = Path("abide_td_surrogate_qg_gaf_mtf")
SUBJECT_CSV = BASE_DIR / "td_subject_surrogate_qg_gaf_mtf.csv"

OUT_DIR = BASE_DIR / "barplots_pretty_onesided"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QTN", "GAF", "MTF"]

REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

PALETTE_REP = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

METRICS = [
    "sigma_small_world",
    "transitivity",
    "global_efficiency",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "char_path_len_gcc",
    "omega",
    "phi",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "transitivity": "Trans.",
    "global_efficiency": "GE",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "char_path_len_gcc": "Path",
    "omega": r"$\omega$",
    "phi": r"$\phi$",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False,
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

USE_SEM = True  # more common for bar plots


# ==========================================================
# HELPERS
# ==========================================================
def p_to_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def safe_wilcoxon_greater(x, y):
    """
    One-sided paired Wilcoxon:
    H1 = x > y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan

    if np.allclose(x, y, equal_nan=True):
        return 0.0, 1.0

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="greater")
        return stat, p
    except Exception:
        return np.nan, np.nan


# ==========================================================
# LOAD
# ==========================================================
df = pd.read_csv(SUBJECT_CSV)

# map names from surrogate csv to plot groups
df["method_plot"] = df["method"].replace({
    "QG": "QTN",
    "GAF": "GAF",
    "MTF": "MTF"
})

# ==========================================================
# SUBJECT-LEVEL SUMMARY TABLE
# ==========================================================
rows = []

for rep in REP_ORDER:
    sub = df[df["method_plot"] == rep].copy()
    if sub.empty:
        continue

    for metric in METRICS:
        obs_col = f"obs_{metric}"
        surr_col = f"surr_mean_{metric}"

        if obs_col not in sub.columns or surr_col not in sub.columns:
            continue

        x = pd.to_numeric(sub[obs_col], errors="coerce").to_numpy()
        y = pd.to_numeric(sub[surr_col], errors="coerce").to_numpy()

        mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[mask]
        y_valid = y[mask]
        diff = x_valid - y_valid

        stat, p = safe_wilcoxon_greater(x_valid, y_valid)

        rows.append({
            "representation": rep,
            "metric": metric,
            "n": len(diff),
            "mean_obs": np.nanmean(x_valid) if len(x_valid) else np.nan,
            "mean_surr": np.nanmean(y_valid) if len(y_valid) else np.nan,
            "mean_diff": np.nanmean(diff) if len(diff) else np.nan,
            "std_diff": np.nanstd(diff, ddof=1) if len(diff) > 1 else np.nan,
            "sem_diff": (np.nanstd(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan,
            "wilcoxon_stat": stat,
            "wilcoxon_p_greater": p,
        })

stats_df = pd.DataFrame(rows)

# FDR correction across all rep x metric tests
if not stats_df.empty:
    reject, p_corr, _, _ = multipletests(
        stats_df["wilcoxon_p_greater"].fillna(1.0),
        method="fdr_bh"
    )
    stats_df["wilcoxon_p_greater_fdr"] = p_corr
    stats_df["significant_greater_fdr"] = reject

stats_df["metric_label"] = stats_df["metric"].map(METRIC_LABELS).fillna(stats_df["metric"])
stats_df["rep_label"] = stats_df["representation"].map(REP_DISPLAY)

stats_df.to_csv(OUT_DIR / "stats_barplot_onesided_greater.csv", index=False)

# ==========================================================
# KEEP ONLY METRICS WITH AT LEAST ONE FINITE VALUE
# ==========================================================
metric_keep = []
for metric in METRICS:
    subm = stats_df[stats_df["metric"] == metric]
    if not subm.empty and np.isfinite(subm["mean_diff"]).any():
        metric_keep.append(metric)

stats_df = stats_df[stats_df["metric"].isin(metric_keep)].copy()

# ==========================================================
# BARPLOT
# ==========================================================
metric_positions = np.arange(len(metric_keep))
bar_width = 0.24
offsets = {
    "QTN": -bar_width,
    "GAF": 0.0,
    "MTF": +bar_width,
}

fig_width = max(10, 1.25 * len(metric_keep) + 2)
fig, ax = plt.subplots(figsize=(fig_width, 6.5))

all_bar_tops = []
all_bar_bottoms = []

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    x = metric_positions + offsets[rep]
    y = sub["mean_diff"].to_numpy(dtype=float)

    if USE_SEM:
        yerr = sub["sem_diff"].to_numpy(dtype=float)
    else:
        yerr = sub["std_diff"].to_numpy(dtype=float)

    yerr = np.nan_to_num(yerr, nan=0.0)

    ax.bar(
        x,
        y,
        width=bar_width * 0.95,
        color=PALETTE_REP[rep],
        edgecolor="black",
        linewidth=1.0,
        label=REP_DISPLAY[rep],
        alpha=0.95,
        zorder=3,
        yerr=yerr,
        error_kw={
            "elinewidth": 1.4,
            "ecolor": "black",
            "capsize": 3,
            "capthick": 1.4
        }
    )

    all_bar_tops.extend((y + yerr)[np.isfinite(y + yerr)])
    all_bar_bottoms.extend((y - yerr)[np.isfinite(y - yerr)])

# zero line
ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=2)

# ==========================================================
# STARS FOR ONE-SIDED TEST
# Only positive and significant bars get stars
# ==========================================================
if len(all_bar_tops) > 0 and len(all_bar_bottoms) > 0:
    ymax_data = float(np.max(all_bar_tops))
    ymin_data = float(np.min(all_bar_bottoms))
else:
    ymax_data, ymin_data = 1.0, -1.0

yrange = ymax_data - ymin_data
if yrange == 0:
    yrange = 1.0

star_offset = 0.03 * yrange

for rep in REP_ORDER:
    sub = stats_df[stats_df["representation"] == rep].copy()
    sub = sub.set_index("metric").reindex(metric_keep).reset_index()

    xvals = metric_positions + offsets[rep]
    yvals = sub["mean_diff"].to_numpy(dtype=float)

    if USE_SEM:
        errvals = sub["sem_diff"].to_numpy(dtype=float)
    else:
        errvals = sub["std_diff"].to_numpy(dtype=float)

    errvals = np.nan_to_num(errvals, nan=0.0)
    pvals = sub["wilcoxon_p_greater_fdr"].to_numpy(dtype=float)

    for x, y, err, p in zip(xvals, yvals, errvals, pvals):
        if not np.isfinite(y):
            continue

        # One-sided interpretation: only positive bars can support the claim
        if y <= 0:
            continue

        stars = p_to_stars(p)
        if not stars:
            continue

        y_text = y + err + star_offset

        ax.text(
            x, y_text, stars,
            ha="center", va="bottom",
            fontsize=15, fontweight="bold"
        )

# ==========================================================
# AXES / LAYOUT
# ==========================================================
top_margin = 0.12 * yrange
bottom_margin = 0.10 * yrange
ax.set_ylim(ymin_data - bottom_margin, ymax_data + top_margin + 2 * star_offset)

ax.set_xticks(metric_positions)
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_keep], rotation=0)
ax.set_ylabel("Mean difference (real - surrogate)")

ax.legend(
    title="Representation",
    frameon=False,
    fontsize=14,
    title_fontsize=15,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.12)
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_onesided_greater_sem.png", dpi=600, bbox_inches="tight")
plt.savefig(OUT_DIR / "barplot_all_metrics_real_minus_surrogate_onesided_greater_sem.pdf", dpi=600, bbox_inches="tight")
plt.close()

print("Saved:", OUT_DIR / "barplot_all_metrics_real_minus_surrogate_onesided_greater_sem.png")
print("Saved:", OUT_DIR / "stats_barplot_onesided_greater.csv")

# In[ ]:
