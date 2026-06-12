#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``Statitical-test.ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[1]:

import os
import re
import glob
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ---------------------------------------
# CONFIG
# ---------------------------------------
INPUT_DIR = "csv"
OUT_DIR = "stats_from_csv"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}

METRICS_TO_TEST = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
    "zC",
    "zL",
    "density",
    "n_nodes",
]

# ---------------------------------------
# HELPERS
# ---------------------------------------
def extract_rep_dataset(filepath: str):
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper(), m.group(2)

def normalize_key(x):
    return str(x).strip()

def load_metric_table(path: str, rep: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if metric not in df.columns:
        return pd.DataFrame()

    if "patient_id" in df.columns:
        ids = df["patient_id"].astype(str).map(normalize_key)
    elif "subject_id" in df.columns:
        ids = df["subject_id"].astype(str).map(normalize_key)
    elif "file_id" in df.columns:
        ids = df["file_id"].astype(str).map(normalize_key)
    else:
        ids = pd.Series([f"{rep}_{i}" for i in range(len(df))], name="id")

    out = pd.DataFrame({
        "key": ids,
        rep: pd.to_numeric(df[metric], errors="coerce")
    }).dropna(subset=[rep])

    out = out.groupby("key", as_index=False)[rep].mean(numeric_only=True)
    return out

def merge_three_reps(qtn, gaf, mtf):
    if qtn.empty or gaf.empty or mtf.empty:
        return pd.DataFrame()
    wide = qtn.merge(gaf, on="key", how="inner").merge(mtf, on="key", how="inner")
    return wide[["key", "QTN", "GAF", "MTF"]].dropna()

def rank_biserial_from_wilcoxon(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    n_pos = np.sum(d > 0)
    n_neg = np.sum(d < 0)
    return (n_pos - n_neg) / (n_pos + n_neg)

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "metrics_*.csv")))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {INPUT_DIR}")

    grouped = {}
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue
        rep, dataset = parsed
        grouped.setdefault(dataset, {})
        grouped[dataset][rep] = f

    omnibus_rows = []
    posthoc_rows = []

    for dataset_name, rep_files in grouped.items():
        if not all(rep in rep_files for rep in REP_ORDER):
            continue

        print(f"\n===== DATASET: {dataset_name} =====")

        for metric in METRICS_TO_TEST:
            qtn = load_metric_table(rep_files["QTN"], "QTN", metric)
            gaf = load_metric_table(rep_files["GAF"], "GAF", metric)
            mtf = load_metric_table(rep_files["MTF"], "MTF", metric)

            wide = merge_three_reps(qtn, gaf, mtf)
            if wide.empty or len(wide) < 3:
                print(f"[SKIP] {dataset_name} | {metric}: not enough matched samples")
                continue

            x_qtn = wide["QTN"].to_numpy()
            x_gaf = wide["GAF"].to_numpy()
            x_mtf = wide["MTF"].to_numpy()

            # omnibus Friedman
            try:
                stat, p = friedmanchisquare(x_qtn, x_gaf, x_mtf)
            except Exception:
                stat, p = np.nan, np.nan

            omnibus_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "n_matched": len(wide),
                "friedman_stat": stat,
                "friedman_p": p,
                "QG_mean": np.mean(x_qtn),
                "GAF_mean": np.mean(x_gaf),
                "MTF_mean": np.mean(x_mtf),
                "QG_median": np.median(x_qtn),
                "GAF_median": np.median(x_gaf),
                "MTF_median": np.median(x_mtf),
            })

            # post hoc only if omnibus valid
            pairs = [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]
            for a, b in pairs:
                xa = wide[a].to_numpy()
                xb = wide[b].to_numpy()

                try:
                    w_stat, w_p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    w_stat, w_p = np.nan, np.nan

                posthoc_rows.append({
                    "dataset": dataset_name,
                    "metric": metric,
                    "comparison": f"{REP_DISPLAY[a]} vs {REP_DISPLAY[b]}",
                    "n_matched": len(wide),
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_p": w_p,
                    "rank_biserial": rank_biserial_from_wilcoxon(xa, xb),
                    f"{REP_DISPLAY[a]}_mean": np.mean(xa),
                    f"{REP_DISPLAY[b]}_mean": np.mean(xb),
                    f"{REP_DISPLAY[a]}_median": np.median(xa),
                    f"{REP_DISPLAY[b]}_median": np.median(xb),
                })

            print(f"[OK] {dataset_name} | {metric} | n={len(wide)}")

    omnibus_df = pd.DataFrame(omnibus_rows)
    posthoc_df = pd.DataFrame(posthoc_rows)

    if not omnibus_df.empty:
        reject, p_corr, _, _ = multipletests(omnibus_df["friedman_p"].fillna(1.0), method="fdr_bh")
        omnibus_df["friedman_p_fdr"] = p_corr
        omnibus_df["friedman_significant_fdr"] = reject

    if not posthoc_df.empty:
        reject, p_corr, _, _ = multipletests(posthoc_df["wilcoxon_p"].fillna(1.0), method="fdr_bh")
        posthoc_df["wilcoxon_p_fdr"] = p_corr
        posthoc_df["wilcoxon_significant_fdr"] = reject

    omnibus_df.to_csv(os.path.join(OUT_DIR, "friedman_omnibus_results.csv"), index=False)
    posthoc_df.to_csv(os.path.join(OUT_DIR, "wilcoxon_posthoc_results.csv"), index=False)

    print("\n[DONE] Wrote:")
    print(os.path.join(OUT_DIR, "friedman_omnibus_results.csv"))
    print(os.path.join(OUT_DIR, "wilcoxon_posthoc_results.csv"))

if __name__ == "__main__":
    main()

# In[3]:


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.float = float
np.int = int
np.object = object
np.bool = bool

# --------------------------------------------------
# INPUT
# --------------------------------------------------
FRIEDMAN_CSV = "stats_from_csv/friedman_omnibus_results.csv"
WILCOXON_CSV = "stats_from_csv/wilcoxon_posthoc_results.csv"
OUT_DIR = "stat_plots"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# STYLE
# --------------------------------------------------
sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# dataset order if you want a fixed order
DATASET_ORDER = [
    "ABIDE", "ADHD", "SCZ", "MEArecs",
    "CaFast_Sham_byDIV",
    "cap_sleep_controls_ALL_SLEEP_NREM",
    "cap_sleep_controls_EEG_NREM",
    "cap_sleep_controls_EMG_NREM",
    "cap_sleep_controls_RESP_NREM",
    "emg_plantar_EMGONLY_per_subject",
    "fantasia_all",
    "nsrdb",
    "resp_aeration_stream_per_subject"
]

METRIC_ORDER = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
    "zC",
    "zL"
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Transitivity",
    "char_path_len_gcc": "Path length",
    "global_efficiency": "Global efficiency",
    "zC": r"$z_C$",
    "zL": r"$z_L$"
}

PAIR_ORDER = ["QG vs GAF", "QG vs MTF", "GAF vs MTF"]

# --------------------------------------------------
# LOAD
# --------------------------------------------------
friedman = pd.read_csv(FRIEDMAN_CSV)
wilcoxon = pd.read_csv(WILCOXON_CSV)

# keep only the metrics of interest
friedman = friedman[friedman["metric"].isin(METRIC_ORDER)].copy()
wilcoxon = wilcoxon[wilcoxon["metric"].isin(METRIC_ORDER)].copy()

# rename comparisons if needed
wilcoxon["comparison"] = wilcoxon["comparison"].replace({
    "QTN vs GAF": "QG vs GAF",
    "QTN vs MTF": "QG vs MTF",
    "QG vs GAF": "QG vs GAF",
    "QG vs MTF": "QG vs MTF",
    "GAF vs MTF": "GAF vs MTF"
})

# keep only known ordering if present
dataset_order = [d for d in DATASET_ORDER if d in friedman["dataset"].unique()]
dataset_order += [d for d in friedman["dataset"].unique() if d not in dataset_order]

metric_order = [m for m in METRIC_ORDER if m in friedman["metric"].unique()]
metric_order += [m for m in friedman["metric"].unique() if m not in metric_order]

# --------------------------------------------------
# FIGURE A: Friedman heatmap
# --------------------------------------------------
friedman_plot = friedman.copy()
friedman_plot["value_plot"] = -np.log10(friedman_plot["friedman_p_fdr"].clip(lower=1e-300))

pivot_f = friedman_plot.pivot(index="dataset", columns="metric", values="value_plot")
pivot_f = pivot_f.reindex(index=dataset_order, columns=metric_order)

sig_f = friedman_plot.pivot(index="dataset", columns="metric", values="friedman_significant_fdr")
sig_f = sig_f.reindex(index=dataset_order, columns=metric_order)

annot_f = pivot_f.copy().astype(object)
for i in annot_f.index:
    for j in annot_f.columns:
        if pd.isna(pivot_f.loc[i, j]):
            annot_f.loc[i, j] = ""
        else:
            annot_f.loc[i, j] = "*" if bool(sig_f.loc[i, j]) else ""

plt.figure(figsize=(1.25 * len(metric_order) + 3, 0.55 * len(dataset_order) + 2))
ax = sns.heatmap(
    pivot_f,
    cmap="viridis",
    linewidths=0.5,
    linecolor="white",
    annot=annot_f,
    fmt="",
    cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"}
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Overall paired differences across representations (Friedman test)")
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in pivot_f.columns], rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "friedman_heatmap.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "friedman_heatmap.pdf"), dpi=600, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FIGURE B: Wilcoxon adjusted p-value heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(5.2 * 3, 0.55 * len(dataset_order) + 2), constrained_layout=True)

for ax, pair in zip(axes, PAIR_ORDER):
    sub = wilcoxon[wilcoxon["comparison"] == pair].copy()
    sub["value_plot"] = -np.log10(sub["wilcoxon_p_fdr"].clip(lower=1e-300))

    pivot_p = sub.pivot(index="dataset", columns="metric", values="value_plot")
    pivot_p = pivot_p.reindex(index=dataset_order, columns=metric_order)

    sig_p = sub.pivot(index="dataset", columns="metric", values="wilcoxon_significant_fdr")
    sig_p = sig_p.reindex(index=dataset_order, columns=metric_order)

    annot_p = pivot_p.copy().astype(object)
    for i in annot_p.index:
        for j in annot_p.columns:
            if pd.isna(pivot_p.loc[i, j]):
                annot_p.loc[i, j] = ""
            else:
                annot_p.loc[i, j] = "*" if bool(sig_p.loc[i, j]) else ""

    sns.heatmap(
        pivot_p,
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        annot=annot_p,
        fmt="",
        cbar=ax is axes[-1],
        cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"} if ax is axes[-1] else None,
        ax=ax
    )
    ax.set_title(pair)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in pivot_p.columns], rotation=45, ha="right")

plt.savefig(os.path.join(OUT_DIR, "wilcoxon_pvalue_heatmaps.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wilcoxon_pvalue_heatmaps.pdf"), dpi=600, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FIGURE C: Wilcoxon effect-size heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(5.2 * 3, 0.55 * len(dataset_order) + 2), constrained_layout=True)

vlim = np.nanmax(np.abs(wilcoxon["rank_biserial"].to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

for ax, pair in zip(axes, PAIR_ORDER):
    sub = wilcoxon[wilcoxon["comparison"] == pair].copy()

    pivot_e = sub.pivot(index="dataset", columns="metric", values="rank_biserial")
    pivot_e = pivot_e.reindex(index=dataset_order, columns=metric_order)

    sns.heatmap(
        pivot_e,
        cmap="coolwarm",
        center=0,
        vmin=-vlim,
        vmax=vlim,
        linewidths=0.5,
        linecolor="white",
        cbar=ax is axes[-1],
        cbar_kws={"label": "Rank-biserial effect size"} if ax is axes[-1] else None,
        ax=ax
    )
    ax.set_title(pair)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in pivot_e.columns], rotation=45, ha="right")

plt.savefig(os.path.join(OUT_DIR, "wilcoxon_effectsize_heatmaps.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wilcoxon_effectsize_heatmaps.pdf"), dpi=600, bbox_inches="tight")
plt.close()

print("Saved plots to:", OUT_DIR)

# In[4]:

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.float = float
np.int = int
np.object = object
np.bool = bool

# --------------------------------------------------
# INPUT
# --------------------------------------------------
FRIEDMAN_CSV = "stats_from_csv/friedman_omnibus_results.csv"
WILCOXON_CSV = "stats_from_csv/wilcoxon_posthoc_results.csv"
OUT_DIR = "stat_plots"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# STYLE
# --------------------------------------------------
sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# dataset order
DATASET_ORDER = [
    "ABIDE", "ADHD", "SCZ", "MEArecs",
    "CaFast_Sham_byDIV",
    "cap_sleep_controls_ALL_SLEEP_NREM",
    "cap_sleep_controls_EEG_NREM",
    "cap_sleep_controls_EMG_NREM",
    "cap_sleep_controls_RESP_NREM",
    "emg_plantar_EMGONLY_per_subject",
    "fantasia_all",
    "nsrdb",
    "resp_aeration_stream_per_subject"
]

# prettier dataset labels
DATASET_LABELS = {
    "ABIDE": "ABIDE",
    "ADHD": "ADHD",
    "SCZ": "SCZ",
    "MEArecs": "MEA",
    "CaFast_Sham_byDIV": "Calcium image",
    "cap_sleep_controls_ALL_SLEEP_NREM": "Sleep-All",
    "cap_sleep_controls_EEG_NREM": "Sleep-EEG",
    "cap_sleep_controls_EMG_NREM": "Sleep-EMG",
    "cap_sleep_controls_RESP_NREM": "Sleep-Resp",
    "emg_plantar_EMGONLY_per_subject": "EMG",
    "fantasia_all": "ECG-Fantasia",
    "nsrdb": "ECG-NSRDB",
    "resp_aeration_stream_per_subject": "Resp"
}

METRIC_ORDER = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
    "zC",
    "zL"
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Transitivity",
    "char_path_len_gcc": "Path length",
    "global_efficiency": "Global efficiency",
    "zC": r"$z_C$",
    "zL": r"$z_L$"
}

PAIR_ORDER = ["QG vs GAF", "QG vs MTF", "GAF vs MTF"]

# --------------------------------------------------
# LOAD
# --------------------------------------------------
friedman = pd.read_csv(FRIEDMAN_CSV)
wilcoxon = pd.read_csv(WILCOXON_CSV)

friedman = friedman[friedman["metric"].isin(METRIC_ORDER)].copy()
wilcoxon = wilcoxon[wilcoxon["metric"].isin(METRIC_ORDER)].copy()

wilcoxon["comparison"] = wilcoxon["comparison"].replace({
    "QTN vs GAF": "QG vs GAF",
    "QTN vs MTF": "QG vs MTF",
    "QG vs GAF": "QG vs GAF",
    "QG vs MTF": "QG vs MTF",
    "GAF vs MTF": "GAF vs MTF"
})

dataset_order = [d for d in DATASET_ORDER if d in friedman["dataset"].unique()]
dataset_order += [d for d in friedman["dataset"].unique() if d not in dataset_order]

metric_order = [m for m in METRIC_ORDER if m in friedman["metric"].unique()]
metric_order += [m for m in friedman["metric"].unique() if m not in metric_order]

dataset_labels_ordered = [DATASET_LABELS.get(d, d) for d in dataset_order]

# --------------------------------------------------
# FIGURE A: Friedman heatmap
# --------------------------------------------------
friedman_plot = friedman.copy()
friedman_plot["value_plot"] = -np.log10(friedman_plot["friedman_p_fdr"].clip(lower=1e-300))

pivot_f = friedman_plot.pivot(index="dataset", columns="metric", values="value_plot")
pivot_f = pivot_f.reindex(index=dataset_order, columns=metric_order)

sig_f = friedman_plot.pivot(index="dataset", columns="metric", values="friedman_significant_fdr")
sig_f = sig_f.reindex(index=dataset_order, columns=metric_order)

annot_f = pivot_f.copy().astype(object)
for i in annot_f.index:
    for j in annot_f.columns:
        if pd.isna(pivot_f.loc[i, j]):
            annot_f.loc[i, j] = ""
        else:
            annot_f.loc[i, j] = "*" if bool(sig_f.loc[i, j]) else ""

plt.figure(figsize=(1.15 * len(metric_order) + 3.8, 0.52 * len(dataset_order) + 2.2))
ax = sns.heatmap(
    pivot_f,
    cmap="viridis",
    linewidths=0.5,
    linecolor="white",
    annot=annot_f,
    fmt="",
    cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"}
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Overall paired differences across representations", pad=12)
ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in pivot_f.columns], rotation=0, ha="center")
ax.set_yticklabels(dataset_labels_ordered, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "friedman_heatmap.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "friedman_heatmap.pdf"), dpi=600, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FIGURE B: Wilcoxon adjusted p-value heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(5.0 * 3, 0.52 * len(dataset_order) + 2.2), constrained_layout=True)

for ax, pair in zip(axes, PAIR_ORDER):
    sub = wilcoxon[wilcoxon["comparison"] == pair].copy()
    sub["value_plot"] = -np.log10(sub["wilcoxon_p_fdr"].clip(lower=1e-300))

    pivot_p = sub.pivot(index="dataset", columns="metric", values="value_plot")
    pivot_p = pivot_p.reindex(index=dataset_order, columns=metric_order)

    sig_p = sub.pivot(index="dataset", columns="metric", values="wilcoxon_significant_fdr")
    sig_p = sig_p.reindex(index=dataset_order, columns=metric_order)

    annot_p = pivot_p.copy().astype(object)
    for i in annot_p.index:
        for j in annot_p.columns:
            if pd.isna(pivot_p.loc[i, j]):
                annot_p.loc[i, j] = ""
            else:
                annot_p.loc[i, j] = "*" if bool(sig_p.loc[i, j]) else ""

    sns.heatmap(
        pivot_p,
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        annot=annot_p,
        fmt="",
        cbar=ax is axes[-1],
        cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"} if ax is axes[-1] else None,
        ax=ax
    )
    ax.set_title(pair, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in pivot_p.columns], rotation=0, ha="center")
    ax.set_yticklabels(dataset_labels_ordered, rotation=0)

plt.savefig(os.path.join(OUT_DIR, "wilcoxon_pvalue_heatmaps.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wilcoxon_pvalue_heatmaps.pdf"), dpi=600, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FIGURE C: Wilcoxon effect-size heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(5.0 * 3, 0.52 * len(dataset_order) + 2.2), constrained_layout=True)

vlim = np.nanmax(np.abs(wilcoxon["rank_biserial"].to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

for ax, pair in zip(axes, PAIR_ORDER):
    sub = wilcoxon[wilcoxon["comparison"] == pair].copy()

    pivot_e = sub.pivot(index="dataset", columns="metric", values="rank_biserial")
    pivot_e = pivot_e.reindex(index=dataset_order, columns=metric_order)

    sns.heatmap(
        pivot_e,
        cmap="coolwarm",
        center=0,
        vmin=-vlim,
        vmax=vlim,
        linewidths=0.5,
        linecolor="white",
        cbar=ax is axes[-1],
        cbar_kws={"label": "Rank-biserial effect size"} if ax is axes[-1] else None,
        ax=ax
    )
    ax.set_title(pair, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in pivot_e.columns], rotation=0, ha="center")
    ax.set_yticklabels(dataset_labels_ordered, rotation=0)

plt.savefig(os.path.join(OUT_DIR, "wilcoxon_effectsize_heatmaps.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wilcoxon_effectsize_heatmaps.pdf"), dpi=600, bbox_inches="tight")
plt.close()

print("Saved plots to:", OUT_DIR)

# In[5]:

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# compatibility for older packages
np.float = float
np.int = int
np.object = object
np.bool = bool

# --------------------------------------------------
# INPUT
# --------------------------------------------------
FRIEDMAN_CSV = "stats_from_csv/friedman_omnibus_results.csv"
WILCOXON_CSV = "stats_from_csv/wilcoxon_posthoc_results.csv"
OUT_DIR = "stat_plots"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# STYLE
# --------------------------------------------------
sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# --------------------------------------------------
# ORDER / LABELS
# --------------------------------------------------
DATASET_ORDER = [
    "ABIDE", "ADHD", "SCZ", "MEArecs",
    "CaFast_Sham_byDIV",
    "cap_sleep_controls_ALL_SLEEP_NREM",
    "cap_sleep_controls_EEG_NREM",
    "cap_sleep_controls_EMG_NREM",
    "cap_sleep_controls_RESP_NREM",
    "emg_plantar_EMGONLY_per_subject",
    "fantasia_all",
    "nsrdb",
    "resp_aeration_stream_per_subject"
]

DATASET_LABELS = {
    "ABIDE": "ABIDE",
    "ADHD": "ADHD",
    "SCZ": "SCZ",
    "MEArecs": "MEA",
    "CaFast_Sham_byDIV": "Calcium",
    "cap_sleep_controls_ALL_SLEEP_NREM": "Sleep-All",
    "cap_sleep_controls_EEG_NREM": "Sleep-EEG",
    "cap_sleep_controls_EMG_NREM": "Sleep-EMG",
    "cap_sleep_controls_RESP_NREM": "Sleep-Resp",
    "emg_plantar_EMGONLY_per_subject": "EMG",
    "fantasia_all": "ECG-Fantasia",
    "nsrdb": "ECG-NSRDB",
    "resp_aeration_stream_per_subject": "Resp"
}

METRIC_ORDER = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
    "zC",
    "zL"
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
    "zC": r"$z_C$",
    "zL": r"$z_L$"
}

PAIR_ORDER = ["QG vs GAF", "QG vs MTF", "GAF vs MTF"]

# --------------------------------------------------
# LOAD
# --------------------------------------------------
friedman = pd.read_csv(FRIEDMAN_CSV)
wilcoxon = pd.read_csv(WILCOXON_CSV)

# keep only metrics of interest
friedman = friedman[friedman["metric"].isin(METRIC_ORDER)].copy()
wilcoxon = wilcoxon[wilcoxon["metric"].isin(METRIC_ORDER)].copy()

# rename comparisons if needed
wilcoxon["comparison"] = wilcoxon["comparison"].replace({
    "QTN vs GAF": "QG vs GAF",
    "QTN vs MTF": "QG vs MTF",
    "QG vs GAF": "QG vs GAF",
    "QG vs MTF": "QG vs MTF",
    "GAF vs MTF": "GAF vs MTF"
})

# keep only known ordering if present
dataset_order = [d for d in DATASET_ORDER if d in friedman["dataset"].unique()]
dataset_order += [d for d in friedman["dataset"].unique() if d not in dataset_order]

metric_order = [m for m in METRIC_ORDER if m in friedman["metric"].unique()]
metric_order += [m for m in friedman["metric"].unique() if m not in metric_order]

dataset_labels_ordered = [DATASET_LABELS.get(d, d) for d in dataset_order]

# --------------------------------------------------
# FIGURE A: Friedman heatmap
# --------------------------------------------------
friedman_plot = friedman.copy()
friedman_plot["value_plot"] = -np.log10(friedman_plot["friedman_p_fdr"].clip(lower=1e-300))

pivot_f = friedman_plot.pivot(index="dataset", columns="metric", values="value_plot")
pivot_f = pivot_f.reindex(index=dataset_order, columns=metric_order)

sig_f = friedman_plot.pivot(index="dataset", columns="metric", values="friedman_significant_fdr")
sig_f = sig_f.reindex(index=dataset_order, columns=metric_order)

annot_f = pivot_f.copy().astype(object)
for i in annot_f.index:
    for j in annot_f.columns:
        if pd.isna(pivot_f.loc[i, j]):
            annot_f.loc[i, j] = ""
        else:
            annot_f.loc[i, j] = "*" if bool(sig_f.loc[i, j]) else ""

plt.figure(figsize=(13, 9))
ax = sns.heatmap(
    pivot_f,
    cmap="viridis",
    linewidths=0.5,
    linecolor="white",
    annot=annot_f,
    fmt="",
    cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"}
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Overall paired differences across representations", pad=12)

ax.set_xticklabels(
    [METRIC_LABELS.get(m, m) for m in pivot_f.columns],
    rotation=25,
    ha="right",
    rotation_mode="anchor",
    fontsize=12
)
ax.set_yticklabels(
    dataset_labels_ordered,
    rotation=0,
    fontsize=13
)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "friedman_heatmap.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "friedman_heatmap.pdf"), dpi=600, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FIGURE B: Wilcoxon adjusted p-value heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 9),
    constrained_layout=True
)

for ax, pair in zip(axes, PAIR_ORDER):
    sub = wilcoxon[wilcoxon["comparison"] == pair].copy()
    sub["value_plot"] = -np.log10(sub["wilcoxon_p_fdr"].clip(lower=1e-300))

    pivot_p = sub.pivot(index="dataset", columns="metric", values="value_plot")
    pivot_p = pivot_p.reindex(index=dataset_order, columns=metric_order)

    sig_p = sub.pivot(index="dataset", columns="metric", values="wilcoxon_significant_fdr")
    sig_p = sig_p.reindex(index=dataset_order, columns=metric_order)

    annot_p = pivot_p.copy().astype(object)
    for i in annot_p.index:
        for j in annot_p.columns:
            if pd.isna(pivot_p.loc[i, j]):
                annot_p.loc[i, j] = ""
            else:
                annot_p.loc[i, j] = "*" if bool(sig_p.loc[i, j]) else ""

    sns.heatmap(
        pivot_p,
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        annot=annot_p,
        fmt="",
        cbar=ax is axes[-1],
        cbar_kws={"label": r"$-\log_{10}(\mathrm{FDR}\ p)$"} if ax is axes[-1] else None,
        ax=ax
    )
    ax.set_title(pair, pad=10, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in pivot_p.columns],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
        fontsize=11
    )
    ax.set_yticklabels(
        dataset_labels_ordered,
        rotation=0,
        fontsize=13
    )

plt.savefig(os.path.join(OUT_DIR, "wilcoxon_pvalue_heatmaps.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wilcoxon_pvalue_heatmaps.pdf"), dpi=600, bbox_inches="tight")
plt.close()

# --------------------------------------------------
# FIGURE C: Wilcoxon effect-size heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 9),
    constrained_layout=True
)

vlim = np.nanmax(np.abs(wilcoxon["rank_biserial"].to_numpy()))
if not np.isfinite(vlim) or vlim == 0:
    vlim = 1.0

for ax, pair in zip(axes, PAIR_ORDER):
    sub = wilcoxon[wilcoxon["comparison"] == pair].copy()

    pivot_e = sub.pivot(index="dataset", columns="metric", values="rank_biserial")
    pivot_e = pivot_e.reindex(index=dataset_order, columns=metric_order)

    sns.heatmap(
        pivot_e,
        cmap="coolwarm",
        center=0,
        vmin=-vlim,
        vmax=vlim,
        linewidths=0.5,
        linecolor="white",
        cbar=ax is axes[-1],
        cbar_kws={"label": "Rank-biserial effect size"} if ax is axes[-1] else None,
        ax=ax
    )

    ax.set_title(pair, pad=10, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticklabels(
        [METRIC_LABELS.get(m, m) for m in pivot_e.columns],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
        fontsize=11
    )
    ax.set_yticklabels(
        dataset_labels_ordered,
        rotation=0,
        fontsize=13
    )

plt.savefig(os.path.join(OUT_DIR, "wilcoxon_effectsize_heatmaps.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "wilcoxon_effectsize_heatmaps.pdf"), dpi=600, bbox_inches="tight")
plt.close()

print("Saved plots to:", OUT_DIR)

# In[ ]:
