#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``similarity .ipynb``.

Code cells are reproduced verbatim. Markdown cells are kept as comments.
Jupyter shell-magic lines (if any) are commented out so the file is valid
Python; they are preserved for reference. Cell boundaries are marked with
``# In[...]`` to match the original notebook ordering.
"""


# In[16]:

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist, squareform
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

# ---------------------------------------
# compatibility for older packages
# ---------------------------------------
np.float = float
np.int = int
np.object = object
np.bool = bool

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_DIR = "csv"
OUT_DIR = "mtf_clustered_similarity_triangle"

METRICS_TO_USE = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
    "zC",
    "zL",
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
    "ds000117_raw_meg_per_subject": "MEG",
    "emg_plantar_EMGONLY_per_subject": "EMG",
    "fantasia_all": "ECG-Fantasia",
    "nsrdb": "ECG-NSRDB",
    "resp_aeration_stream_per_subject": "Resp",
}

PRETTY_COLS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
    "zC": r"$z_C$",
    "zL": r"$z_L$",
}

SUMMARY_STAT = "median"   # or "mean"
FIG_DPI = 600

sns.set(style="white", context="talk")
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": False
})

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_rep_dataset(filepath: str):
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    rep = m.group(1).upper()
    dataset = m.group(2)
    return rep, dataset

def prettify_dataset_name(name: str) -> str:
    return DATASET_LABELS.get(name, name)

def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = pd.to_numeric(out[col], errors="coerce")
        mu = vals.mean()
        sd = vals.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            out[col] = 0.0
        else:
            out[col] = (vals - mu) / sd
    return out

def summarize_dataset(df: pd.DataFrame, metrics, summary_stat="median"):
    row = {}
    for metric in metrics:
        if metric not in df.columns:
            row[metric] = np.nan
            continue

        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) == 0:
            row[metric] = np.nan
        elif summary_stat == "mean":
            row[metric] = vals.mean()
        else:
            row[metric] = vals.median()
    return row

# --------------------------------------------------
# LOAD MTF FILES
# --------------------------------------------------
def build_mtf_profile_table(input_dir, metrics, summary_stat="median"):
    input_dir = Path(input_dir)
    all_files = sorted(glob.glob(str(input_dir / "metrics_*.csv")))

    rows = []
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue

        rep, dataset = parsed
        if rep != "MTF":
            continue

        df = pd.read_csv(f)
        summary = summarize_dataset(df, metrics, summary_stat=summary_stat)
        summary["dataset"] = dataset
        summary["dataset_label"] = prettify_dataset_name(dataset)
        rows.append(summary)

    profile_df = pd.DataFrame(rows)
    if profile_df.empty:
        return profile_df

    cols = ["dataset", "dataset_label"] + metrics
    profile_df = profile_df[cols]
    profile_df = profile_df.drop_duplicates(subset=["dataset"]).reset_index(drop=True)
    return profile_df

# --------------------------------------------------
# SIMILARITY
# --------------------------------------------------
def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = np.nan
    Xn = X / norms
    sim = np.dot(Xn, Xn.T)
    return sim

def build_similarity_from_profiles(profile_df: pd.DataFrame):
    df = profile_df.set_index("dataset_label")[METRICS_TO_USE].copy()
    df = df.dropna(axis=1, thresh=2)
    df = df.dropna(axis=0, how="any")
    df_z = zscore_columns(df)

    X = df_z.to_numpy(dtype=float)
    sim = cosine_similarity_matrix(X)
    sim_df = pd.DataFrame(sim, index=df_z.index.tolist(), columns=df_z.index.tolist())

    return df_z, sim_df

# --------------------------------------------------
# TRIANGULAR CLUSTERED SIMILARITY HEATMAP
# --------------------------------------------------
def plot_triangular_clustered_similarity(sim_df: pd.DataFrame, out_dir: str, summary_stat="median"):
    os.makedirs(out_dir, exist_ok=True)

    dist_mat = 1 - sim_df.values
    np.fill_diagonal(dist_mat, 0.0)
    dist_condensed = squareform(dist_mat, checks=False)

    Z = linkage(dist_condensed, method="average")
    order = leaves_list(Z)

    sim_ord = sim_df.iloc[order, order]
    labels = sim_ord.index.tolist()

    sim_csv = os.path.join(out_dir, f"mtf_similarity_ordered_{summary_stat}.csv")
    sim_ord.to_csv(sim_csv)

    # lower triangle only
    mask = np.triu(np.ones_like(sim_ord, dtype=bool), k=1)

    fig = plt.figure(figsize=(12, 11))

    # Manual placement
    ax_cbar = fig.add_axes([0.05, 0.86, 0.22, 0.035])  # more in white space
    ax_row  = fig.add_axes([0.04, 0.12, 0.24, 0.68])
    ax_heat = fig.add_axes([0.31, 0.12, 0.62, 0.68])

    # Dendrogram
    dendrogram(
        Z,
        ax=ax_row,
        orientation="left",
        no_labels=True,
        color_threshold=None,
        above_threshold_color="black"
    )
    ax_row.set_xticks([])
    ax_row.set_yticks([])
    ax_row.set_ylabel("")
    ax_row.set_xlabel("")
    ax_row.tick_params(axis="both", left=False, right=False, labelleft=False, labelbottom=False)
    for spine in ax_row.spines.values():
        spine.set_visible(False)

    # Heatmap
    sns.heatmap(
        sim_ord,
        mask=mask,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax_heat
    )

    ax_heat.set_xlabel("Dataset", fontsize=14, labelpad=12)
    ax_heat.set_ylabel("")

    ax_heat.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor", fontsize=10)

    # remove all vertical dataset names on the left
    ax_heat.set_yticks([])
    ax_heat.set_yticklabels([])
    ax_heat.tick_params(axis="y", left=False, right=False, length=0)

    # Colorbar
    norm = Normalize(vmin=-1, vmax=1)
    cb = ColorbarBase(
        ax_cbar,
        cmap=plt.get_cmap("viridis"),
        norm=norm,
        orientation="horizontal"
    )
    ax_cbar.set_title("Similarity", fontsize=11, pad=8)
    ax_cbar.tick_params(labelsize=9)
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)

    png_path = os.path.join(out_dir, f"mtf_similarity_triangle_clean_{summary_stat}.png")
    pdf_path = os.path.join(out_dir, f"mtf_similarity_triangle_clean_{summary_stat}.pdf")

    plt.savefig(png_path, dpi=FIG_DPI)
    plt.savefig(pdf_path, dpi=FIG_DPI)
    plt.close()

    print("Saved CLEAN triangular clustered similarity heatmap:")
    print(sim_csv)
    print(png_path)
    print(pdf_path)

# --------------------------------------------------
# CLUSTERED METRIC PROFILE HEATMAP
# --------------------------------------------------
def plot_clustered_metric_profiles(profile_df: pd.DataFrame, out_dir: str, summary_stat="median"):
    os.makedirs(out_dir, exist_ok=True)

    df = profile_df.set_index("dataset_label")[METRICS_TO_USE].copy()
    df = df.dropna(axis=1, thresh=2)
    df = df.dropna(axis=0, how="any")
    df_z = zscore_columns(df)

    row_link = linkage(pdist(df_z.values, metric="euclidean"), method="ward")
    col_link = linkage(pdist(df_z.values.T, metric="euclidean"), method="ward")

    row_order = leaves_list(row_link)
    col_order = leaves_list(col_link)

    df_ord = df_z.iloc[row_order, col_order].rename(columns=PRETTY_COLS)

    csv_path = os.path.join(out_dir, f"mtf_profiles_zscored_ordered_{summary_stat}.csv")
    df_ord.to_csv(csv_path)

    fig = plt.figure(figsize=(11, 9))

    ax_cbar = fig.add_axes([0.05, 0.86, 0.22, 0.035])
    ax_row  = fig.add_axes([0.04, 0.12, 0.24, 0.68])
    ax_heat = fig.add_axes([0.31, 0.12, 0.62, 0.68])

    dendrogram(
        row_link,
        ax=ax_row,
        orientation="left",
        no_labels=True,
        color_threshold=None,
        above_threshold_color="black"
    )
    ax_row.set_xticks([])
    ax_row.set_yticks([])
    ax_row.set_ylabel("")
    ax_row.set_xlabel("")
    ax_row.tick_params(axis="both", left=False, right=False, labelleft=False, labelbottom=False)
    for spine in ax_row.spines.values():
        spine.set_visible(False)

    sns.heatmap(
        df_ord,
        cmap="vlag",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar=False,
        ax=ax_heat
    )

    ax_heat.set_xlabel("Complex network metrics", fontsize=14, labelpad=12)
    ax_heat.set_ylabel("")
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor", fontsize=10)

    # remove all vertical dataset names on the left
    ax_heat.set_yticks([])
    ax_heat.set_yticklabels([])
    ax_heat.tick_params(axis="y", left=False, right=False, length=0)

    norm = Normalize(vmin=np.nanmin(df_ord.values), vmax=np.nanmax(df_ord.values))
    cb = ColorbarBase(
        ax_cbar,
        cmap=plt.get_cmap("vlag"),
        norm=norm,
        orientation="horizontal"
    )
    ax_cbar.set_title("Z-score", fontsize=11, pad=8)
    ax_cbar.tick_params(labelsize=9)
    for spine in ax_cbar.spines.values():
        spine.set_visible(False)

    png_path = os.path.join(out_dir, f"mtf_metric_profiles_clustered_{summary_stat}.png")
    pdf_path = os.path.join(out_dir, f"mtf_metric_profiles_clustered_{summary_stat}.pdf")

    plt.savefig(png_path, dpi=FIG_DPI)
    plt.savefig(pdf_path, dpi=FIG_DPI)
    plt.close()

    print("Saved clustered metric profile heatmap:")
    print(csv_path)
    print(png_path)
    print(pdf_path)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    profile_df = build_mtf_profile_table(
        INPUT_DIR,
        metrics=METRICS_TO_USE,
        summary_stat=SUMMARY_STAT
    )

    if profile_df.empty:
        raise RuntimeError("No MTF files found or no valid metrics available.")

    raw_profile_csv = os.path.join(
        OUT_DIR, f"mtf_profiles_raw_with_dataset_column_{SUMMARY_STAT}.csv"
    )
    profile_df.to_csv(raw_profile_csv, index=False)

    df_z, sim_df = build_similarity_from_profiles(profile_df)

    z_csv = os.path.join(OUT_DIR, f"mtf_profiles_zscored_{SUMMARY_STAT}.csv")
    sim_csv = os.path.join(OUT_DIR, f"mtf_dataset_cosine_similarity_{SUMMARY_STAT}.csv")
    df_z.to_csv(z_csv)
    sim_df.to_csv(sim_csv)

    plot_triangular_clustered_similarity(sim_df, OUT_DIR, summary_stat=SUMMARY_STAT)
    plot_clustered_metric_profiles(profile_df, OUT_DIR, summary_stat=SUMMARY_STAT)

    print("Saved:")
    print(raw_profile_csv)
    print(z_csv)
    print(sim_csv)

if __name__ == "__main__":
    main()

# In[ ]:
