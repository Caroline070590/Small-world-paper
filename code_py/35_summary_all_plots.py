#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``Summary-all-plots.ipynb``.

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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# compatibility for older packages
np.float = float
np.int = int
np.object = object
np.bool = bool

# ==========================================================
# CONFIG
# ==========================================================
INPUT_DIR = "csv"
OUT_DIR = "representation_heatmaps"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

DATASET_ORDER = [
    "ABIDE",
    "ADHD",
    "SCZ",
    "MEArecs",
    "CaFast_Sham_byDIV",
    "cap_sleep_controls_ALL_SLEEP_NREM",
    "cap_sleep_controls_EEG_NREM",
    "cap_sleep_controls_EMG_NREM",
    "cap_sleep_controls_RESP_NREM",
    "ds000117_raw_meg_per_subject",
    "emg_plantar_EMGONLY_per_subject",
    "fantasia_all",
    "nsrdb",
    "resp_aeration_stream_per_subject",
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

# use medians per dataset; can switch to "mean"
SUMMARY_STAT = "median"   # or "mean"

# different palette from the statistical test plots
RAW_CMAP = "YlGnBu"
ZSCORE_CMAP = "PuOr_r"

DPI = 600

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
def extract_rep_dataset(filepath: str):
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper(), m.group(2)

def normalize_key(x):
    return str(x).strip()

def get_id_series(df: pd.DataFrame, rep: str) -> pd.Series:
    id_candidates = [
        "patient_id",
        "subject_id",
        "file_id",
        "simulation_id",
        "div",
        "record_id",
        "sample_id",
        "id",
    ]
    for col in id_candidates:
        if col in df.columns:
            return df[col].astype(str).map(normalize_key)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        return df[object_cols[0]].astype(str).map(normalize_key)

    return pd.Series([f"{rep}_{i}" for i in range(len(df))], name="auto_id")

def summarize_dataset_representation(path: str, metric_order: list, summary_stat: str = "median") -> dict:
    df = pd.read_csv(path)

    summary = {}
    for metric in metric_order:
        if metric not in df.columns:
            summary[metric] = np.nan
            continue

        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) == 0:
            summary[metric] = np.nan
        else:
            if summary_stat == "median":
                summary[metric] = float(np.median(vals))
            elif summary_stat == "mean":
                summary[metric] = float(np.mean(vals))
            else:
                raise ValueError("summary_stat must be 'median' or 'mean'")

    return summary

def build_rep_tables(input_dir: str):
    all_files = sorted(glob.glob(os.path.join(input_dir, "metrics_*.csv")))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir}")

    grouped = {}
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue
        rep, dataset = parsed
        grouped.setdefault(dataset, {})
        grouped[dataset][rep] = f

    rep_tables = {rep: [] for rep in REP_ORDER}

    for dataset in grouped:
        for rep in REP_ORDER:
            if rep not in grouped[dataset]:
                continue

            summary = summarize_dataset_representation(
                grouped[dataset][rep],
                metric_order=METRIC_ORDER,
                summary_stat=SUMMARY_STAT
            )

            row = {"dataset": dataset}
            row.update(summary)
            rep_tables[rep].append(row)

    for rep in REP_ORDER:
        rep_tables[rep] = pd.DataFrame(rep_tables[rep])

    return grouped, rep_tables

def reorder_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    dataset_order = [d for d in DATASET_ORDER if d in df["dataset"].tolist()]
    dataset_order += [d for d in df["dataset"].tolist() if d not in dataset_order]

    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values("dataset")

    cols = ["dataset"] + [m for m in METRIC_ORDER if m in df.columns]
    return df[cols]

def zscore_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for metric in METRIC_ORDER:
        if metric in out.columns:
            vals = pd.to_numeric(out[metric], errors="coerce")
            mu = vals.mean()
            sd = vals.std(ddof=0)
            if pd.isna(sd) or sd == 0:
                out[metric] = np.nan
            else:
                out[metric] = (vals - mu) / sd
    return out

def dataset_labels_for(df: pd.DataFrame):
    return [DATASET_LABELS.get(str(d), str(d)) for d in df["dataset"].tolist()]

def metric_labels_for(df: pd.DataFrame):
    return [METRIC_LABELS.get(c, c) for c in df.columns if c != "dataset"]

def draw_three_heatmaps(rep_tables: dict, out_dir: str, zscore: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    processed = {}
    for rep in REP_ORDER:
        df = rep_tables.get(rep, pd.DataFrame()).copy()
        df = reorder_table(df)
        if zscore:
            df = zscore_by_metric(df)
        processed[rep] = df

    # collect global min/max for comparable scales across the three panels
    all_vals = []
    for rep in REP_ORDER:
        df = processed[rep]
        if not df.empty:
            vals = df.drop(columns=["dataset"]).to_numpy(dtype=float)
            all_vals.append(vals.flatten())

    all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0])
    all_vals = all_vals[np.isfinite(all_vals)]

    if len(all_vals) == 0:
        vmin, vmax = -1, 1
    else:
        if zscore:
            vabs = max(abs(np.nanmin(all_vals)), abs(np.nanmax(all_vals)))
            vmin, vmax = -vabs, vabs
        else:
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    cmap = ZSCORE_CMAP if zscore else RAW_CMAP

    fig, axes = plt.subplots(1, 3, figsize=(18, 9), constrained_layout=True)

    for ax, rep in zip(axes, REP_ORDER):
        df = processed[rep]

        if df.empty:
            ax.axis("off")
            ax.set_title(f"{REP_DISPLAY[rep]} (missing)")
            continue

        heat = df.set_index("dataset")
        heat.index = dataset_labels_for(df)

        sns.heatmap(
            heat,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={
                "label": "Z-scored value" if zscore else f"{SUMMARY_STAT.capitalize()} value"
            } if ax is axes[-1] else None,
            center=0 if zscore else None,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        ax.set_title(REP_DISPLAY[rep], pad=10, fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.set_xticklabels(
            metric_labels_for(df),
            rotation=25,
            ha="right",
            rotation_mode="anchor",
            fontsize=11
        )
        ax.set_yticklabels(
            heat.index.tolist(),
            rotation=0,
            fontsize=12
        )

    suffix = "zscore" if zscore else "raw"
    png = os.path.join(out_dir, f"three_heatmaps_{suffix}.png")
    pdf = os.path.join(out_dir, f"three_heatmaps_{suffix}.pdf")

    plt.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.savefig(pdf, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"[OK] wrote: {png}")
    print(f"[OK] wrote: {pdf}")

# ==========================================================
# MAIN
# ==========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    grouped, rep_tables = build_rep_tables(INPUT_DIR)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    # Save summary tables too
    for rep in REP_ORDER:
        df = reorder_table(rep_tables[rep])
        if not df.empty:
            df.to_csv(
                os.path.join(OUT_DIR, f"summary_{REP_DISPLAY[rep]}_{SUMMARY_STAT}.csv"),
                index=False
            )

    # Raw median/mean heatmaps
    draw_three_heatmaps(rep_tables, OUT_DIR, zscore=False)

    # Z-scored heatmaps
    draw_three_heatmaps(rep_tables, OUT_DIR, zscore=True)

    print(f"\n[DONE] All outputs saved in: {Path(OUT_DIR).resolve()}")

if __name__ == "__main__":
    main()

# In[4]:

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# compatibility for older packages
np.float = float
np.int = int
np.object = object
np.bool = bool

# ==========================================================
# CONFIG
# ==========================================================
INPUT_DIR = "csv"
OUT_DIR = "representation_heatmaps"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {
    "QTN": "QG",
    "GAF": "GAF",
    "MTF": "MTF"
}

DATASET_ORDER = [
    "ABIDE",
    "ADHD",
    "SCZ",
    "MEArecs",
    "CaFast_Sham_byDIV",
    "cap_sleep_controls_ALL_SLEEP_NREM",
    "cap_sleep_controls_EEG_NREM",
    "cap_sleep_controls_EMG_NREM",
    "cap_sleep_controls_RESP_NREM",
    "ds000117_raw_meg_per_subject",
    "emg_plantar_EMGONLY_per_subject",
    "fantasia_all",
    "nsrdb",
    "resp_aeration_stream_per_subject",
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

MAIN_METRICS = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
]

Z_METRICS = ["zC", "zL"]

METRIC_LABELS = {
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
DPI = 600

MAIN_CMAP = "YlGnBu"
Z_CMAP = "PuOr_r"

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
def extract_rep_dataset(filepath: str):
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper(), m.group(2)

def summarize_dataset_representation(path: str, metric_order: list, summary_stat: str = "median") -> dict:
    df = pd.read_csv(path)

    summary = {}
    for metric in metric_order:
        if metric not in df.columns:
            summary[metric] = np.nan
            continue

        vals = pd.to_numeric(df[metric], errors="coerce").dropna()
        if len(vals) == 0:
            summary[metric] = np.nan
        else:
            if summary_stat == "median":
                summary[metric] = float(np.median(vals))
            elif summary_stat == "mean":
                summary[metric] = float(np.mean(vals))
            else:
                raise ValueError("summary_stat must be 'median' or 'mean'")

    return summary

def build_rep_tables(input_dir: str, metric_order: list):
    all_files = sorted(glob.glob(os.path.join(input_dir, "metrics_*.csv")))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir}")

    grouped = {}
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue
        rep, dataset = parsed
        grouped.setdefault(dataset, {})
        grouped[dataset][rep] = f

    rep_tables = {rep: [] for rep in REP_ORDER}

    for dataset in grouped:
        for rep in REP_ORDER:
            if rep not in grouped[dataset]:
                continue

            summary = summarize_dataset_representation(
                grouped[dataset][rep],
                metric_order=metric_order,
                summary_stat=SUMMARY_STAT
            )

            row = {"dataset": dataset}
            row.update(summary)
            rep_tables[rep].append(row)

    for rep in REP_ORDER:
        rep_tables[rep] = pd.DataFrame(rep_tables[rep])

    return grouped, rep_tables

def reorder_table(df: pd.DataFrame, metric_order: list) -> pd.DataFrame:
    if df.empty:
        return df

    dataset_order = [d for d in DATASET_ORDER if d in df["dataset"].tolist()]
    dataset_order += [d for d in df["dataset"].tolist() if d not in dataset_order]

    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)
    df = df.sort_values("dataset")

    cols = ["dataset"] + [m for m in metric_order if m in df.columns]
    return df[cols]

def dataset_labels_for(df: pd.DataFrame):
    return [DATASET_LABELS.get(str(d), str(d)) for d in df["dataset"].tolist()]

def metric_labels_for(metric_order):
    return [METRIC_LABELS.get(c, c) for c in metric_order]

def build_annotation_table(df: pd.DataFrame, metric_order: list, sci_for_large: bool = False) -> pd.DataFrame:
    ann = df.copy()
    for metric in metric_order:
        if metric in ann.columns:
            vals = pd.to_numeric(ann[metric], errors="coerce")
            formatted = []
            for v in vals:
                if pd.isna(v):
                    formatted.append("")
                else:
                    if sci_for_large and abs(v) >= 1e4:
                        formatted.append(f"{v:.2e}")
                    else:
                        formatted.append(f"{v:.2f}")
            ann[metric] = formatted
    return ann

# ==========================================================
# MAIN METRIC HEATMAPS (3 side by side)
# ==========================================================
def draw_main_metrics_three_heatmaps(rep_tables: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    processed = {}
    for rep in REP_ORDER:
        df = rep_tables.get(rep, pd.DataFrame()).copy()
        df = reorder_table(df, MAIN_METRICS)
        processed[rep] = df

    # shared raw range across the 3 panels
    all_vals = []
    for rep in REP_ORDER:
        df = processed[rep]
        if not df.empty:
            vals = df.drop(columns=["dataset"]).to_numpy(dtype=float).flatten()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                all_vals.append(vals)

    all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0])
    vmin = np.nanmin(all_vals)
    vmax = np.nanmax(all_vals)

    fig, axes = plt.subplots(1, 3, figsize=(17, 9), constrained_layout=True)

    for ax, rep in zip(axes, REP_ORDER):
        df = processed[rep]
        if df.empty:
            ax.axis("off")
            continue

        heat = df.set_index("dataset")
        heat.index = dataset_labels_for(df)

        ann = build_annotation_table(df, MAIN_METRICS, sci_for_large=False).set_index("dataset")
        ann.index = heat.index

        sns.heatmap(
            heat,
            cmap=MAIN_CMAP,
            linewidths=0.5,
            linecolor="white",
            annot=ann,
            fmt="",
            annot_kws={"fontsize": 9},
            cbar=ax is axes[-1],
            cbar_kws={"label": f"{SUMMARY_STAT.capitalize()} value"} if ax is axes[-1] else None,
            vmin=vmin,
            vmax=vmax,
            ax=ax
        )

        ax.set_title(REP_DISPLAY[rep], fontsize=16, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(metric_labels_for(MAIN_METRICS), rotation=25, ha="right", rotation_mode="anchor", fontsize=11)
        ax.set_yticklabels(heat.index.tolist(), rotation=0, fontsize=12)

    plt.savefig(os.path.join(out_dir, "main_metrics_three_heatmaps.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "main_metrics_three_heatmaps.pdf"), dpi=DPI, bbox_inches="tight")
    plt.close()

    print("[OK] wrote main metrics heatmaps")

# ==========================================================
# zC / zL HEATMAPS (3x2) with asinh color compression
# ==========================================================
def asinh_transform(values: np.ndarray, scale: float):
    return np.arcsinh(values / scale)

def draw_zc_zl_heatmaps_3x2(rep_tables: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # reorder per rep
    processed = {}
    for rep in REP_ORDER:
        df = rep_tables.get(rep, pd.DataFrame()).copy()
        df = reorder_table(df, Z_METRICS)
        processed[rep] = df

    # choose robust scale per metric using medians across all reps/datasets
    metric_scales = {}
    metric_vlims = {}

    for metric in Z_METRICS:
        vals = []
        for rep in REP_ORDER:
            df = processed[rep]
            if not df.empty and metric in df.columns:
                x = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                if len(x) > 0:
                    vals.append(x)

        if len(vals) == 0:
            metric_scales[metric] = 1.0
            metric_vlims[metric] = 1.0
            continue

        vals = np.concatenate(vals)

        # robust scale: median absolute value or percentile-based fallback
        scale = np.nanmedian(np.abs(vals))
        if not np.isfinite(scale) or scale == 0:
            scale = np.nanpercentile(np.abs(vals), 75)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        transformed = asinh_transform(vals, scale)
        vlim = np.nanmax(np.abs(transformed))
        if not np.isfinite(vlim) or vlim == 0:
            vlim = 1.0

        metric_scales[metric] = scale
        metric_vlims[metric] = vlim

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)

    for i, metric in enumerate(Z_METRICS):
        scale = metric_scales[metric]
        vlim = metric_vlims[metric]

        for j, rep in enumerate(REP_ORDER):
            ax = axes[i, j]
            df = processed[rep]

            if df.empty or metric not in df.columns:
                ax.axis("off")
                continue

            heat_raw = df[["dataset", metric]].copy()
            heat_raw = heat_raw.set_index("dataset")
            heat_raw.index = dataset_labels_for(df)

            heat_color = heat_raw.copy()
            heat_color[metric] = asinh_transform(
                pd.to_numeric(heat_color[metric], errors="coerce").to_numpy(dtype=float),
                scale
            )

            ann = build_annotation_table(df[["dataset", metric]].copy(), [metric], sci_for_large=True).set_index("dataset")
            ann.index = heat_raw.index

            sns.heatmap(
                heat_color,
                cmap=Z_CMAP,
                center=0,
                vmin=-vlim,
                vmax=vlim,
                linewidths=0.5,
                linecolor="white",
                annot=ann,
                fmt="",
                annot_kws={"fontsize": 9},
                cbar=(j == 2),
                cbar_kws={"label": f"{METRIC_LABELS[metric]} (asinh-scaled color)"} if j == 2 else None,
                ax=ax
            )

            if i == 0:
                ax.set_title(REP_DISPLAY[rep], fontsize=15, pad=10)

            ax.set_xlabel("")
            ax.set_ylabel(METRIC_LABELS[metric] if j == 0 else "")
            ax.set_xticklabels([METRIC_LABELS[metric]], rotation=0, fontsize=11)
            ax.set_yticklabels(heat_raw.index.tolist(), rotation=0, fontsize=11)

    plt.savefig(os.path.join(out_dir, "zc_zl_heatmaps_3x2_asinh.png"), dpi=DPI, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "zc_zl_heatmaps_3x2_asinh.pdf"), dpi=DPI, bbox_inches="tight")
    plt.close()

    print("[OK] wrote zC/zL heatmaps")

# ==========================================================
# MAIN
# ==========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    grouped, rep_tables_main = build_rep_tables(INPUT_DIR, MAIN_METRICS)
    _, rep_tables_z = build_rep_tables(INPUT_DIR, Z_METRICS)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    # save summary tables
    for rep in REP_ORDER:
        df_main = reorder_table(rep_tables_main[rep], MAIN_METRICS)
        if not df_main.empty:
            df_main.to_csv(os.path.join(OUT_DIR, f"summary_{REP_DISPLAY[rep]}_{SUMMARY_STAT}_main.csv"), index=False)

        df_z = reorder_table(rep_tables_z[rep], Z_METRICS)
        if not df_z.empty:
            df_z.to_csv(os.path.join(OUT_DIR, f"summary_{REP_DISPLAY[rep]}_{SUMMARY_STAT}_z.csv"), index=False)

    draw_main_metrics_three_heatmaps(rep_tables_main, OUT_DIR)
    draw_zc_zl_heatmaps_3x2(rep_tables_z, OUT_DIR)

    print(f"\n[DONE] All outputs saved in: {Path(OUT_DIR).resolve()}")

if __name__ == "__main__":
    main()

# In[ ]:
