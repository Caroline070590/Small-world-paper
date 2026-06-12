#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Auto-converted from the Jupyter notebook ``circular-agreament.ipynb``.

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
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = "csv"
OUTDIR = Path("representation_agreement_plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "circos_representation_agreement.png"
OUT_PDF = OUTDIR / "circos_representation_agreement.pdf"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}

# same colors as your boxplots
SECTOR_COLORS = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # darkgray
    "MTF": "#DDA0DD",   # plum
}

# link colors
SIGN_COLORING = False
POS_COLOR = "#5A91C2"
NEG_COLOR = "#C45A5A"
LINK_COLOR = "#7A7A7A"

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

TOPN = None   # None = use all datasets

DPI_SCREEN = 300
DPI_EXPORT = 600

R_OUT  = 1.00
R_IN   = 0.84
R_LINK = 0.78
GAP_DEG = 18.0

SECTOR_LABEL_FONTSIZE = 15
DATASET_FONTSIZE = 8
SHOW_DATASET_NAMES = True

# ============================================================
# HELPERS
# ============================================================
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
            else:
                summary[metric] = float(np.mean(vals))
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
                summary_stat="median"
            )

            row = {"dataset": dataset}
            row.update(summary)
            rep_tables[rep].append(row)

    for rep in REP_ORDER:
        rep_tables[rep] = pd.DataFrame(rep_tables[rep])

    return grouped, rep_tables

def reorder_datasets(values):
    ordered = [d for d in DATASET_ORDER if d in values]
    ordered += [d for d in values if d not in ordered]
    return ordered

def pol2cart(theta, r):
    return np.array([r*np.cos(theta), r*np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.60):
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)

def nice_text_rotation(theta):
    rot = np.degrees(theta) - 90
    if rot < -90:
        rot += 180
        ha = "right"
    elif rot > 90:
        rot -= 180
        ha = "right"
    else:
        ha = "left"
    return rot, ha

# ============================================================
# BUILD AGREEMENT TABLE
# ============================================================
def build_agreement_table(rep_tables: dict):
    """
    For each dataset:
      Spearman(QG,GAF), Spearman(QG,MTF), Spearman(GAF,MTF)
    computed across MAIN_METRICS.
    """
    qtn = rep_tables["QTN"].copy()
    gaf = rep_tables["GAF"].copy()
    mtf = rep_tables["MTF"].copy()

    for df in [qtn, gaf, mtf]:
        if not df.empty:
            df["dataset"] = df["dataset"].astype(str)

    common = set(qtn["dataset"]) & set(gaf["dataset"]) & set(mtf["dataset"])
    common = reorder_datasets(list(common))

    rows = []
    for ds in common:
        v_qtn = qtn.loc[qtn["dataset"] == ds, MAIN_METRICS].iloc[0].to_numpy(dtype=float)
        v_gaf = gaf.loc[gaf["dataset"] == ds, MAIN_METRICS].iloc[0].to_numpy(dtype=float)
        v_mtf = mtf.loc[mtf["dataset"] == ds, MAIN_METRICS].iloc[0].to_numpy(dtype=float)

        def safe_spearman(a, b):
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 3:
                return np.nan
            rho, _ = spearmanr(a[mask], b[mask])
            return float(rho) if np.isfinite(rho) else np.nan

        rows.append({
            "dataset": ds,
            "label": DATASET_LABELS.get(ds, ds),
            "rho_QG_GAF": safe_spearman(v_qtn, v_gaf),
            "rho_QG_MTF": safe_spearman(v_qtn, v_mtf),
            "rho_GAF_MTF": safe_spearman(v_gaf, v_mtf),
        })

    agree = pd.DataFrame(rows)

    if TOPN is not None and len(agree) > TOPN:
        # use mean absolute agreement across pairs to keep the most informative datasets
        agree["mean_abs_rho"] = agree[["rho_QG_GAF", "rho_QG_MTF", "rho_GAF_MTF"]].abs().mean(axis=1)
        agree = agree.sort_values("mean_abs_rho", ascending=False).head(TOPN).copy()

    return agree

# ============================================================
# PLOT
# ============================================================
def plot_representation_circos(agree_df, out_png, out_pdf, title):
    sectors = [
        ("QTN", agree_df[["dataset", "label"]].copy()),
        ("GAF", agree_df[["dataset", "label"]].copy()),
        ("MTF", agree_df[["dataset", "label"]].copy()),
    ]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    theta0 = np.deg2rad(40.0)
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(10, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)

    block_pos = {}

    # ----- draw sectors + dataset blocks -----
    for sec_name, df_items in sectors:
        start, end = sector_ranges[sec_name]

        base = Wedge((0, 0), R_OUT,
                     np.degrees(end), np.degrees(start),
                     width=(R_OUT - R_IN),
                     facecolor=SECTOR_COLORS[sec_name],
                     edgecolor="white", lw=2)
        ax.add_patch(base)

        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.10)
        rot, _ = nice_text_rotation(mid)
        ax.text(p[0], p[1], REP_DISPLAY[sec_name], fontsize=SECTOR_LABEL_FONTSIZE,
                fontweight="bold", ha="center", va="center", rotation=rot)

        labels = df_items["label"].astype(str).tolist()
        ids = df_items["dataset"].astype(str).tolist()
        n = len(labels)

        for i, (ds, lbl) in enumerate(zip(ids, labels)):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(sec_name, ds)] = th_center

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            blk = Wedge((0, 0), R_OUT,
                        np.degrees(th2), np.degrees(th1),
                        width=(R_OUT - R_IN),
                        facecolor=SECTOR_COLORS[sec_name],
                        edgecolor="white", lw=1.4)
            ax.add_patch(blk)

            if SHOW_DATASET_NAMES:
                ptxt = pol2cart(th_center, 1.05)
                rrot, ha2 = nice_text_rotation(th_center)
                ax.text(ptxt[0], ptxt[1], lbl,
                        fontsize=DATASET_FONTSIZE,
                        ha=ha2, va="center",
                        rotation=rrot, color="#111")

    # ----- agreement lookup -----
    agree_map = agree_df.set_index("dataset")[["rho_QG_GAF", "rho_QG_MTF", "rho_GAF_MTF"]].to_dict(orient="index")

    def get_rho(dataset, pair):
        if dataset not in agree_map:
            return np.nan
        d = agree_map[dataset]
        if pair == ("QTN", "GAF"):
            return float(d.get("rho_QG_GAF", np.nan))
        if pair == ("QTN", "MTF"):
            return float(d.get("rho_QG_MTF", np.nan))
        if pair == ("GAF", "MTF"):
            return float(d.get("rho_GAF_MTF", np.nan))
        return np.nan

    dataset_set = set(agree_df["dataset"].astype(str))

    abs_rhos = []
    for ds in dataset_set:
        for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
            rho = get_rho(ds, (a, b))
            if np.isfinite(rho):
                abs_rhos.append(abs(rho))
    max_abs = max(abs_rhos) if abs_rhos else 1.0
    max_abs = max(max_abs, 1e-9)

    # ----- links -----
    for ds in dataset_set:
        for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
            th_a = block_pos.get((a, ds), None)
            th_b = block_pos.get((b, ds), None)
            if th_a is None or th_b is None:
                continue

            rho = get_rho(ds, (a, b))
            if not np.isfinite(rho):
                continue

            lw = 0.6 + 6.0 * (abs(rho) / max_abs)
            alpha = 0.10 + 0.50 * (abs(rho) / max_abs)

            if SIGN_COLORING:
                col = POS_COLOR if rho >= 0 else NEG_COLOR
            else:
                col = LINK_COLOR

            path = bezier_link(th_a, th_b, r=R_LINK, bend=0.60)
            ax.add_patch(PathPatch(path, facecolor="none", edgecolor=col,
                                   lw=lw, alpha=alpha, capstyle="round"))

    ax.set_title(title, fontsize=16, pad=18)

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    grouped, rep_tables = build_rep_tables(INPUT_DIR, MAIN_METRICS)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    agree_df = build_agreement_table(rep_tables)
    agree_df.to_csv(OUTDIR / "representation_agreement_per_dataset.csv", index=False)

    plot_representation_circos(
        agree_df=agree_df,
        out_png=OUT_PNG,
        out_pdf=OUT_PDF,
        title="Agreement across QG, GAF, and MTF (Spearman ρ across topology metrics)"
    )

    print("[OK] Saved:")
    print(" -", OUT_PNG.resolve())
    print(" -", OUT_PDF.resolve())
    print(" -", (OUTDIR / "representation_agreement_per_dataset.csv").resolve())

if __name__ == "__main__":
    main()

# In[2]:

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = "csv"
OUTDIR = Path("representation_metric_circos")
OUTDIR.mkdir(parents=True, exist_ok=True)

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}

# same colors as boxplots
SECTOR_COLORS = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # gray
    "MTF": "#DDA0DD",   # plum
}

# neutral link color
LINK_COLOR = "#7A7A7A"

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

METRICS = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
}

SUMMARY_STAT = "median"

DPI_SCREEN = 250
DPI_EXPORT = 600

R_OUT  = 1.00
R_IN   = 0.84
R_LINK = 0.78
GAP_DEG = 18.0

SECTOR_LABEL_FONTSIZE = 16
METRIC_FONTSIZE = 10

# ============================================================
# HELPERS
# ============================================================
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
            else:
                summary[metric] = float(np.mean(vals))
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

def reorder_datasets(values):
    ordered = [d for d in DATASET_ORDER if d in values]
    ordered += [d for d in values if d not in ordered]
    return ordered

def pol2cart(theta, r):
    return np.array([r*np.cos(theta), r*np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.58):
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)

def nice_text_rotation(theta):
    rot = np.degrees(theta) - 90
    if rot < -90:
        rot += 180
        ha = "right"
    elif rot > 90:
        rot -= 180
        ha = "right"
    else:
        ha = "left"
    return rot, ha

def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

# ============================================================
# AGREEMENT / DIFFERENCE
# ============================================================
def metric_pair_agreement(v1, v2, vmin, vmax):
    """
    Agreement score in [0,1]:
    1 = same value, 0 = maximally different in the observed global metric range
    """
    if not np.isfinite(v1) or not np.isfinite(v2):
        return np.nan

    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return 1.0 if np.isclose(v1, v2) else 0.5

    d = abs(v1 - v2) / span
    score = 1.0 - np.clip(d, 0.0, 1.0)
    return float(score)

def build_global_metric_ranges(rep_tables: dict):
    metric_ranges = {}
    for metric in METRICS:
        vals = []
        for rep in REP_ORDER:
            df = rep_tables[rep]
            if df.empty or metric not in df.columns:
                continue
            x = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
            if len(x) > 0:
                vals.append(x)

        if len(vals) == 0:
            metric_ranges[metric] = (0.0, 1.0)
        else:
            vals = np.concatenate(vals)
            metric_ranges[metric] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    return metric_ranges

# ============================================================
# ONE DATASET = ONE CIRCOS
# ============================================================
def plot_dataset_metric_circos(dataset_name, values_by_rep, metric_ranges, out_png, out_pdf):
    """
    values_by_rep:
      {
        "QTN": {"sigma_small_world": ..., ...},
        "GAF": {...},
        "MTF": {...}
      }
    """
    sectors = [(rep, METRICS) for rep in REP_ORDER]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    theta0 = np.deg2rad(40.0)
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(9, 9), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)

    block_pos = {}

    # ----- draw sectors + metric blocks -----
    for rep, metrics in sectors:
        start, end = sector_ranges[rep]

        base = Wedge((0, 0), R_OUT,
                     np.degrees(end), np.degrees(start),
                     width=(R_OUT - R_IN),
                     facecolor=SECTOR_COLORS[rep],
                     edgecolor="white", lw=2)
        ax.add_patch(base)

        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.10)
        rot, _ = nice_text_rotation(mid)
        ax.text(
            p[0], p[1], REP_DISPLAY[rep],
            fontsize=SECTOR_LABEL_FONTSIZE,
            fontweight="bold",
            ha="center", va="center",
            rotation=rot
        )

        n = len(metrics)
        for i, metric in enumerate(metrics):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(rep, metric)] = th_center

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            blk = Wedge((0, 0), R_OUT,
                        np.degrees(th2), np.degrees(th1),
                        width=(R_OUT - R_IN),
                        facecolor=SECTOR_COLORS[rep],
                        edgecolor="white", lw=1.5)
            ax.add_patch(blk)

            ptxt = pol2cart(th_center, 1.04)
            rrot, ha2 = nice_text_rotation(th_center)
            ax.text(
                ptxt[0], ptxt[1],
                METRIC_LABELS.get(metric, metric),
                fontsize=METRIC_FONTSIZE,
                ha=ha2, va="center",
                rotation=rrot, color="#111"
            )

    # ----- links per metric -----
    abs_scores = []
    for metric in METRICS:
        pairs = [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]
        vmin, vmax = metric_ranges[metric]

        for a, b in pairs:
            va = values_by_rep[a].get(metric, np.nan)
            vb = values_by_rep[b].get(metric, np.nan)
            score = metric_pair_agreement(va, vb, vmin, vmax)
            if np.isfinite(score):
                abs_scores.append(score)

    max_score = max(abs_scores) if abs_scores else 1.0
    max_score = max(max_score, 1e-9)

    for metric in METRICS:
        pairs = [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]
        vmin, vmax = metric_ranges[metric]

        for a, b in pairs:
            th_a = block_pos.get((a, metric), None)
            th_b = block_pos.get((b, metric), None)
            if th_a is None or th_b is None:
                continue

            va = values_by_rep[a].get(metric, np.nan)
            vb = values_by_rep[b].get(metric, np.nan)
            score = metric_pair_agreement(va, vb, vmin, vmax)
            if not np.isfinite(score):
                continue

            lw = 0.5 + 6.0 * (score / max_score)
            alpha = 0.08 + 0.55 * (score / max_score)

            path = bezier_link(th_a, th_b, r=R_LINK, bend=0.58)
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=LINK_COLOR,
                    lw=lw,
                    alpha=alpha,
                    capstyle="round"
                )
            )

    label = DATASET_LABELS.get(dataset_name, dataset_name)
    ax.set_title(
        f"{label}: agreement across QG, GAF, and MTF\n(metric-wise similarity across topology measures)",
        fontsize=14,
        pad=18
    )

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    grouped, rep_tables = build_rep_tables(INPUT_DIR, METRICS)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    metric_ranges = build_global_metric_ranges(rep_tables)

    # convert rep tables into lookup
    rep_lookup = {}
    for rep in REP_ORDER:
        df = rep_tables[rep].copy()
        if not df.empty:
            df["dataset"] = df["dataset"].astype(str)
            rep_lookup[rep] = df.set_index("dataset").to_dict(orient="index")
        else:
            rep_lookup[rep] = {}

    common = set(rep_lookup["QTN"].keys()) & set(rep_lookup["GAF"].keys()) & set(rep_lookup["MTF"].keys())
    common = reorder_datasets(list(common))

    summary_rows = []

    for ds in common:
        values_by_rep = {
            "QTN": rep_lookup["QTN"][ds],
            "GAF": rep_lookup["GAF"][ds],
            "MTF": rep_lookup["MTF"][ds],
        }

        # average agreement per pair for export
        pair_scores = {}
        for a, b, label in [
            ("QTN", "GAF", "QG_GAF"),
            ("QTN", "MTF", "QG_MTF"),
            ("GAF", "MTF", "GAF_MTF"),
        ]:
            scores = []
            for metric in METRICS:
                vmin, vmax = metric_ranges[metric]
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc):
                    scores.append(sc)
            pair_scores[label] = float(np.mean(scores)) if scores else np.nan

        summary_rows.append({
            "dataset": ds,
            "label": DATASET_LABELS.get(ds, ds),
            **pair_scores
        })

        fname = safe_filename(DATASET_LABELS.get(ds, ds))
        out_png = OUTDIR / f"{fname}_metric_circos.png"
        out_pdf = OUTDIR / f"{fname}_metric_circos.pdf"

        plot_dataset_metric_circos(
            dataset_name=ds,
            values_by_rep=values_by_rep,
            metric_ranges=metric_ranges,
            out_png=out_png,
            out_pdf=out_pdf
        )

    pd.DataFrame(summary_rows).to_csv(OUTDIR / "dataset_metric_agreement_summary.csv", index=False)

    print("[OK] Saved circular plots to:", OUTDIR.resolve())

if __name__ == "__main__":
    main()

# In[5]:

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = "csv"
OUTDIR = Path("representation_metric_circos")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "all_datasets_metric_circos_colored_clean.png"
OUT_PDF = OUTDIR / "all_datasets_metric_circos_colored_clean.pdf"
OUT_SUMMARY = OUTDIR / "dataset_metric_agreement_summary.csv"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}

# same colors as boxplots
SECTOR_COLORS = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # gray
    "MTF": "#DDA0DD",   # plum
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

METRICS = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
}

SUMMARY_STAT = "median"

# set to None to keep all datasets
TOP_DATASETS = 8

DPI_SCREEN = 250
DPI_EXPORT = 600

R_OUT  = 1.00
R_IN   = 0.84
R_LINK = 0.78
GAP_DEG = 18.0

SECTOR_LABEL_FONTSIZE = 16
METRIC_FONTSIZE = 10

# small angular spread so links do not overlap perfectly
JITTER_FRACTION_OF_BLOCK = 0.32

# ============================================================
# HELPERS
# ============================================================
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
            else:
                summary[metric] = float(np.mean(vals))
    return summary

def build_rep_tables(input_dir: str, metric_order: list):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir.resolve()}")

    all_files = sorted(input_dir.glob("metrics_*.csv"))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir.resolve()}")

    all_files = [str(f) for f in all_files]

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

def reorder_datasets(values):
    ordered = [d for d in DATASET_ORDER if d in values]
    ordered += [d for d in values if d not in ordered]
    return ordered

def pol2cart(theta, r):
    return np.array([r * np.cos(theta), r * np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.58):
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)

def nice_text_rotation(theta):
    rot = np.degrees(theta) - 90
    if rot < -90:
        rot += 180
        ha = "right"
    elif rot > 90:
        rot -= 180
        ha = "right"
    else:
        ha = "left"
    return rot, ha

def safe_filename(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

# ============================================================
# AGREEMENT
# ============================================================
def metric_pair_agreement(v1, v2, vmin, vmax):
    if not np.isfinite(v1) or not np.isfinite(v2):
        return np.nan

    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return 1.0 if np.isclose(v1, v2) else 0.5

    d = abs(v1 - v2) / span
    score = 1.0 - np.clip(d, 0.0, 1.0)
    return float(score)

def build_global_metric_ranges(rep_tables: dict):
    metric_ranges = {}
    for metric in METRICS:
        vals = []
        for rep in REP_ORDER:
            df = rep_tables[rep]
            if df.empty or metric not in df.columns:
                continue
            x = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
            if len(x) > 0:
                vals.append(x)

        if len(vals) == 0:
            metric_ranges[metric] = (0.0, 1.0)
        else:
            vals = np.concatenate(vals)
            metric_ranges[metric] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    return metric_ranges

def build_dataset_colors(dataset_names):
    # harmonious custom palette
    palette = [
        "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D",
        "#59A14F", "#8CD17D", "#E15759", "#FF9D9A",
        "#B07AA1", "#D4A6C8", "#9C755F", "#D7B5A6",
        "#E377C2", "#F7B6D2"
    ]
    return {ds: palette[i % len(palette)] for i, ds in enumerate(dataset_names)}

def build_dataset_summary(dataset_values, metric_ranges):
    rows = []
    for ds, values_by_rep in dataset_values.items():
        pair_scores = {}
        for a, b, label in [
            ("QTN", "GAF", "QG_GAF"),
            ("QTN", "MTF", "QG_MTF"),
            ("GAF", "MTF", "GAF_MTF"),
        ]:
            scores = []
            for metric in METRICS:
                vmin, vmax = metric_ranges[metric]
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc):
                    scores.append(sc)
            pair_scores[label] = float(np.mean(scores)) if scores else np.nan

        mean_agreement = np.nanmean(list(pair_scores.values()))
        rows.append({
            "dataset": ds,
            "label": DATASET_LABELS.get(ds, ds),
            "mean_agreement": mean_agreement,
            **pair_scores
        })

    return pd.DataFrame(rows).sort_values("mean_agreement", ascending=False)

def metric_block_halfwidth(sector_start, sector_end, n_metrics):
    return abs(sector_end - sector_start) / n_metrics / 2.0

def jittered_angle(base_theta, dataset_index, n_datasets, half_block_width):
    if n_datasets <= 1:
        return base_theta
    frac = (dataset_index / (n_datasets - 1)) - 0.5
    return base_theta + frac * (2.0 * half_block_width * JITTER_FRACTION_OF_BLOCK)

# ============================================================
# PLOT
# ============================================================
def plot_all_datasets_metric_circos(dataset_values, metric_ranges, dataset_colors, out_png, out_pdf):
    sectors = [(rep, METRICS) for rep in REP_ORDER]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    theta0 = np.deg2rad(40.0)
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(12, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.28, 1.50)
    ax.set_ylim(-1.22, 1.22)

    block_pos = {}

    for rep, metrics in sectors:
        start, end = sector_ranges[rep]

        base = Wedge(
            (0, 0), R_OUT,
            np.degrees(end), np.degrees(start),
            width=(R_OUT - R_IN),
            facecolor=SECTOR_COLORS[rep],
            edgecolor="white", lw=2
        )
        ax.add_patch(base)

        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.10)
        rot, _ = nice_text_rotation(mid)
        ax.text(
            p[0], p[1], REP_DISPLAY[rep],
            fontsize=SECTOR_LABEL_FONTSIZE,
            fontweight="bold",
            ha="center", va="center",
            rotation=rot
        )

        n = len(metrics)
        for i, metric in enumerate(metrics):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(rep, metric)] = th_center

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            blk = Wedge(
                (0, 0), R_OUT,
                np.degrees(th2), np.degrees(th1),
                width=(R_OUT - R_IN),
                facecolor=SECTOR_COLORS[rep],
                edgecolor="white", lw=1.5
            )
            ax.add_patch(blk)

            ptxt = pol2cart(th_center, 1.05)
            rrot, ha2 = nice_text_rotation(th_center)
            ax.text(
                ptxt[0], ptxt[1],
                METRIC_LABELS.get(metric, metric),
                fontsize=METRIC_FONTSIZE,
                ha=ha2, va="center",
                rotation=rrot, color="#111"
            )

    dataset_names = list(dataset_values.keys())
    n_datasets = len(dataset_names)

    all_scores = []
    for ds, values_by_rep in dataset_values.items():
        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]
            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc):
                    all_scores.append(sc)

    max_score = max(all_scores) if all_scores else 1.0
    max_score = max(max_score, 1e-9)

    for ds_idx, (ds, values_by_rep) in enumerate(dataset_values.items()):
        link_color = dataset_colors[ds]

        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]

            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                th_a_base = block_pos.get((a, metric), None)
                th_b_base = block_pos.get((b, metric), None)
                if th_a_base is None or th_b_base is None:
                    continue

                start_a, end_a = sector_ranges[a]
                start_b, end_b = sector_ranges[b]
                half_a = metric_block_halfwidth(start_a, end_a, len(METRICS))
                half_b = metric_block_halfwidth(start_b, end_b, len(METRICS))

                th_a = jittered_angle(th_a_base, ds_idx, n_datasets, half_a)
                th_b = jittered_angle(th_b_base, ds_idx, n_datasets, half_b)

                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if not np.isfinite(sc):
                    continue

                lw = 0.25 + 3.8 * (sc / max_score)
                alpha = 0.06 + 0.38 * (sc / max_score)

                path = bezier_link(th_a, th_b, r=R_LINK, bend=0.58)
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=link_color,
                        lw=lw,
                        alpha=alpha,
                        capstyle="round"
                    )
                )

    legend_handles = [
        Line2D([0], [0], color=dataset_colors[ds], lw=3, label=DATASET_LABELS.get(ds, ds))
        for ds in dataset_values.keys()
    ]

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        title="Datasets",
        title_fontsize=11,
        ncol=1
    )

    ax.set_title(
        "Agreement across QG, GAF, and MTF\n(dataset-colored links; thicker = stronger agreement)",
        fontsize=16,
        pad=18
    )

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    grouped, rep_tables = build_rep_tables(INPUT_DIR, METRICS)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    metric_ranges = build_global_metric_ranges(rep_tables)

    rep_lookup = {}
    for rep in REP_ORDER:
        df = rep_tables[rep].copy()
        if not df.empty:
            df["dataset"] = df["dataset"].astype(str)
            rep_lookup[rep] = df.set_index("dataset").to_dict(orient="index")
        else:
            rep_lookup[rep] = {}

    common = set(rep_lookup["QTN"].keys()) & set(rep_lookup["GAF"].keys()) & set(rep_lookup["MTF"].keys())
    common = reorder_datasets(list(common))

    dataset_values_full = {
        ds: {
            "QTN": rep_lookup["QTN"][ds],
            "GAF": rep_lookup["GAF"][ds],
            "MTF": rep_lookup["MTF"][ds],
        }
        for ds in common
    }

    summary_df = build_dataset_summary(dataset_values_full, metric_ranges)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    if TOP_DATASETS is not None and TOP_DATASETS < len(summary_df):
        keep = summary_df["dataset"].head(TOP_DATASETS).tolist()
        dataset_values = {ds: dataset_values_full[ds] for ds in keep}
    else:
        dataset_values = dataset_values_full

    dataset_colors = build_dataset_colors(list(dataset_values.keys()))

    plot_all_datasets_metric_circos(
        dataset_values=dataset_values,
        metric_ranges=metric_ranges,
        dataset_colors=dataset_colors,
        out_png=OUT_PNG,
        out_pdf=OUT_PDF
    )

    print("[OK] Saved:")
    print(" -", OUT_PNG.resolve())
    print(" -", OUT_PDF.resolve())
    print(" -", OUT_SUMMARY.resolve())

if __name__ == "__main__":
    main()

# In[6]:

import os
import re
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = "csv"
OUTDIR = Path("representation_metric_circos")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "all_datasets_metric_circos_colored_clean_thin.png"
OUT_PDF = OUTDIR / "all_datasets_metric_circos_colored_clean_thin.pdf"
OUT_SUMMARY = OUTDIR / "dataset_metric_agreement_summary.csv"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}

# same colors as boxplots
SECTOR_COLORS = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # gray
    "MTF": "#DDA0DD",   # plum
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

METRICS = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
}

SUMMARY_STAT = "median"

# set to None to keep all datasets
TOP_DATASETS = 8

DPI_SCREEN = 250
DPI_EXPORT = 600

R_OUT  = 1.00
R_IN   = 0.84
R_LINK = 0.78
GAP_DEG = 18.0

SECTOR_LABEL_FONTSIZE = 16
METRIC_FONTSIZE = 10

# smaller angular spread so links do not overlap perfectly
JITTER_FRACTION_OF_BLOCK = 0.32

# thinner links
LW_MIN = 0.10
LW_SCALE = 1.70
ALPHA_MIN = 0.035
ALPHA_SCALE = 0.18
LINK_BEND = 0.52

# ============================================================
# HELPERS
# ============================================================
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
            else:
                summary[metric] = float(np.mean(vals))
    return summary

def build_rep_tables(input_dir: str, metric_order: list):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir.resolve()}")

    all_files = sorted(input_dir.glob("metrics_*.csv"))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir.resolve()}")

    all_files = [str(f) for f in all_files]

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

def reorder_datasets(values):
    ordered = [d for d in DATASET_ORDER if d in values]
    ordered += [d for d in values if d not in ordered]
    return ordered

def pol2cart(theta, r):
    return np.array([r * np.cos(theta), r * np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.52):
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)

def nice_text_rotation(theta):
    rot = np.degrees(theta) - 90
    if rot < -90:
        rot += 180
        ha = "right"
    elif rot > 90:
        rot -= 180
        ha = "right"
    else:
        ha = "left"
    return rot, ha

# ============================================================
# AGREEMENT
# ============================================================
def metric_pair_agreement(v1, v2, vmin, vmax):
    if not np.isfinite(v1) or not np.isfinite(v2):
        return np.nan

    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return 1.0 if np.isclose(v1, v2) else 0.5

    d = abs(v1 - v2) / span
    score = 1.0 - np.clip(d, 0.0, 1.0)
    return float(score)

def build_global_metric_ranges(rep_tables: dict):
    metric_ranges = {}
    for metric in METRICS:
        vals = []
        for rep in REP_ORDER:
            df = rep_tables[rep]
            if df.empty or metric not in df.columns:
                continue
            x = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
            if len(x) > 0:
                vals.append(x)

        if len(vals) == 0:
            metric_ranges[metric] = (0.0, 1.0)
        else:
            vals = np.concatenate(vals)
            metric_ranges[metric] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    return metric_ranges

def build_dataset_colors(dataset_names):
    palette = [
        "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D",
        "#59A14F", "#8CD17D", "#E15759", "#FF9D9A",
        "#B07AA1", "#D4A6C8", "#9C755F", "#D7B5A6",
        "#E377C2", "#F7B6D2"
    ]
    return {ds: palette[i % len(palette)] for i, ds in enumerate(dataset_names)}

def build_dataset_summary(dataset_values, metric_ranges):
    rows = []
    for ds, values_by_rep in dataset_values.items():
        pair_scores = {}
        for a, b, label in [
            ("QTN", "GAF", "QG_GAF"),
            ("QTN", "MTF", "QG_MTF"),
            ("GAF", "MTF", "GAF_MTF"),
        ]:
            scores = []
            for metric in METRICS:
                vmin, vmax = metric_ranges[metric]
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc):
                    scores.append(sc)
            pair_scores[label] = float(np.mean(scores)) if scores else np.nan

        mean_agreement = np.nanmean(list(pair_scores.values()))
        rows.append({
            "dataset": ds,
            "label": DATASET_LABELS.get(ds, ds),
            "mean_agreement": mean_agreement,
            **pair_scores
        })

    return pd.DataFrame(rows).sort_values("mean_agreement", ascending=False)

def metric_block_halfwidth(sector_start, sector_end, n_metrics):
    return abs(sector_end - sector_start) / n_metrics / 2.0

def jittered_angle(base_theta, dataset_index, n_datasets, half_block_width):
    if n_datasets <= 1:
        return base_theta
    frac = (dataset_index / (n_datasets - 1)) - 0.5
    return base_theta + frac * (2.0 * half_block_width * JITTER_FRACTION_OF_BLOCK)

# ============================================================
# PLOT
# ============================================================
def plot_all_datasets_metric_circos(dataset_values, metric_ranges, dataset_colors, out_png, out_pdf):
    sectors = [(rep, METRICS) for rep in REP_ORDER]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    theta0 = np.deg2rad(40.0)
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(12, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.28, 1.50)
    ax.set_ylim(-1.22, 1.22)

    block_pos = {}

    # draw sectors + metric blocks
    for rep, metrics in sectors:
        start, end = sector_ranges[rep]

        base = Wedge(
            (0, 0), R_OUT,
            np.degrees(end), np.degrees(start),
            width=(R_OUT - R_IN),
            facecolor=SECTOR_COLORS[rep],
            edgecolor="white", lw=2
        )
        ax.add_patch(base)

        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.10)
        rot, _ = nice_text_rotation(mid)
        ax.text(
            p[0], p[1], REP_DISPLAY[rep],
            fontsize=SECTOR_LABEL_FONTSIZE,
            fontweight="bold",
            ha="center", va="center",
            rotation=rot
        )

        n = len(metrics)
        for i, metric in enumerate(metrics):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(rep, metric)] = th_center

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            blk = Wedge(
                (0, 0), R_OUT,
                np.degrees(th2), np.degrees(th1),
                width=(R_OUT - R_IN),
                facecolor=SECTOR_COLORS[rep],
                edgecolor="white", lw=1.5
            )
            ax.add_patch(blk)

            ptxt = pol2cart(th_center, 1.05)
            rrot, ha2 = nice_text_rotation(th_center)
            ax.text(
                ptxt[0], ptxt[1],
                METRIC_LABELS.get(metric, metric),
                fontsize=METRIC_FONTSIZE,
                ha=ha2, va="center",
                rotation=rrot, color="#111"
            )

    dataset_names = list(dataset_values.keys())
    n_datasets = len(dataset_names)

    all_scores = []
    for ds, values_by_rep in dataset_values.items():
        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]
            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc):
                    all_scores.append(sc)

    max_score = max(all_scores) if all_scores else 1.0
    max_score = max(max_score, 1e-9)

    for ds_idx, (ds, values_by_rep) in enumerate(dataset_values.items()):
        link_color = dataset_colors[ds]

        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]

            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                th_a_base = block_pos.get((a, metric), None)
                th_b_base = block_pos.get((b, metric), None)
                if th_a_base is None or th_b_base is None:
                    continue

                start_a, end_a = sector_ranges[a]
                start_b, end_b = sector_ranges[b]
                half_a = metric_block_halfwidth(start_a, end_a, len(METRICS))
                half_b = metric_block_halfwidth(start_b, end_b, len(METRICS))

                th_a = jittered_angle(th_a_base, ds_idx, n_datasets, half_a)
                th_b = jittered_angle(th_b_base, ds_idx, n_datasets, half_b)

                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if not np.isfinite(sc):
                    continue

                lw = LW_MIN + LW_SCALE * (sc / max_score)
                alpha = ALPHA_MIN + ALPHA_SCALE * (sc / max_score)

                path = bezier_link(th_a, th_b, r=R_LINK, bend=LINK_BEND)
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=link_color,
                        lw=lw,
                        alpha=alpha,
                        capstyle="round"
                    )
                )

    legend_handles = [
        Line2D([0], [0], color=dataset_colors[ds], lw=3, label=DATASET_LABELS.get(ds, ds))
        for ds in dataset_values.keys()
    ]

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        title="Datasets",
        title_fontsize=11,
        ncol=1
    )

    ax.set_title(
        "Agreement across QG, GAF, and MTF\n(dataset-colored links; thicker = stronger agreement)",
        fontsize=16,
        pad=18
    )

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    grouped, rep_tables = build_rep_tables(INPUT_DIR, METRICS)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    metric_ranges = build_global_metric_ranges(rep_tables)

    rep_lookup = {}
    for rep in REP_ORDER:
        df = rep_tables[rep].copy()
        if not df.empty:
            df["dataset"] = df["dataset"].astype(str)
            rep_lookup[rep] = df.set_index("dataset").to_dict(orient="index")
        else:
            rep_lookup[rep] = {}

    common = set(rep_lookup["QTN"].keys()) & set(rep_lookup["GAF"].keys()) & set(rep_lookup["MTF"].keys())
    common = reorder_datasets(list(common))

    dataset_values_full = {
        ds: {
            "QTN": rep_lookup["QTN"][ds],
            "GAF": rep_lookup["GAF"][ds],
            "MTF": rep_lookup["MTF"][ds],
        }
        for ds in common
    }

    summary_df = build_dataset_summary(dataset_values_full, metric_ranges)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    if TOP_DATASETS is not None and TOP_DATASETS < len(summary_df):
        keep = summary_df["dataset"].head(TOP_DATASETS).tolist()
        dataset_values = {ds: dataset_values_full[ds] for ds in keep}
    else:
        dataset_values = dataset_values_full

    dataset_colors = build_dataset_colors(list(dataset_values.keys()))

    plot_all_datasets_metric_circos(
        dataset_values=dataset_values,
        metric_ranges=metric_ranges,
        dataset_colors=dataset_colors,
        out_png=OUT_PNG,
        out_pdf=OUT_PDF
    )

    print("[OK] Saved:")
    print(" -", OUT_PNG.resolve())
    print(" -", OUT_PDF.resolve())
    print(" -", OUT_SUMMARY.resolve())

if __name__ == "__main__":
    main()

# In[8]:

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
INPUT_DIR = "csv"
OUTDIR = Path("representation_metric_circos")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUTDIR / "all_datasets_metric_circos_thresholded.png"
OUT_PDF = OUTDIR / "all_datasets_metric_circos_thresholded.pdf"
OUT_SUMMARY = OUTDIR / "dataset_metric_agreement_summary.csv"
OUT_LINKS = OUTDIR / "dataset_metric_agreement_links_thresholded.csv"

REP_ORDER = ["QTN", "GAF", "MTF"]
REP_DISPLAY = {"QTN": "QG", "GAF": "GAF", "MTF": "MTF"}

# same colors as boxplots
SECTOR_COLORS = {
    "QTN": "#008080",   # teal
    "GAF": "#A9A9A9",   # gray
    "MTF": "#DDA0DD",   # plum
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

METRICS = [
    "sigma_small_world",
    "gamma_C_over_Crand",
    "lambda_L_over_Lrand",
    "transitivity",
    "char_path_len_gcc",
    "global_efficiency",
]

METRIC_LABELS = {
    "sigma_small_world": r"$\sigma$",
    "gamma_C_over_Crand": r"$\gamma$",
    "lambda_L_over_Lrand": r"$\lambda$",
    "transitivity": "Trans.",
    "char_path_len_gcc": "Path",
    "global_efficiency": "GE",
}

SUMMARY_STAT = "median"

# Keep only top datasets by mean agreement. Set None to keep all.
TOP_DATASETS = 8

# MAIN NEW PARAMETER:
# show only links with agreement >= threshold
AGREEMENT_THRESHOLD = 0.95

DPI_SCREEN = 250
DPI_EXPORT = 600

R_OUT  = 1.00
R_IN   = 0.84
R_LINK = 0.78
GAP_DEG = 18.0

SECTOR_LABEL_FONTSIZE = 16
METRIC_FONTSIZE = 10

# smaller angular spread so links do not overlap perfectly
JITTER_FRACTION_OF_BLOCK = 0.32

# thinner links
LW_MIN = 0.10
LW_SCALE = 1.70
ALPHA_MIN = 0.035
ALPHA_SCALE = 0.18
LINK_BEND = 0.52

# ============================================================
# HELPERS
# ============================================================
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
            else:
                summary[metric] = float(np.mean(vals))
    return summary

def build_rep_tables(input_dir: str, metric_order: list):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR does not exist: {input_dir.resolve()}")

    all_files = sorted(input_dir.glob("metrics_*.csv"))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir.resolve()}")

    all_files = [str(f) for f in all_files]

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

def reorder_datasets(values):
    ordered = [d for d in DATASET_ORDER if d in values]
    ordered += [d for d in values if d not in ordered]
    return ordered

def pol2cart(theta, r):
    return np.array([r * np.cos(theta), r * np.sin(theta)], float)

def bezier_link(theta_a, theta_b, r=R_LINK, bend=0.52):
    p0 = pol2cart(theta_a, r)
    p3 = pol2cart(theta_b, r)
    c0 = p0 * (1 - bend)
    c1 = p3 * (1 - bend)
    verts = [tuple(p0), tuple(c0), tuple(c1), tuple(p3)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    return MplPath(verts, codes)

def angle_for_block(sector_start, sector_end, i, n):
    frac = (i + 0.5) / n
    return sector_start + frac * (sector_end - sector_start)

def nice_text_rotation(theta):
    rot = np.degrees(theta) - 90
    if rot < -90:
        rot += 180
        ha = "right"
    elif rot > 90:
        rot -= 180
        ha = "right"
    else:
        ha = "left"
    return rot, ha

# ============================================================
# AGREEMENT
# ============================================================
def metric_pair_agreement(v1, v2, vmin, vmax):
    if not np.isfinite(v1) or not np.isfinite(v2):
        return np.nan

    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        return 1.0 if np.isclose(v1, v2) else 0.5

    d = abs(v1 - v2) / span
    score = 1.0 - np.clip(d, 0.0, 1.0)
    return float(score)

def build_global_metric_ranges(rep_tables: dict):
    metric_ranges = {}
    for metric in METRICS:
        vals = []
        for rep in REP_ORDER:
            df = rep_tables[rep]
            if df.empty or metric not in df.columns:
                continue
            x = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
            if len(x) > 0:
                vals.append(x)

        if len(vals) == 0:
            metric_ranges[metric] = (0.0, 1.0)
        else:
            vals = np.concatenate(vals)
            metric_ranges[metric] = (float(np.nanmin(vals)), float(np.nanmax(vals)))
    return metric_ranges

def build_dataset_colors(dataset_names):
    palette = [
        "#4E79A7", "#A0CBE8", "#F28E2B", "#FFBE7D",
        "#59A14F", "#8CD17D", "#E15759", "#FF9D9A",
        "#B07AA1", "#D4A6C8", "#9C755F", "#D7B5A6",
        "#E377C2", "#F7B6D2"
    ]
    return {ds: palette[i % len(palette)] for i, ds in enumerate(dataset_names)}

def build_dataset_summary(dataset_values, metric_ranges):
    rows = []
    for ds, values_by_rep in dataset_values.items():
        pair_scores = {}
        for a, b, label in [
            ("QTN", "GAF", "QG_GAF"),
            ("QTN", "MTF", "QG_MTF"),
            ("GAF", "MTF", "GAF_MTF"),
        ]:
            scores = []
            for metric in METRICS:
                vmin, vmax = metric_ranges[metric]
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc):
                    scores.append(sc)
            pair_scores[label] = float(np.mean(scores)) if scores else np.nan

        mean_agreement = np.nanmean(list(pair_scores.values()))
        rows.append({
            "dataset": ds,
            "label": DATASET_LABELS.get(ds, ds),
            "mean_agreement": mean_agreement,
            **pair_scores
        })

    return pd.DataFrame(rows).sort_values("mean_agreement", ascending=False)

def build_link_table(dataset_values, metric_ranges):
    rows = []
    for ds, values_by_rep in dataset_values.items():
        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]
            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                rows.append({
                    "dataset": ds,
                    "label": DATASET_LABELS.get(ds, ds),
                    "metric": metric,
                    "metric_label": METRIC_LABELS.get(metric, metric),
                    "pair": f"{REP_DISPLAY[a]}-{REP_DISPLAY[b]}",
                    "agreement_score": sc
                })
    return pd.DataFrame(rows)

def metric_block_halfwidth(sector_start, sector_end, n_metrics):
    return abs(sector_end - sector_start) / n_metrics / 2.0

def jittered_angle(base_theta, dataset_index, n_datasets, half_block_width):
    if n_datasets <= 1:
        return base_theta
    frac = (dataset_index / (n_datasets - 1)) - 0.5
    return base_theta + frac * (2.0 * half_block_width * JITTER_FRACTION_OF_BLOCK)

# ============================================================
# PLOT
# ============================================================
def plot_all_datasets_metric_circos(dataset_values, metric_ranges, dataset_colors, out_png, out_pdf):
    sectors = [(rep, METRICS) for rep in REP_ORDER]

    total_gap = GAP_DEG * len(sectors)
    usable = 360.0 - total_gap
    span = usable / len(sectors)

    theta0 = np.deg2rad(40.0)
    sector_ranges = {}
    cursor = theta0

    for name, _ in sectors:
        start = cursor
        end = cursor - np.deg2rad(span)
        sector_ranges[name] = (start, end)
        cursor = end - np.deg2rad(GAP_DEG)

    fig = plt.figure(figsize=(12, 10), dpi=DPI_SCREEN, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.28, 1.50)
    ax.set_ylim(-1.22, 1.22)

    block_pos = {}

    # draw sectors + metric blocks
    for rep, metrics in sectors:
        start, end = sector_ranges[rep]

        base = Wedge(
            (0, 0), R_OUT,
            np.degrees(end), np.degrees(start),
            width=(R_OUT - R_IN),
            facecolor=SECTOR_COLORS[rep],
            edgecolor="white", lw=2
        )
        ax.add_patch(base)

        mid = 0.5 * (start + end)
        p = pol2cart(mid, 1.10)
        rot, _ = nice_text_rotation(mid)
        ax.text(
            p[0], p[1], REP_DISPLAY[rep],
            fontsize=SECTOR_LABEL_FONTSIZE,
            fontweight="bold",
            ha="center", va="center",
            rotation=rot
        )

        n = len(metrics)
        for i, metric in enumerate(metrics):
            th_center = angle_for_block(start, end, i, n)
            block_pos[(rep, metric)] = th_center

            th1 = start + (i / n) * (end - start)
            th2 = start + ((i + 1) / n) * (end - start)

            blk = Wedge(
                (0, 0), R_OUT,
                np.degrees(th2), np.degrees(th1),
                width=(R_OUT - R_IN),
                facecolor=SECTOR_COLORS[rep],
                edgecolor="white", lw=1.5
            )
            ax.add_patch(blk)

            ptxt = pol2cart(th_center, 1.05)
            rrot, ha2 = nice_text_rotation(th_center)
            ax.text(
                ptxt[0], ptxt[1],
                METRIC_LABELS.get(metric, metric),
                fontsize=METRIC_FONTSIZE,
                ha=ha2, va="center",
                rotation=rrot, color="#111"
            )

    dataset_names = list(dataset_values.keys())
    n_datasets = len(dataset_names)

    kept_scores = []
    for ds, values_by_rep in dataset_values.items():
        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]
            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if np.isfinite(sc) and sc >= AGREEMENT_THRESHOLD:
                    kept_scores.append(sc)

    max_score = max(kept_scores) if kept_scores else 1.0
    max_score = max(max_score, 1e-9)

    for ds_idx, (ds, values_by_rep) in enumerate(dataset_values.items()):
        link_color = dataset_colors[ds]

        for metric in METRICS:
            vmin, vmax = metric_ranges[metric]

            for a, b in [("QTN", "GAF"), ("QTN", "MTF"), ("GAF", "MTF")]:
                th_a_base = block_pos.get((a, metric), None)
                th_b_base = block_pos.get((b, metric), None)
                if th_a_base is None or th_b_base is None:
                    continue

                sc = metric_pair_agreement(
                    values_by_rep[a].get(metric, np.nan),
                    values_by_rep[b].get(metric, np.nan),
                    vmin, vmax
                )
                if not np.isfinite(sc):
                    continue
                if sc < AGREEMENT_THRESHOLD:
                    continue

                start_a, end_a = sector_ranges[a]
                start_b, end_b = sector_ranges[b]
                half_a = metric_block_halfwidth(start_a, end_a, len(METRICS))
                half_b = metric_block_halfwidth(start_b, end_b, len(METRICS))

                th_a = jittered_angle(th_a_base, ds_idx, n_datasets, half_a)
                th_b = jittered_angle(th_b_base, ds_idx, n_datasets, half_b)

                lw = LW_MIN + LW_SCALE * (sc / max_score)
                alpha = ALPHA_MIN + ALPHA_SCALE * (sc / max_score)

                path = bezier_link(th_a, th_b, r=R_LINK, bend=LINK_BEND)
                ax.add_patch(
                    PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=link_color,
                        lw=lw,
                        alpha=alpha,
                        capstyle="round"
                    )
                )

    legend_handles = [
        Line2D([0], [0], color=dataset_colors[ds], lw=3, label=DATASET_LABELS.get(ds, ds))
        for ds in dataset_values.keys()
    ]

    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        title="Datasets",
        title_fontsize=11,
        ncol=1
    )

    ax.set_title(
        f"High-agreement links across QG, GAF, and MTF\n"
        f"(dataset-colored links; shown only if agreement ≥ {AGREEMENT_THRESHOLD:.2f})",
        fontsize=16,
        pad=18
    )

    fig.savefig(out_png, dpi=DPI_EXPORT, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# MAIN
# ============================================================
def main():
    grouped, rep_tables = build_rep_tables(INPUT_DIR, METRICS)

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    metric_ranges = build_global_metric_ranges(rep_tables)

    rep_lookup = {}
    for rep in REP_ORDER:
        df = rep_tables[rep].copy()
        if not df.empty:
            df["dataset"] = df["dataset"].astype(str)
            rep_lookup[rep] = df.set_index("dataset").to_dict(orient="index")
        else:
            rep_lookup[rep] = {}

    common = set(rep_lookup["QTN"].keys()) & set(rep_lookup["GAF"].keys()) & set(rep_lookup["MTF"].keys())
    common = reorder_datasets(list(common))

    dataset_values_full = {
        ds: {
            "QTN": rep_lookup["QTN"][ds],
            "GAF": rep_lookup["GAF"][ds],
            "MTF": rep_lookup["MTF"][ds],
        }
        for ds in common
    }

    summary_df = build_dataset_summary(dataset_values_full, metric_ranges)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    if TOP_DATASETS is not None and TOP_DATASETS < len(summary_df):
        keep = summary_df["dataset"].head(TOP_DATASETS).tolist()
        dataset_values = {ds: dataset_values_full[ds] for ds in keep}
    else:
        dataset_values = dataset_values_full

    link_df = build_link_table(dataset_values, metric_ranges)
    link_df = link_df[link_df["agreement_score"].ge(AGREEMENT_THRESHOLD, fill_value=False)].copy()
    link_df = link_df.sort_values(["agreement_score", "dataset", "metric"], ascending=[False, True, True])
    link_df.to_csv(OUT_LINKS, index=False)

    dataset_colors = build_dataset_colors(list(dataset_values.keys()))

    plot_all_datasets_metric_circos(
        dataset_values=dataset_values,
        metric_ranges=metric_ranges,
        dataset_colors=dataset_colors,
        out_png=OUT_PNG,
        out_pdf=OUT_PDF
    )

    print("[OK] Saved:")
    print(" -", OUT_PNG.resolve())
    print(" -", OUT_PDF.resolve())
    print(" -", OUT_SUMMARY.resolve())
    print(" -", OUT_LINKS.resolve())
    print(f"[INFO] Agreement threshold used: {AGREEMENT_THRESHOLD:.2f}")

if __name__ == "__main__":
    main()

# In[ ]:
