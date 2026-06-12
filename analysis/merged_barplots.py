"""Merged bar-plot summaries across datasets/representations (supporting figure).

Provenance: extracted verbatim from the notebook ``merged-barplots.ipynb``.
"""

import os
import re
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------
# compatibility for older packages
# ---------------------------------------
np.float = float
np.int = int
np.object = object
np.bool = bool

# ---------------------------------------
# CONFIG
# ---------------------------------------
INPUT_DIR = "csv"                      # folder with metrics_*.csv
OUT_ROOT = "plots_all_datasets_boxplots"

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

METRICS_TO_PLOT = {
    "sigma_small_world": {
        "ylabel": r"Small-worldness $\sigma$",
        "threshold": 1.0,
        "threshold_label": r"$\sigma$ = 1.0",
        "group": "small_world"
    },
    "gamma_C_over_Crand": {
        "ylabel": r"$\gamma = C/C_{rand}$",
        "threshold": 1.0,
        "threshold_label": r"$\gamma$ = 1.0",
        "group": "small_world"
    },
    "lambda_L_over_Lrand": {
        "ylabel": r"$\lambda = L/L_{rand}$",
        "threshold": 1.0,
        "threshold_label": r"$\lambda$ = 1.0",
        "group": "small_world"
    },
    "transitivity": {
        "ylabel": "Transitivity",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "char_path_len_gcc": {
        "ylabel": "Characteristic path length",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "global_efficiency": {
        "ylabel": "Global efficiency",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "zC": {
        "ylabel": r"$z_C$",
        "threshold": 0.0,
        "threshold_label": r"$z_C$ = 0.0",
        "group": "null_model"
    },
    "zL": {
        "ylabel": r"$z_L$",
        "threshold": 0.0,
        "threshold_label": r"$z_L$ = 0.0",
        "group": "null_model"
    },
    "density": {
        "ylabel": "Density",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "n_nodes": {
        "ylabel": "Number of nodes",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
}

DPI = 1000
BOX_WIDTH = 0.72
SHOW_POINTS = True
POINT_SIZE = 2.2
POINT_ALPHA = 0.35
ROTATION = 35

sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12
})

# ---------------------------------------
# HELPERS
# ---------------------------------------
def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())

def extract_rep_dataset(filepath: str):
    """
    Expected files:
    metrics_QTN_ABIDE.csv
    metrics_GAF_ABIDE.csv
    metrics_MTF_ABIDE.csv
    """
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    rep = m.group(1).upper()
    dataset = m.group(2)
    return rep, dataset

def normalize_key(s: str) -> str:
    return str(s).strip()

def get_id_series(df: pd.DataFrame, rep: str) -> pd.Series:
    """
    Broad ID detection so different datasets are not skipped.
    """
    id_candidates = [
        "patient_id",
        "subject_id",
        "file_id",
        "simulation_id",
        "div",
        "record_id",
        "sample_id",
        "id"
    ]

    for col in id_candidates:
        if col in df.columns:
            return df[col].astype(str).map(normalize_key)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        return df[object_cols[0]].astype(str).map(normalize_key)

    return pd.Series([f"{rep}_{i}" for i in range(len(df))], name="auto_id")

def load_metric_table(path: str, rep: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if metric not in df.columns:
        return pd.DataFrame()

    ids = get_id_series(df, rep)

    out = pd.DataFrame({
        "key": ids,
        rep: pd.to_numeric(df[metric], errors="coerce")
    })

    out = out.dropna(subset=[rep])
    out = out.groupby("key", as_index=False)[rep].mean(numeric_only=True)
    return out

def merge_three_reps(qtn: pd.DataFrame, gaf: pd.DataFrame, mtf: pd.DataFrame,
                     dataset_name: str = "", metric: str = "") -> pd.DataFrame:
    if qtn.empty or gaf.empty or mtf.empty:
        return pd.DataFrame()

    qset = set(qtn["key"])
    gset = set(gaf["key"])
    mset = set(mtf["key"])

    print(
        f"[DEBUG] {dataset_name} | {metric} | "
        f"QTN={len(qset)} GAF={len(gset)} MTF={len(mset)} | "
        f"Q∩G={len(qset & gset)} Q∩M={len(qset & mset)} G∩M={len(gset & mset)} | "
        f"Q∩G∩M={len(qset & gset & mset)}"
    )

    wide = qtn.merge(gaf, on="key", how="inner").merge(mtf, on="key", how="inner")
    wide = wide[["key", "QTN", "GAF", "MTF"]].dropna()
    return wide

def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars=["key"],
        value_vars=REP_ORDER,
        var_name="representation",
        value_name="value"
    )
    long["representation"] = pd.Categorical(
        long["representation"],
        categories=REP_ORDER,
        ordered=True
    )
    return long

def pretty_dataset_name(ds: str) -> str:
    """
    Optional prettier labels.
    You can customize this mapping if you want.
    """
    custom = {
        "ABIDE": "ABIDE",
        "COBRE": "COBRE",
        "ADHD": "ADHD-200",
        "ADHD200": "ADHD-200",
        "MEA": "MEA",
        "CALCIUM": "Calcium",
        "ECG": "ECG",
        "MEG": "MEG",
        "EEG": "EEG",
    }
    ds_clean = ds.strip()
    if ds_clean in custom:
        return custom[ds_clean]
    return ds_clean.replace("_", " ").replace("-", " ")

def collect_all_long_data(grouped: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - all_long: matched long-format table for plotting
      - summary_df: processing summary
    """
    all_rows = []
    summary_rows = []

    for dataset_name, rep_files in grouped.items():
        if not all(rep in rep_files for rep in REP_ORDER):
            print(f"[SKIP] {dataset_name}: missing one of QTN/GAF/MTF")
            continue

        print(f"\n===== DATASET: {dataset_name} =====")

        for metric, cfg in METRICS_TO_PLOT.items():
            qtn = load_metric_table(rep_files["QTN"], "QTN", metric)
            gaf = load_metric_table(rep_files["GAF"], "GAF", metric)
            mtf = load_metric_table(rep_files["MTF"], "MTF", metric)

            wide = merge_three_reps(qtn, gaf, mtf, dataset_name, metric)

            if wide.empty:
                print(f"[SKIP] {dataset_name} | {metric}: no matched subjects")
                summary_rows.append({
                    "dataset": dataset_name,
                    "metric": metric,
                    "group": cfg["group"],
                    "n_matched": 0
                })
                continue

            long = to_long(wide).copy()
            long["dataset"] = dataset_name
            long["dataset_display"] = pretty_dataset_name(dataset_name)
            long["metric"] = metric
            long["metric_label"] = cfg["ylabel"]
            long["group"] = cfg["group"]

            all_rows.append(long)

            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "group": cfg["group"],
                "n_matched": len(wide)
            })

            print(f"[OK] {dataset_name} | {metric} | matched={len(wide)}")

    all_long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return all_long, summary_df

def add_threshold(ax, metric: str):
    cfg = METRICS_TO_PLOT[metric]
    thr = cfg["threshold"]
    if thr is not None:
        ax.axhline(thr, ls="--", lw=1.1, color="black", alpha=0.9, zorder=1)

def plot_combined_grid(all_long: pd.DataFrame, out_dir: Path):
    """
    One big figure:
    each subplot = one metric
    x-axis = datasets
    hue = representation
    """
    if all_long.empty:
        print("[WARN] No matched data available to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [m for m in METRICS_TO_PLOT if m in all_long["metric"].unique()]
    dataset_order = sorted(
        all_long["dataset_display"].dropna().unique().tolist(),
        key=lambda x: x.lower()
    )

    n_metrics = len(metric_order)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(16, len(dataset_order) * 1.2), nrows * 5.2),
        squeeze=False
    )
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sub = all_long[all_long["metric"] == metric].copy()

        sns.boxplot(
            data=sub,
            x="dataset_display",
            y="value",
            hue="representation",
            order=dataset_order,
            hue_order=REP_ORDER,
            palette=PALETTE_REP,
            width=BOX_WIDTH,
            showfliers=False,
            linewidth=1.1,
            ax=ax
        )

        if SHOW_POINTS:
            sns.stripplot(
                data=sub,
                x="dataset_display",
                y="value",
                hue="representation",
                order=dataset_order,
                hue_order=REP_ORDER,
                dodge=True,
                size=POINT_SIZE,
                alpha=POINT_ALPHA,
                color="black",
                ax=ax
            )

        add_threshold(ax, metric)

        ax.set_title(METRICS_TO_PLOT[metric]["ylabel"], fontsize=14, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=ROTATION)

        # keep only one legend for the whole figure
        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None and legend_labels is None:
            # first 3 correspond to boxplot hue entries
            legend_handles = handles[:3]
            legend_labels = [REP_DISPLAY[r] for r in REP_ORDER]

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        sns.despine(ax=ax, trim=True)

    # remove empty subplots
    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Representation",
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=13,
            title_fontsize=14
        )

    fig.suptitle("All datasets across representations", y=1.04, fontsize=18)
    plt.tight_layout()

    out_png = out_dir / "all_metrics_grouped_boxplots_by_dataset.png"
    out_pdf = out_dir / "all_metrics_grouped_boxplots_by_dataset.pdf"
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")

def plot_one_metric_per_figure(all_long: pd.DataFrame, out_dir: Path):
    """
    Optional: one cleaner figure per metric
    """
    if all_long.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_order = sorted(
        all_long["dataset_display"].dropna().unique().tolist(),
        key=lambda x: x.lower()
    )

    for metric in [m for m in METRICS_TO_PLOT if m in all_long["metric"].unique()]:
        sub = all_long[all_long["metric"] == metric].copy()

        fig, ax = plt.subplots(figsize=(max(10, len(dataset_order) * 1.2), 6))

        sns.boxplot(
            data=sub,
            x="dataset_display",
            y="value",
            hue="representation",
            order=dataset_order,
            hue_order=REP_ORDER,
            palette=PALETTE_REP,
            width=BOX_WIDTH,
            showfliers=False,
            linewidth=1.1,
            ax=ax
        )

        if SHOW_POINTS:
            sns.stripplot(
                data=sub,
                x="dataset_display",
                y="value",
                hue="representation",
                order=dataset_order,
                hue_order=REP_ORDER,
                dodge=True,
                size=POINT_SIZE,
                alpha=POINT_ALPHA,
                color="black",
                ax=ax
            )

        add_threshold(ax, metric)

        ax.set_xlabel("")
        ax.set_ylabel(METRICS_TO_PLOT[metric]["ylabel"])
        ax.tick_params(axis="x", rotation=ROTATION)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:3],
            [REP_DISPLAY[r] for r in REP_ORDER],
            title="Representation",
            frameon=False,
            loc="best"
        )

        sns.despine(ax=ax, trim=True)
        plt.tight_layout()

        base = sanitize_name(metric)
        out_png = out_dir / f"{base}_grouped_boxplots_by_dataset.png"
        out_pdf = out_dir / f"{base}_grouped_boxplots_by_dataset.pdf"
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_png}")
        print(f"[SAVED] {out_pdf}")

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir.resolve()}")

    all_files = sorted(glob.glob(str(input_dir / "metrics_*.csv")))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir.resolve()}")

    grouped = {}
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue
        rep, dataset = parsed
        grouped.setdefault(dataset, {})
        grouped[dataset][rep] = f

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # collect all matched long-format data
    all_long, summary_df = collect_all_long_data(grouped)

    if all_long.empty:
        raise RuntimeError("No matched data were found across QTN/GAF/MTF.")

    # save tables
    all_long["representation_display"] = all_long["representation"].map(REP_DISPLAY)
    all_long.to_csv(out_root / "all_metrics_matched_long.csv", index=False)
    summary_df.sort_values(["group", "metric", "dataset"]).to_csv(
        out_root / "summary_processed_metrics.csv", index=False
    )

    # plots
    plot_combined_grid(all_long, out_root / "combined")
    plot_one_metric_per_figure(all_long, out_root / "one_metric_per_figure")

    print(f"\n[DONE] Outputs saved in: {out_root.resolve()}")

if __name__ == "__main__":
    main()

# %% ---- next notebook cell ----

import os
import re
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------
# compatibility for older packages
# ---------------------------------------
np.float = float
np.int = int
np.object = object
np.bool = bool

# ---------------------------------------
# CONFIG
# ---------------------------------------
INPUT_DIR = "csv"
OUT_ROOT = "plots_all_datasets_boxplots"

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

METRICS_TO_PLOT = {
    "sigma_small_world": {
        "ylabel": r"Small-worldness $\sigma$",
        "threshold": 1.0,
        "threshold_label": r"$\sigma$ = 1.0",
        "group": "small_world"
    },
    "gamma_C_over_Crand": {
        "ylabel": r"$\gamma = C/C_{rand}$",
        "threshold": 1.0,
        "threshold_label": r"$\gamma$ = 1.0",
        "group": "small_world"
    },
    "lambda_L_over_Lrand": {
        "ylabel": r"$\lambda = L/L_{rand}$",
        "threshold": 1.0,
        "threshold_label": r"$\lambda$ = 1.0",
        "group": "small_world"
    },
    "transitivity": {
        "ylabel": "Transitivity",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "char_path_len_gcc": {
        "ylabel": "Characteristic path length",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "global_efficiency": {
        "ylabel": "Global efficiency",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "zC": {
        "ylabel": r"$z_C$",
        "threshold": 0.0,
        "threshold_label": r"$z_C$ = 0.0",
        "group": "null_model"
    },
    "zL": {
        "ylabel": r"$z_L$",
        "threshold": 0.0,
        "threshold_label": r"$z_L$ = 0.0",
        "group": "null_model"
    },
    "density": {
        "ylabel": "Density",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "n_nodes": {
        "ylabel": "Number of nodes",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
}

DPI = 1000
BOX_WIDTH = 0.72
SHOW_POINTS = True
POINT_SIZE = 2.2
POINT_ALPHA = 0.35
ROTATION = 35

# separator style
SHOW_DATASET_SEPARATORS = True
SHOW_MODALITY_SEPARATORS = True
DATASET_SEPARATOR_COLOR = "0.90"
MODALITY_SEPARATOR_COLOR = "0.72"
DATASET_SEPARATOR_LW = 0.8
MODALITY_SEPARATOR_LW = 1.5

sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12
})

# ---------------------------------------
# HELPERS
# ---------------------------------------
def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())

def extract_rep_dataset(filepath: str):
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    rep = m.group(1).upper()
    dataset = m.group(2)
    return rep, dataset

def normalize_key(s: str) -> str:
    return str(s).strip()

def get_id_series(df: pd.DataFrame, rep: str) -> pd.Series:
    id_candidates = [
        "patient_id",
        "subject_id",
        "file_id",
        "simulation_id",
        "div",
        "record_id",
        "sample_id",
        "id"
    ]

    for col in id_candidates:
        if col in df.columns:
            return df[col].astype(str).map(normalize_key)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        return df[object_cols[0]].astype(str).map(normalize_key)

    return pd.Series([f"{rep}_{i}" for i in range(len(df))], name="auto_id")

def load_metric_table(path: str, rep: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if metric not in df.columns:
        return pd.DataFrame()

    ids = get_id_series(df, rep)

    out = pd.DataFrame({
        "key": ids,
        rep: pd.to_numeric(df[metric], errors="coerce")
    })

    out = out.dropna(subset=[rep])
    out = out.groupby("key", as_index=False)[rep].mean(numeric_only=True)
    return out

def merge_three_reps(qtn: pd.DataFrame, gaf: pd.DataFrame, mtf: pd.DataFrame,
                     dataset_name: str = "", metric: str = "") -> pd.DataFrame:
    if qtn.empty or gaf.empty or mtf.empty:
        return pd.DataFrame()

    qset = set(qtn["key"])
    gset = set(gaf["key"])
    mset = set(mtf["key"])

    print(
        f"[DEBUG] {dataset_name} | {metric} | "
        f"QTN={len(qset)} GAF={len(gset)} MTF={len(mset)} | "
        f"Q∩G={len(qset & gset)} Q∩M={len(qset & mset)} G∩M={len(gset & mset)} | "
        f"Q∩G∩M={len(qset & gset & mset)}"
    )

    wide = qtn.merge(gaf, on="key", how="inner").merge(mtf, on="key", how="inner")
    wide = wide[["key", "QTN", "GAF", "MTF"]].dropna()
    return wide

def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars=["key"],
        value_vars=REP_ORDER,
        var_name="representation",
        value_name="value"
    )
    long["representation"] = pd.Categorical(
        long["representation"],
        categories=REP_ORDER,
        ordered=True
    )
    return long

def pretty_dataset_name(ds: str) -> str:
    custom = {
        "ABIDE": "ABIDE",
        "COBRE": "COBRE",
        "ADHD": "ADHD-200",
        "ADHD200": "ADHD-200",
        "SCZ": "SCZ",
        "MEA": "MEA",
        "CALCIUM": "Calcium",
        "ECG": "ECG",
        "MEG": "MEG",
        "EEG": "EEG",
    }
    ds_clean = ds.strip()
    if ds_clean in custom:
        return custom[ds_clean]
    return ds_clean.replace("_", " ").replace("-", " ")

# ---------------------------------------
# DATASET GROUPING / ORDER
# ---------------------------------------
MODALITY_ORDER = [
    "fMRI",
    "Calcium imaging",
    "Sleep recordings",
    "MEG",
    "EMG",
    "ECG",
    "MEA",
    "Respiration",
    "Other"
]

# force some important datasets to appear together and in a sensible order
WITHIN_MODALITY_PRIORITY = {
    "ABIDE": 0,
    "ADHD-200": 1,
    "COBRE": 2,
    "SCZ": 3,
}

def infer_modality(display_name: str) -> str:
    s = str(display_name).lower()

    if any(k in s for k in ["abide", "adhd", "cobre", "scz"]):
        return "fMRI"

    if any(k in s for k in ["cafast", "calcium"]):
        return "Calcium imaging"

    if "cap sleep" in s:
        return "Sleep recordings"

    if "meg" in s:
        return "MEG"

    if "emg plantar" in s:
        return "EMG"

    if any(k in s for k in ["fantasia", "nsrdb", "ecg"]):
        return "ECG"

    if "mea" in s:
        return "MEA"

    if "resp aeration" in s:
        return "Respiration"

    return "Other"

def build_dataset_layout(all_long: pd.DataFrame):
    dataset_labels = pd.unique(all_long["dataset_display"].dropna())

    rows = []
    for label in dataset_labels:
        modality = infer_modality(label)
        rows.append({
            "dataset_display": label,
            "modality": modality,
            "mod_rank": MODALITY_ORDER.index(modality) if modality in MODALITY_ORDER else len(MODALITY_ORDER),
            "within_rank": WITHIN_MODALITY_PRIORITY.get(label, 999),
            "label_sort": str(label).lower()
        })

    meta = pd.DataFrame(rows).sort_values(
        ["mod_rank", "within_rank", "label_sort"]
    ).reset_index(drop=True)

    dataset_order = meta["dataset_display"].tolist()
    modality_per_dataset = dict(zip(meta["dataset_display"], meta["modality"]))

    boundaries = []
    if len(meta) > 1:
        current_mod = meta.loc[0, "modality"]
        for i in range(1, len(meta)):
            this_mod = meta.loc[i, "modality"]
            if this_mod != current_mod:
                boundaries.append(i - 0.5)
                current_mod = this_mod

    return dataset_order, boundaries, modality_per_dataset, meta

def add_threshold(ax, metric: str):
    cfg = METRICS_TO_PLOT[metric]
    thr = cfg["threshold"]
    if thr is not None:
        ax.axhline(thr, ls="--", lw=1.1, color="black", alpha=0.9, zorder=1)

def add_vertical_separators(ax, dataset_order, modality_boundaries):
    # light separator between every dataset
    if SHOW_DATASET_SEPARATORS:
        for i in range(len(dataset_order) - 1):
            ax.axvline(
                i + 0.5,
                color=DATASET_SEPARATOR_COLOR,
                lw=DATASET_SEPARATOR_LW,
                zorder=0
            )

    # stronger separator between modality groups
    if SHOW_MODALITY_SEPARATORS:
        for x in modality_boundaries:
            ax.axvline(
                x,
                color=MODALITY_SEPARATOR_COLOR,
                lw=MODALITY_SEPARATOR_LW,
                zorder=0
            )

def collect_all_long_data(grouped: dict):
    all_rows = []
    summary_rows = []

    for dataset_name, rep_files in grouped.items():
        if not all(rep in rep_files for rep in REP_ORDER):
            print(f"[SKIP] {dataset_name}: missing one of QTN/GAF/MTF")
            continue

        print(f"\n===== DATASET: {dataset_name} =====")

        for metric, cfg in METRICS_TO_PLOT.items():
            qtn = load_metric_table(rep_files["QTN"], "QTN", metric)
            gaf = load_metric_table(rep_files["GAF"], "GAF", metric)
            mtf = load_metric_table(rep_files["MTF"], "MTF", metric)

            wide = merge_three_reps(qtn, gaf, mtf, dataset_name, metric)

            if wide.empty:
                print(f"[SKIP] {dataset_name} | {metric}: no matched subjects")
                summary_rows.append({
                    "dataset": dataset_name,
                    "metric": metric,
                    "group": cfg["group"],
                    "n_matched": 0
                })
                continue

            long = to_long(wide).copy()
            long["dataset"] = dataset_name
            long["dataset_display"] = pretty_dataset_name(dataset_name)
            long["metric"] = metric
            long["metric_label"] = cfg["ylabel"]
            long["group"] = cfg["group"]

            all_rows.append(long)

            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "group": cfg["group"],
                "n_matched": len(wide)
            })

            print(f"[OK] {dataset_name} | {metric} | matched={len(wide)}")

    all_long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return all_long, summary_df

def plot_combined_grid(all_long: pd.DataFrame, out_dir: Path):
    if all_long.empty:
        print("[WARN] No matched data available to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [m for m in METRICS_TO_PLOT if m in all_long["metric"].unique()]
    dataset_order, modality_boundaries, _, _ = build_dataset_layout(all_long)

    n_metrics = len(metric_order)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(18, len(dataset_order) * 1.45), nrows * 5.2),
        squeeze=False
    )
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sub = all_long[all_long["metric"] == metric].copy()

        sns.boxplot(
            data=sub,
            x="dataset_display",
            y="value",
            hue="representation",
            order=dataset_order,
            hue_order=REP_ORDER,
            palette=PALETTE_REP,
            width=BOX_WIDTH,
            showfliers=False,
            linewidth=1.1,
            ax=ax
        )

        if SHOW_POINTS:
            sns.stripplot(
                data=sub,
                x="dataset_display",
                y="value",
                hue="representation",
                order=dataset_order,
                hue_order=REP_ORDER,
                dodge=True,
                size=POINT_SIZE,
                alpha=POINT_ALPHA,
                color="black",
                ax=ax
            )

        add_vertical_separators(ax, dataset_order, modality_boundaries)
        add_threshold(ax, metric)

        ax.set_title(METRICS_TO_PLOT[metric]["ylabel"], fontsize=14, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=ROTATION, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None and legend_labels is None:
            legend_handles = handles[:3]
            legend_labels = [REP_DISPLAY[r] for r in REP_ORDER]

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        sns.despine(ax=ax, trim=True)

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Representation",
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=13,
            title_fontsize=14
        )

    fig.suptitle("All datasets across representations", y=1.04, fontsize=18)
    plt.tight_layout()

    out_png = out_dir / "all_metrics_grouped_boxplots_by_dataset.png"
    out_pdf = out_dir / "all_metrics_grouped_boxplots_by_dataset.pdf"
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")

def plot_one_metric_per_figure(all_long: pd.DataFrame, out_dir: Path):
    if all_long.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_order, modality_boundaries, _, _ = build_dataset_layout(all_long)

    for metric in [m for m in METRICS_TO_PLOT if m in all_long["metric"].unique()]:
        sub = all_long[all_long["metric"] == metric].copy()

        fig, ax = plt.subplots(figsize=(max(12, len(dataset_order) * 1.45), 6))

        sns.boxplot(
            data=sub,
            x="dataset_display",
            y="value",
            hue="representation",
            order=dataset_order,
            hue_order=REP_ORDER,
            palette=PALETTE_REP,
            width=BOX_WIDTH,
            showfliers=False,
            linewidth=1.1,
            ax=ax
        )

        if SHOW_POINTS:
            sns.stripplot(
                data=sub,
                x="dataset_display",
                y="value",
                hue="representation",
                order=dataset_order,
                hue_order=REP_ORDER,
                dodge=True,
                size=POINT_SIZE,
                alpha=POINT_ALPHA,
                color="black",
                ax=ax
            )

        add_vertical_separators(ax, dataset_order, modality_boundaries)
        add_threshold(ax, metric)

        ax.set_xlabel("")
        ax.set_ylabel(METRICS_TO_PLOT[metric]["ylabel"])
        plt.setp(ax.get_xticklabels(), rotation=ROTATION, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:3],
            [REP_DISPLAY[r] for r in REP_ORDER],
            title="Representation",
            frameon=False,
            loc="best"
        )

        sns.despine(ax=ax, trim=True)
        plt.tight_layout()

        base = sanitize_name(metric)
        out_png = out_dir / f"{base}_grouped_boxplots_by_dataset.png"
        out_pdf = out_dir / f"{base}_grouped_boxplots_by_dataset.pdf"
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_png}")
        print(f"[SAVED] {out_pdf}")

# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir.resolve()}")

    all_files = sorted(glob.glob(str(input_dir / "metrics_*.csv")))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir.resolve()}")

    grouped = {}
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue
        rep, dataset = parsed
        grouped.setdefault(dataset, {})
        grouped[dataset][rep] = f

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    all_long, summary_df = collect_all_long_data(grouped)

    if all_long.empty:
        raise RuntimeError("No matched data were found across QTN/GAF/MTF.")

    all_long["representation_display"] = all_long["representation"].map(REP_DISPLAY)
    all_long.to_csv(out_root / "all_metrics_matched_long.csv", index=False)
    summary_df.sort_values(["group", "metric", "dataset"]).to_csv(
        out_root / "summary_processed_metrics.csv", index=False
    )

    plot_combined_grid(all_long, out_root / "combined")
    plot_one_metric_per_figure(all_long, out_root / "one_metric_per_figure")

    print(f"\n[DONE] Outputs saved in: {out_root.resolve()}")

if __name__ == "__main__":
    main()

# %% ---- next notebook cell ----

import os
import re
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------
# compatibility for older packages
# ---------------------------------------
np.float = float
np.int = int
np.object = object
np.bool = bool

# ---------------------------------------
# CONFIG
# ---------------------------------------
INPUT_DIR = "csv"
OUT_ROOT = "plots_all_datasets_boxplots"

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

METRICS_TO_PLOT = {
    "sigma_small_world": {
        "ylabel": r"Small-worldness $\sigma$",
        "threshold": 1.0,
        "threshold_label": r"$\sigma$ = 1.0",
        "group": "small_world"
    },
    "gamma_C_over_Crand": {
        "ylabel": r"$\gamma = C/C_{rand}$",
        "threshold": 1.0,
        "threshold_label": r"$\gamma$ = 1.0",
        "group": "small_world"
    },
    "lambda_L_over_Lrand": {
        "ylabel": r"$\lambda = L/L_{rand}$",
        "threshold": 1.0,
        "threshold_label": r"$\lambda$ = 1.0",
        "group": "small_world"
    },
    "transitivity": {
        "ylabel": "Transitivity",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "char_path_len_gcc": {
        "ylabel": "Characteristic path length",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "global_efficiency": {
        "ylabel": "Global efficiency",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "zC": {
        "ylabel": r"$z_C$",
        "threshold": 0.0,
        "threshold_label": r"$z_C$ = 0.0",
        "group": "null_model"
    },
    "zL": {
        "ylabel": r"$z_L$",
        "threshold": 0.0,
        "threshold_label": r"$z_L$ = 0.0",
        "group": "null_model"
    },
    "density": {
        "ylabel": "Density",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
    "n_nodes": {
        "ylabel": "Number of nodes",
        "threshold": None,
        "threshold_label": None,
        "group": "topology"
    },
}

DPI = 1000
BOX_WIDTH = 0.72
SHOW_POINTS = True
POINT_SIZE = 2.2
POINT_ALPHA = 0.35
ROTATION = 35

# separator style
SHOW_DATASET_SEPARATORS = True
SHOW_MODALITY_SEPARATORS = True
DATASET_SEPARATOR_COLOR = "0.90"
MODALITY_SEPARATOR_COLOR = "0.72"
DATASET_SEPARATOR_LW = 0.8
MODALITY_SEPARATOR_LW = 1.5

sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 12
})

# ---------------------------------------
# HELPERS
# ---------------------------------------
def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s).strip())


def extract_rep_dataset(filepath: str):
    base = os.path.basename(filepath)
    m = re.match(r"metrics_(QTN|GAF|MTF)_(.+)\.csv$", base, flags=re.IGNORECASE)
    if not m:
        return None
    rep = m.group(1).upper()
    dataset = m.group(2)
    return rep, dataset


def normalize_key(s: str) -> str:
    return str(s).strip()


def get_id_series(df: pd.DataFrame, rep: str) -> pd.Series:
    id_candidates = [
        "patient_id",
        "subject_id",
        "file_id",
        "simulation_id",
        "div",
        "record_id",
        "sample_id",
        "id"
    ]

    for col in id_candidates:
        if col in df.columns:
            return df[col].astype(str).map(normalize_key)

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        return df[object_cols[0]].astype(str).map(normalize_key)

    return pd.Series([f"{rep}_{i}" for i in range(len(df))], name="auto_id")


def load_metric_table(path: str, rep: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if metric not in df.columns:
        return pd.DataFrame()

    ids = get_id_series(df, rep)

    out = pd.DataFrame({
        "key": ids,
        rep: pd.to_numeric(df[metric], errors="coerce")
    })

    out = out.dropna(subset=[rep])
    out = out.groupby("key", as_index=False)[rep].mean(numeric_only=True)
    return out


def merge_three_reps(
    qtn: pd.DataFrame,
    gaf: pd.DataFrame,
    mtf: pd.DataFrame,
    dataset_name: str = "",
    metric: str = ""
) -> pd.DataFrame:
    if qtn.empty or gaf.empty or mtf.empty:
        return pd.DataFrame()

    qset = set(qtn["key"])
    gset = set(gaf["key"])
    mset = set(mtf["key"])

    print(
        f"[DEBUG] {dataset_name} | {metric} | "
        f"QTN={len(qset)} GAF={len(gset)} MTF={len(mset)} | "
        f"Q∩G={len(qset & gset)} Q∩M={len(qset & mset)} G∩M={len(gset & mset)} | "
        f"Q∩G∩M={len(qset & gset & mset)}"
    )

    wide = qtn.merge(gaf, on="key", how="inner").merge(mtf, on="key", how="inner")
    wide = wide[["key", "QTN", "GAF", "MTF"]].dropna()
    return wide


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars=["key"],
        value_vars=REP_ORDER,
        var_name="representation",
        value_name="value"
    )
    long["representation"] = pd.Categorical(
        long["representation"],
        categories=REP_ORDER,
        ordered=True
    )
    return long


def pretty_dataset_name(ds: str) -> str:
    ds_clean = str(ds).strip()

    custom = {
        # fMRI
        "ABIDE": "ABIDE",
        "ADHD": "ADHD",
        "ADHD200": "ADHD",
        "SCZ": "SCZ",
        "COBRE": "SCZ",

        # calcium / MEA
        "CaFast_Sham_byDIV": "Calcium",
        "CALCIUM": "Calcium",
        "MEArecs": "MEA",
        "MEA": "MEA",

        # sleep
        "cap_sleep_controls_ALL_SLEEP_NREM": "Sleep-All",
        "cap_sleep_controls_EEG_NREM": "Sleep-EEG",
        "cap_sleep_controls_EMG_NREM": "Sleep-EMG",
        "cap_sleep_controls_RESP_NREM": "Sleep-Resp",

        # MEG / EMG
        "ds000117_raw_meg_per_subject": "MEG",
        "MEG": "MEG",
        "emg_plantar_EMGONLY_per_subject": "EMG",
        "EMG": "EMG",

        # ECG
        "fantasia_all": "ECG-Fantasia",
        "nsrdb": "ECG-NSRDB",

        # respiration
        "resp_aeration_stream_per_subject": "Resp",
        "RESP": "Resp",
    }

    return custom.get(ds_clean, ds_clean)


# ---------------------------------------
# DATASET GROUPING / ORDER
# ---------------------------------------
MODALITY_ORDER = [
    "fMRI",
    "Calcium imaging",
    "Sleep recordings",
    "MEG",
    "EMG",
    "ECG",
    "MEA",
    "Respiration",
    "Other"
]

WITHIN_MODALITY_PRIORITY = {
    "ABIDE": 0,
    "ADHD": 1,
    "SCZ": 2,
}


def infer_modality(display_name: str) -> str:
    s = str(display_name).lower()

    if any(k in s for k in ["abide", "adhd", "cobre", "scz"]):
        return "fMRI"

    if any(k in s for k in ["cafast", "calcium"]):
        return "Calcium imaging"

    if any(k in s for k in ["cap sleep", "sleep-all", "sleep-eeg", "sleep-emg", "sleep-resp"]):
        return "Sleep recordings"

    if s == "meg" or "meg" in s:
        return "MEG"

    if s == "emg" or "emg plantar" in s:
        return "EMG"

    if any(k in s for k in ["fantasia", "nsrdb", "ecg"]):
        return "ECG"

    if s == "mea" or "mea" in s:
        return "MEA"

    if s == "resp" or "resp aeration" in s:
        return "Respiration"

    return "Other"


def build_dataset_layout(all_long: pd.DataFrame):
    dataset_labels = pd.unique(all_long["dataset_display"].dropna())

    rows = []
    for label in dataset_labels:
        modality = infer_modality(label)
        rows.append({
            "dataset_display": label,
            "modality": modality,
            "mod_rank": MODALITY_ORDER.index(modality) if modality in MODALITY_ORDER else len(MODALITY_ORDER),
            "within_rank": WITHIN_MODALITY_PRIORITY.get(label, 999),
            "label_sort": str(label).lower()
        })

    meta = pd.DataFrame(rows).sort_values(
        ["mod_rank", "within_rank", "label_sort"]
    ).reset_index(drop=True)

    dataset_order = meta["dataset_display"].tolist()
    modality_per_dataset = dict(zip(meta["dataset_display"], meta["modality"]))

    boundaries = []
    if len(meta) > 1:
        current_mod = meta.loc[0, "modality"]
        for i in range(1, len(meta)):
            this_mod = meta.loc[i, "modality"]
            if this_mod != current_mod:
                boundaries.append(i - 0.5)
                current_mod = this_mod

    return dataset_order, boundaries, modality_per_dataset, meta


def add_threshold(ax, metric: str):
    cfg = METRICS_TO_PLOT[metric]
    thr = cfg["threshold"]
    if thr is not None:
        ax.axhline(thr, ls="--", lw=1.1, color="black", alpha=0.9, zorder=1)


def add_vertical_separators(ax, dataset_order, modality_boundaries):
    if SHOW_DATASET_SEPARATORS:
        for i in range(len(dataset_order) - 1):
            ax.axvline(
                i + 0.5,
                color=DATASET_SEPARATOR_COLOR,
                lw=DATASET_SEPARATOR_LW,
                zorder=0
            )

    if SHOW_MODALITY_SEPARATORS:
        for x in modality_boundaries:
            ax.axvline(
                x,
                color=MODALITY_SEPARATOR_COLOR,
                lw=MODALITY_SEPARATOR_LW,
                zorder=0
            )


def collect_all_long_data(grouped: dict):
    all_rows = []
    summary_rows = []

    for dataset_name, rep_files in grouped.items():
        if not all(rep in rep_files for rep in REP_ORDER):
            print(f"[SKIP] {dataset_name}: missing one of QTN/GAF/MTF")
            continue

        print(f"\n===== DATASET: {dataset_name} =====")

        for metric, cfg in METRICS_TO_PLOT.items():
            qtn = load_metric_table(rep_files["QTN"], "QTN", metric)
            gaf = load_metric_table(rep_files["GAF"], "GAF", metric)
            mtf = load_metric_table(rep_files["MTF"], "MTF", metric)

            wide = merge_three_reps(qtn, gaf, mtf, dataset_name, metric)

            if wide.empty:
                print(f"[SKIP] {dataset_name} | {metric}: no matched subjects")
                summary_rows.append({
                    "dataset": dataset_name,
                    "metric": metric,
                    "group": cfg["group"],
                    "n_matched": 0
                })
                continue

            long = to_long(wide).copy()
            long["dataset"] = dataset_name
            long["dataset_display"] = pretty_dataset_name(dataset_name)
            long["metric"] = metric
            long["metric_label"] = cfg["ylabel"]
            long["group"] = cfg["group"]

            all_rows.append(long)

            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "group": cfg["group"],
                "n_matched": len(wide)
            })

            print(f"[OK] {dataset_name} | {metric} | matched={len(wide)}")

    all_long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return all_long, summary_df


def plot_combined_grid(all_long: pd.DataFrame, out_dir: Path):
    if all_long.empty:
        print("[WARN] No matched data available to plot.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_order = [m for m in METRICS_TO_PLOT if m in all_long["metric"].unique()]
    dataset_order, modality_boundaries, _, _ = build_dataset_layout(all_long)

    n_metrics = len(metric_order)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(18, len(dataset_order) * 1.45), nrows * 5.2),
        squeeze=False
    )
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sub = all_long[all_long["metric"] == metric].copy()

        sns.boxplot(
            data=sub,
            x="dataset_display",
            y="value",
            hue="representation",
            order=dataset_order,
            hue_order=REP_ORDER,
            palette=PALETTE_REP,
            width=BOX_WIDTH,
            showfliers=False,
            linewidth=1.1,
            ax=ax
        )

        if SHOW_POINTS:
            sns.stripplot(
                data=sub,
                x="dataset_display",
                y="value",
                hue="representation",
                order=dataset_order,
                hue_order=REP_ORDER,
                dodge=True,
                size=POINT_SIZE,
                alpha=POINT_ALPHA,
                color="black",
                ax=ax
            )

        add_vertical_separators(ax, dataset_order, modality_boundaries)
        add_threshold(ax, metric)

        ax.set_title(METRICS_TO_PLOT[metric]["ylabel"], fontsize=14, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=ROTATION, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        if legend_handles is None and legend_labels is None:
            legend_handles = handles[:3]
            legend_labels = [REP_DISPLAY[r] for r in REP_ORDER]

        if ax.get_legend() is not None:
            ax.get_legend().remove()

        sns.despine(ax=ax, trim=True)

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Representation",
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
            fontsize=13,
            title_fontsize=14
        )

  #  fig.suptitle("All datasets across representations", y=1.04, fontsize=18)
    plt.tight_layout()

    out_png = out_dir / "all_metrics_grouped_boxplots_by_dataset.png"
    out_pdf = out_dir / "all_metrics_grouped_boxplots_by_dataset.pdf"
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_png}")
    print(f"[SAVED] {out_pdf}")


def plot_one_metric_per_figure(all_long: pd.DataFrame, out_dir: Path):
    if all_long.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_order, modality_boundaries, _, _ = build_dataset_layout(all_long)

    for metric in [m for m in METRICS_TO_PLOT if m in all_long["metric"].unique()]:
        sub = all_long[all_long["metric"] == metric].copy()

        fig, ax = plt.subplots(figsize=(max(12, len(dataset_order) * 1.45), 6))

        sns.boxplot(
            data=sub,
            x="dataset_display",
            y="value",
            hue="representation",
            order=dataset_order,
            hue_order=REP_ORDER,
            palette=PALETTE_REP,
            width=BOX_WIDTH,
            showfliers=False,
            linewidth=1.1,
            ax=ax
        )

        if SHOW_POINTS:
            sns.stripplot(
                data=sub,
                x="dataset_display",
                y="value",
                hue="representation",
                order=dataset_order,
                hue_order=REP_ORDER,
                dodge=True,
                size=POINT_SIZE,
                alpha=POINT_ALPHA,
                color="black",
                ax=ax
            )

        add_vertical_separators(ax, dataset_order, modality_boundaries)
        add_threshold(ax, metric)

        ax.set_xlabel("")
        ax.set_ylabel(METRICS_TO_PLOT[metric]["ylabel"])
        plt.setp(ax.get_xticklabels(), rotation=ROTATION, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[:3],
            [REP_DISPLAY[r] for r in REP_ORDER],
            title="Representation",
            frameon=False,
            loc="best"
        )

        sns.despine(ax=ax, trim=True)
        plt.tight_layout()

        base = sanitize_name(metric)
        out_png = out_dir / f"{base}_grouped_boxplots_by_dataset.png"
        out_pdf = out_dir / f"{base}_grouped_boxplots_by_dataset.pdf"
        plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
        plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
        plt.close()

        print(f"[SAVED] {out_png}")
        print(f"[SAVED] {out_pdf}")


# ---------------------------------------
# MAIN
# ---------------------------------------
def main():
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir.resolve()}")

    all_files = sorted(glob.glob(str(input_dir / "metrics_*.csv")))
    if not all_files:
        raise RuntimeError(f"No metrics_*.csv files found in {input_dir.resolve()}")

    grouped = {}
    for f in all_files:
        parsed = extract_rep_dataset(f)
        if parsed is None:
            continue
        rep, dataset = parsed
        grouped.setdefault(dataset, {})
        grouped[dataset][rep] = f

    print("\nDetected datasets:")
    for ds, reps in grouped.items():
        print(f"  {ds}: {sorted(reps.keys())}")

    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    all_long, summary_df = collect_all_long_data(grouped)

    if all_long.empty:
        raise RuntimeError("No matched data were found across QTN/GAF/MTF.")

    all_long["representation_display"] = all_long["representation"].map(REP_DISPLAY)
    all_long.to_csv(out_root / "all_metrics_matched_long.csv", index=False)
    summary_df.sort_values(["group", "metric", "dataset"]).to_csv(
        out_root / "summary_processed_metrics.csv", index=False
    )

    plot_combined_grid(all_long, out_root / "combined")
    plot_one_metric_per_figure(all_long, out_root / "one_metric_per_figure")

    print(f"\n[DONE] Outputs saved in: {out_root.resolve()}")


if __name__ == "__main__":
    main()

# %% ---- next notebook cell ----


