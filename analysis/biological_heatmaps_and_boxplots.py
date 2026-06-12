"""Median heatmaps and metric distribution boxplots across biological datasets
for QG/GAF/MTF (Fig. 5, Fig. 6).

Provenance: extracted verbatim from the notebook ``BOXPLOTS-ALL-FINAL.ipynb``.
Figure-generating / analysis script; run top-to-bottom after setting paths.
"""

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
INPUT_DIR = "csv"                      # folder with all metrics_*.csv
OUT_ROOT = "plots_all_datasets"

# Internal file representations
REP_ORDER = ["QTN", "GAF", "MTF"]

# Labels to show in plots
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

# metrics to plot
# omega and phi intentionally removed
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

POINT_SIZE = 34
LINE_WIDTH = 1.2
JITTER_SD = 0.05
RNG_SEED = 2025
DPI = 1000

sns.set(context="notebook", style="white")
plt.rcParams.update({
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
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
    """
    Broader ID detection so datasets like calcium and MEA are not skipped.
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

    # fallback: first object/string-like column
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        return df[object_cols[0]].astype(str).map(normalize_key)

    # final fallback
    return pd.Series([f"{rep}_{i}" for i in range(len(df))], name="auto_id")

def load_metric_table(path: str, rep: str, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    ids = get_id_series(df, rep)

    if metric not in df.columns:
        return pd.DataFrame()

    out = pd.DataFrame({
        "key": ids,
        rep: pd.to_numeric(df[metric], errors="coerce")
    })

    out = out.dropna(subset=[rep])
    out = out.groupby("key", as_index=False)[rep].mean(numeric_only=True)
    return out

def merge_three_reps(qtn: pd.DataFrame, gaf: pd.DataFrame, mtf: pd.DataFrame, dataset_name: str = "", metric: str = "") -> pd.DataFrame:
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

def plot_metric_same_style(wide: pd.DataFrame, dataset_name: str, metric: str, cfg: dict, out_dir: str):
    long = to_long(wide)

    rng = np.random.default_rng(RNG_SEED)
    xcats = {rep: i for i, rep in enumerate(REP_ORDER)}

    fig, ax = plt.subplots(figsize=(9, 5.4))

    sns.violinplot(
        data=long,
        x="representation",
        y="value",
        order=REP_ORDER,
        inner=None,
        cut=0,
        palette=PALETTE_REP,
        linewidth=1,
        saturation=1,
        ax=ax
    )

    sns.boxplot(
        data=long,
        x="representation",
        y="value",
        order=REP_ORDER,
        width=0.28,
        showcaps=True,
        boxprops={"facecolor": "white", "zorder": 3},
        showfliers=False,
        whiskerprops={"linewidth": 1},
        medianprops={"linewidth": 1.5, "color": "black"},
        ax=ax
    )

    thr = cfg["threshold"]
    thr_label = cfg["threshold_label"]

    if thr is not None:
        ax.axhline(thr, ls="--", lw=1.2, color="black", alpha=0.9)
        ymin, ymax = ax.get_ylim()
        y_text = thr + 0.01 * (ymax - ymin)
        ax.text(
            len(REP_ORDER) - 0.05,
            y_text,
            thr_label,
            ha="right",
            va="bottom",
            fontsize=10
        )

    line_color = "#9e9e9e"
    point_face = "#bdbdbd"
    point_edge = "#f2f2f2"

    for _, row in wide.iterrows():
        jitters = rng.normal(0, JITTER_SD, size=len(REP_ORDER))
        xs = [xcats[rep] + jitters[i] for i, rep in enumerate(REP_ORDER)]
        ys = [row[rep] for rep in REP_ORDER]

        ax.plot(xs, ys, color=line_color, alpha=0.3, lw=LINE_WIDTH, zorder=3.5)
        ax.scatter(
            xs, ys,
            s=POINT_SIZE,
            facecolor=point_face,
            edgecolor=point_edge,
            linewidths=0.5,
            alpha=0.7,
            zorder=4
        )

    ax.set_xlabel("")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_xticklabels([REP_DISPLAY[r] for r in REP_ORDER])

    sns.despine(trim=True)

    os.makedirs(out_dir, exist_ok=True)
    base = f"{metric}_violin_box_points_lines-{sanitize_name(dataset_name)}"
    out_png = os.path.join(out_dir, f"{base}.png")
    out_pdf = os.path.join(out_dir, f"{base}.pdf")

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close()

def save_tables(wide: pd.DataFrame, dataset_name: str, metric: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    long = to_long(wide).copy()
    long["representation"] = long["representation"].map(REP_DISPLAY)

    base = f"{metric}_matched-{sanitize_name(dataset_name)}"
    wide.to_csv(os.path.join(out_dir, f"{base}_wide.csv"), index=False)
    long.to_csv(os.path.join(out_dir, f"{base}_long.csv"), index=False)

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
    (out_root / "small_world").mkdir(parents=True, exist_ok=True)
    (out_root / "topology").mkdir(parents=True, exist_ok=True)
    (out_root / "null_model").mkdir(parents=True, exist_ok=True)
    (out_root / "matched_tables").mkdir(parents=True, exist_ok=True)

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

            table_dir = out_root / "matched_tables" / metric
            save_tables(wide, dataset_name, metric, str(table_dir))

            plot_dir = out_root / cfg["group"] / metric
            plot_metric_same_style(wide, dataset_name, metric, cfg, str(plot_dir))

            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric,
                "group": cfg["group"],
                "n_matched": len(wide)
            })

            print(f"[OK] {dataset_name} | {metric} | matched={len(wide)}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["group", "metric", "dataset"])
    summary_df.to_csv(out_root / "summary_processed_metrics.csv", index=False)

    print(f"\n[DONE] All outputs saved in: {out_root.resolve()}")

if __name__ == "__main__":
    main()

# %% ---- next notebook cell ----



# %% ---- next notebook cell ----


