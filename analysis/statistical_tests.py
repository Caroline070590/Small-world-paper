"""Pairwise Wilcoxon signed-rank tests, rank-biserial effect sizes, and Friedman
omnibus comparisons across representations (Fig. 7, Figs. S4-S5).

Provenance: extracted verbatim from the notebook ``Statistical-test-FINAL.ipynb``.
Figure-generating / analysis script; run top-to-bottom after setting paths.
"""

# hc_scz_violin_style.py (white bg, bigger fonts, no title, stars then (p=...))
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# ---------------- user paths ----------------
#FILE_HC  = "CSV-Small-world/metrics_MTF_SCZ-HC_transition_features.csv"
#FILE_SCZ = "CSV-Small-world/metrics_MTF_SCZ-SCZ_transition_features.csv"
#FILE_HC  = "CSV-New-metrics/metrics_GAF_SCZ_HC_transition_features.csv"
#FILE_SCZ = "CSV-New-metrics/metrics_GAF_SCZ_SCZ_transition_features.csv"
FILE_HC  = "CSV-New-metrics/metrics_QTN_ASD_TD_transition_features.csv"
FILE_SCZ = "CSV-New-metrics/metrics_QTN_ASD_ASD_transition_features.csv"
#OUT_DIR  = "HC_vs_SCZ_violin_style/GAF/"
OUT_DIR  = "HC_vs_ASD_violin_style/QTN/"
# -------------- style colors --------------
BG_WHITE   = "#ffffff"   # pure white
GREY50     = "#7F7F7F"
BLACK      = "#282724"
GREY_DARK  = "#747473"
RED_DARK   = "#850e00"

# “Dark2” palette (for dots)
COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3"]
COLOR_HC, COLOR_SCZ = COLOR_SCALE[0], COLOR_SCALE[2]

# -------------- options --------------
SAVE_DPI = 300
#GROUPS   = ["HC", "SCZ"]     # left to right
GROUPS   = ["HC", "ASD"]     # left to right
BASE_FONTSIZE = 15
IGNORE_COLS = {
    "patient_id","id","subject","group","diagnosis","label","key",
    "n_regions","Q_used"
}

# -------------- helpers --------------
def shared_numeric_columns(a: pd.DataFrame, b: pd.DataFrame):
    inter = [c for c in a.columns if c in b.columns and c not in IGNORE_COLS]
    keep = []
    for c in inter:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if a[c].notna().any() and b[c].notna().any():
                keep.append(c)
    return sorted(keep)

def welch_t(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    t, p = ttest_ind(x, y, equal_var=False)
    return float(t), float(p)

def hedges_g(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)
    if sp2 <= 0:
        return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    J = 1 - (3 / (4*(nx+ny) - 9))
    return float(J * d)

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p)
    n = mask.sum()
    q = np.full_like(p, np.nan, dtype=float)
    if n == 0:
        return q
    idx = np.argsort(p[mask])
    ranks = np.arange(1, n+1)
    q_mask = p[mask][idx] * n / ranks
    q_mask = np.minimum.accumulate(q_mask[::-1])[::-1]
    q[mask][idx] = np.clip(q_mask, 0, 1)
    return q

def star_string(p):
    if not np.isfinite(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05:  return "*"
    return "ns"

# -------------- plotting core (one metric) --------------
def plot_metric(metric, x_vals, y_vals, out_png, out_pdf):
    y_data = [np.asarray(x_vals, float), np.asarray(y_vals, float)]
    positions = [0, 1]

    # Welch test label pieces
    _, p = welch_t(x_vals, y_vals)
    label_p = f"p = {p:.3g}" if np.isfinite(p) and p >= 1e-4 else "p < 1e-4"
    stars   = star_string(p)

    # jitter
    rng = np.random.default_rng(7)
    x_jittered = [positions[i] + rng.normal(0, 0.06, size=len(ys)) for i, ys in enumerate(y_data)]

    # figure / fonts / background
    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
    })
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # light reference lines
    y_all = np.concatenate([y for y in y_data if len(y)])
    #if len(y_all):
     #   yr = (y_all.max() - y_all.min()) or 1.0
      #  ref = np.linspace(y_all.min() - 0.05*yr, y_all.max() + 0.05*yr, 4)
       # for h in ref:
        #    ax.axhline(h, color=GREY50, ls=(0, (5,5)), alpha=0.35, zorder=0)

    # violin (outline only)
    violins = ax.violinplot(
        y_data, positions=positions, widths=0.5, bw_method="silverman",
        showmeans=False, showmedians=False, showextrema=False
    )
    for body in violins["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor(BLACK)
        body.set_linewidth(1.6)
        body.set_alpha(1)

    # box (median thick, no caps)
    medianprops = dict(linewidth=4, color=GREY_DARK, solid_capstyle="butt")
    boxprops    = dict(linewidth=2, color=GREY_DARK)
    ax.boxplot(
        y_data, positions=positions, showfliers=False, showcaps=False,
        medianprops=medianprops, whiskerprops=boxprops, boxprops=boxprops
    )

    # jittered dots
    for xj, ys, col in zip(x_jittered, y_data, [COLOR_HC, COLOR_SCZ]):
        if len(ys) == 0: 
            continue
        ax.scatter(xj, ys, s=70, color=col, alpha=0.45, zorder=2.5)

    # means with label/leader line
    means = [np.nan if len(ys)==0 else np.mean(ys) for ys in y_data]
    for i, mu in enumerate(means):
        if not np.isfinite(mu): 
            continue
        ax.scatter(i, mu, s=260, color=RED_DARK, zorder=3)
        ax.plot([i, i + 0.18], [mu, mu], ls="dashdot", color="black", zorder=3)
        ax.text(
            i + 0.20, mu, r"$\hat{\mu}_{\rm{mean}} = $" + f"{mu:.2f}",
            fontsize=BASE_FONTSIZE, va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.18),
            zorder=10
        )

    # comparison bracket with "** (p = ...)"
    if len(y_all):
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        span = max(1e-9, y_max - y_min)
        y0 = y_max + 0.08*span
        tick_len = 0.18*span
        ax.plot([0, 0, 1, 1], [y0 - tick_len, y0, y0, y0 - tick_len], c="black")
        ax.text(
            0.5, y0 + 0.03*span,
            f"{stars} ({label_p})",      # <- stars first, p-value in parentheses
            fontsize=BASE_FONTSIZE, va="bottom", ha="center"
        )
        ax.set_ylim(y_min - 0.06*span, y0 + 0.18*span)

    # axes cosmetics (no N=)
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)   # <- just "HC", "SCZ"
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

# -------------- main --------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hc  = pd.read_csv(FILE_HC)
    scz = pd.read_csv(FILE_SCZ)

    metrics = shared_numeric_columns(hc, scz)
    if not metrics:
        raise RuntimeError("No shared numeric metrics found.")

    rows = []
    for m in metrics:
        x = pd.to_numeric(hc[m], errors="coerce").dropna()
        y = pd.to_numeric(scz[m], errors="coerce").dropna()
        if x.empty or y.empty:
            continue
        t, p = welch_t(x, y)
        g    = hedges_g(x, y)
        rows.append({
            "metric": m,
            "n_HC": len(x), "mean_HC": x.mean(), "sd_HC": x.std(ddof=1),
            "n_SCZ": len(y), "mean_SCZ": y.mean(), "sd_SCZ": y.std(ddof=1),
            "t_welch": t, "p_welch": p, "hedges_g": g
        })

        base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in m)
        plot_metric(
            m, x.values, y.values,
            os.path.join(OUT_DIR, f"{base}.png"),
            os.path.join(OUT_DIR, f"{base}.pdf")
        )

    if rows:
        sm = pd.DataFrame(rows).sort_values("p_welch")
        sm["q_bh_fdr"] = bh_fdr(sm["p_welch"].values)
        sm["sig_stars"] = [star_string(p) for p in sm["q_bh_fdr"].values]
        sm.to_csv(os.path.join(OUT_DIR, "HC_vs_SCZ_stats_summary.csv"), index=False)
        print(f"[OK] Wrote {len(rows)} figures and summary CSV in '{OUT_DIR}'.")
    else:
        print("No comparable metrics with data to plot.")

if __name__ == "__main__":
    main()


# %% ---- next notebook cell ----

# hc_scz_violin_style.py (white bg, bigger fonts, no title, stars then (p=...))
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu   # <-- replaced ttest_ind

# ---------------- user paths ----------------
#FILE_HC  = "CSV-Small-world/metrics_MTF_SCZ-HC_transition_features.csv"
#FILE_SCZ = "CSV-Small-world/metrics_MTF_SCZ-SCZ_transition_features.csv"
#FILE_HC  = "CSV-New-metrics/metrics_GAF_SCZ_HC_transition_features.csv"
#FILE_SCZ = "CSV-New-metrics/metrics_GAF_SCZ_SCZ_transition_features.csv"
#FILE_HC  = "CSV-New-metrics/metrics_GAF_ASD_TD_transition_features.csv"
#FILE_SCZ = "CSV-New-metrics/metrics_GAF_ASD_ASD_transition_features.csv"

FILE_HC  = "CSV-New-metrics-2/metrics_GAF_SCZ_HC_transition_features.csv"
FILE_SCZ = "CSV-New-metrics-2/metrics_GAF_SCZ_SCZ_transition_features.csv"

#OUT_DIR  = "HC_vs_SCZ_violin_style/GAF/"
OUT_DIR  = "HC_vs_ASD_violin_style-non-2/GAF/"

# -------------- style colors --------------
BG_WHITE   = "#ffffff"   # pure white
GREY50     = "#7F7F7F"
BLACK      = "#282724"
GREY_DARK  = "#747473"
RED_DARK   = "#850e00"

# “Dark2” palette (for dots)
COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3"]
COLOR_HC, COLOR_SCZ = COLOR_SCALE[0], COLOR_SCALE[2]

# -------------- options --------------
SAVE_DPI = 1000
#GROUPS   = ["HC", "ASD"]     # left to right
GROUPS   = ["HC", "SCZ"] 
BASE_FONTSIZE = 15
IGNORE_COLS = {
    "patient_id","id","subject","group","diagnosis","label","key",
    "n_regions","Q_used"
}

# -------------- helpers --------------
def shared_numeric_columns(a: pd.DataFrame, b: pd.DataFrame):
    inter = [c for c in a.columns if c in b.columns and c not in IGNORE_COLS]
    keep = []
    for c in inter:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if a[c].notna().any() and b[c].notna().any():
                keep.append(c)
    return sorted(keep)

def welch_t(x, y):
    """
    REPLACED: returns (U, p) from Mann–Whitney U test (two-sided).
    Kept the name so the rest of the script doesn't change.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    U, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(U), float(p)

def hedges_g(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)
    if sp2 <= 0:
        return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    J = 1 - (3 / (4*(nx+ny) - 9))
    return float(J * d)

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p)
    n = mask.sum()
    q = np.full_like(p, np.nan, dtype=float)
    if n == 0:
        return q
    idx = np.argsort(p[mask])
    ranks = np.arange(1, n+1)
    q_mask = p[mask][idx] * n / ranks
    q_mask = np.minimum.accumulate(q_mask[::-1])[::-1]
    q[mask][idx] = np.clip(q_mask, 0, 1)
    return q

def star_string(p):
    if not np.isfinite(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05:  return "*"
    return "ns"

# -------------- plotting core (one metric) --------------
def plot_metric(metric, x_vals, y_vals, out_png, out_pdf):
    y_data = [np.asarray(x_vals, float), np.asarray(y_vals, float)]
    positions = [0, 1]

    # Now using Mann–Whitney U
    _, p = welch_t(x_vals, y_vals)
    label_p = f"p = {p:.3g}" if np.isfinite(p) and p >= 1e-4 else "p < 1e-4"
    stars   = star_string(p)

    # jitter
    rng = np.random.default_rng(7)
    x_jittered = [positions[i] + rng.normal(0, 0.06, size=len(ys)) for i, ys in enumerate(y_data)]

    # figure / fonts / background
    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
    })
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # violin (outline only)
    violins = ax.violinplot(
        y_data, positions=positions, widths=0.5, bw_method="silverman",
        showmeans=False, showmedians=False, showextrema=False
    )
    for body in violins["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor(BLACK)
        body.set_linewidth(1.6)
        body.set_alpha(1)

    # box (median thick, no caps)
    medianprops = dict(linewidth=4, color=GREY_DARK, solid_capstyle="butt")
    boxprops    = dict(linewidth=2, color=GREY_DARK)
    ax.boxplot(
        y_data, positions=positions, showfliers=False, showcaps=False,
        medianprops=medianprops, whiskerprops=boxprops, boxprops=boxprops
    )

    # jittered dots
    for xj, ys, col in zip(x_jittered, y_data, [COLOR_HC, COLOR_SCZ]):
        if len(ys) == 0: 
            continue
        ax.scatter(xj, ys, s=70, color=col, alpha=0.45, zorder=2.5)

    # means with label/leader line
    means = [np.nan if len(ys)==0 else np.mean(ys) for ys in y_data]
    for i, mu in enumerate(means):
        if not np.isfinite(mu): 
            continue
        ax.scatter(i, mu, s=260, color=RED_DARK, zorder=3)
        ax.plot([i, i + 0.18], [mu, mu], ls="dashdot", color="black", zorder=3)
        ax.text(
            i + 0.20, mu, r"$\hat{\mu}_{\rm{mean}} = $" + f"{mu:.2f}",
            fontsize=BASE_FONTSIZE, va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.18),
            zorder=10
        )

    # comparison bracket with "** (p = ...)"
    y_all = np.concatenate([y for y in y_data if len(y)])
    if len(y_all):
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        span = max(1e-9, y_max - y_min)
        y0 = y_max + 0.08*span
        tick_len = 0.18*span
        ax.plot([0, 0, 1, 1], [y0 - tick_len, y0, y0, y0 - tick_len], c="black")
        ax.text(
            0.5, y0 + 0.03*span,
            f"{stars} ({label_p})",
            fontsize=BASE_FONTSIZE, va="bottom", ha="center"
        )
        ax.set_ylim(y_min - 0.06*span, y0 + 0.18*span)

    # axes cosmetics
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

# -------------- main --------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hc  = pd.read_csv(FILE_HC)
    scz = pd.read_csv(FILE_SCZ)

    metrics = shared_numeric_columns(hc, scz)
    if not metrics:
        raise RuntimeError("No shared numeric metrics found.")

    rows = []
    for m in metrics:
        x = pd.to_numeric(hc[m], errors="coerce").dropna()
        y = pd.to_numeric(scz[m], errors="coerce").dropna()
        if x.empty or y.empty:
            continue
        # Using Mann–Whitney U, but keep column names so the rest of the pipeline works
        u_stat, p = welch_t(x, y)   # welch_t() now returns (U, p)
        g = hedges_g(x, y)

        rows.append({
            "metric": m,
            "n_HC": len(x), "mean_HC": x.mean(), "sd_HC": x.std(ddof=1),
            "n_SCZ": len(y), "mean_SCZ": y.mean(), "sd_SCZ": y.std(ddof=1),
            "t_welch": u_stat,   # NOTE: this is U statistic now
            "p_welch": p,        # NOTE: MW p-value
            "hedges_g": g
        })

        base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in m)
        plot_metric(
            m, x.values, y.values,
            os.path.join(OUT_DIR, f"{base}.png"),
            os.path.join(OUT_DIR, f"{base}.pdf")
        )

    if rows:
        sm = pd.DataFrame(rows).sort_values("p_welch")
        sm["q_bh_fdr"] = bh_fdr(sm["p_welch"].values)
        sm["sig_stars"] = [star_string(p) for p in sm["q_bh_fdr"].values]
        sm.to_csv(os.path.join(OUT_DIR, "HC_vs_SCZ_stats_summary.csv"), index=False)
        print(f"[OK] Wrote {len(rows)} figures and summary CSV in '{OUT_DIR}'.")
    else:
        print("No comparable metrics with data to plot.")

if __name__ == "__main__":
    main()


# %% ---- next notebook cell ----

# hc_scz_violin_style_optimized.py
# Same look & feel, optimized for many metrics (no kernel crashes)

import os, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless, safer for batch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu   # nonparametric two-sample

# ---------------- user paths ----------------
#FILE_HC  = "CSV-New-metrics-3/metrics_MTF_SCZ_HC_transition_features.csv"
#FILE_SCZ = "CSV-New-metrics-3/metrics_MTF_SCZ_SCZ_transition_features.csv"
FILE_HC  = "CSV-New-metrics-3/metrics_GAF_ASD_TD_transition_features.csv"
FILE_SCZ = "CSV-New-metrics-3/metrics_GAF_ASD_ASD_transition_features.csv"
#FILE_HC  = "CSV-New-metrics-3/metrics_MTF_ADHD_TD_transition_features.csv"
#FILE_SCZ = "CSV-New-metrics-3/metrics_MTF_ADHD_ADHD_transition_features.csv"

#OUT_DIR  = "HC_vs_ADHD_violin_style-non-3/MTF/"
#OUT_DIR  = "HC_vs_SCZ_violin_style-non-3/MTF/"
OUT_DIR  = "HC_vs_ASD_violin_style-non-3/MTF/"
#OUT_MULTIPAGE_PDF = "HC_vs_ADHD_all_metrics.pdf"   # one PDF with all pages
#OUT_MULTIPAGE_PDF = "HC_vs_SCZ_all_metrics.pdf"
OUT_MULTIPAGE_PDF = "HC_vs_ASD_all_metrics.pdf"
SAVE_TOPK_PNG = 20                                # also export PNGs for top-K most significant metrics (set 0 to disable)

# -------------- style colors --------------
BG_WHITE   = "#ffffff"
BLACK      = "#282724"
GREY_DARK  = "#747473"
RED_DARK   = "#850e00"

# “Dark2” palette (for dots)
COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3"]
COLOR_HC, COLOR_SCZ = COLOR_SCALE[0], COLOR_SCALE[2]

# -------------- options --------------
SAVE_DPI = 600                   # 1000 is overkill for batch; 300 is plenty
GROUPS   = ["HC", "SCZ"]
#GROUPS   = ["HC", "ASD"]
#GROUPS   = ["HC", "ADHD"]
BASE_FONTSIZE = 15
IGNORE_COLS = {
    "patient_id","id","subject","group","diagnosis","label","key",
    "n_regions","Q_used"
}

# -------------- helpers --------------
def shared_numeric_columns(a: pd.DataFrame, b: pd.DataFrame):
    inter = [c for c in a.columns if c in b.columns and c not in IGNORE_COLS]
    keep = []
    for c in inter:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if a[c].notna().any() and b[c].notna().any():
                keep.append(c)
    return sorted(keep)

def mann_whitney(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    U, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(U), float(p)

def hedges_g(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)
    if sp2 <= 0: return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    J = 1 - (3 / (4*(nx+ny) - 9))
    return float(J * d)

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p); n = mask.sum()
    q = np.full_like(p, np.nan, dtype=float)
    if n == 0: return q
    idx = np.argsort(p[mask]); ranks = np.arange(1, n+1)
    q_mask = p[mask][idx] * n / ranks
    q_mask = np.minimum.accumulate(q_mask[::-1])[::-1]
    q[mask][idx] = np.clip(q_mask, 0, 1)
    return q

def star_string(p):
    if not np.isfinite(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05:  return "*"
    return "ns"

# -------- single plot (returns a Matplotlib Figure) --------
def build_violin_box(metric, x_vals, y_vals):
    y_data = [np.asarray(x_vals, float), np.asarray(y_vals, float)]
    positions = [0, 1]

    # Mann–Whitney U
    _, p = mann_whitney(x_vals, y_vals)
    label_p = f"p = {p:.3g}" if np.isfinite(p) and p >= 1e-4 else "p < 1e-4"
    stars   = star_string(p)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # violin (outline only)
    violins = ax.violinplot(
        y_data, positions=positions, widths=0.5, bw_method="silverman",
        showmeans=False, showmedians=False, showextrema=False
    )
    for body in violins["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor(BLACK)
        body.set_linewidth(1.6)
        body.set_alpha(1)

    # box (median thick, no caps)
    medianprops = dict(linewidth=4, color=GREY_DARK, solid_capstyle="butt")
    boxprops    = dict(linewidth=2, color=GREY_DARK)
    ax.boxplot(
        y_data, positions=positions, showfliers=False, showcaps=False,
        medianprops=medianprops, whiskerprops=boxprops, boxprops=boxprops
    )

    # jittered dots (rasterized to keep PDF small)
    rng = np.random.default_rng(7)
    x_jittered = [positions[i] + rng.normal(0, 0.06, size=len(ys)) for i, ys in enumerate(y_data)]
    for xj, ys, col in zip(x_jittered, y_data, [COLOR_HC, COLOR_SCZ]):
        if len(ys) == 0: 
            continue
        ax.scatter(xj, ys, s=70, color=col, alpha=0.45, zorder=2.5, rasterized=True)

    # means with label/leader line
    means = [np.nan if len(ys)==0 else np.mean(ys) for ys in y_data]
    for i, mu in enumerate(means):
        if not np.isfinite(mu): 
            continue
        ax.scatter(i, mu, s=260, color=RED_DARK, zorder=3)
        ax.plot([i, i + 0.18], [mu, mu], ls="dashdot", color="black", zorder=3)
        ax.text(
            i + 0.20, mu, r"$\hat{\mu}_{\rm{mean}} = $" + f"{mu:.2f}",
            fontsize=BASE_FONTSIZE, va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.18),
            zorder=10
        )

    # comparison bracket with "** (p = ...)"
    y_all = np.concatenate([y for y in y_data if len(y)])
    if len(y_all):
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        span = max(1e-9, y_max - y_min)
        y0 = y_max + 0.08*span
        tick_len = 0.18*span
        ax.plot([0, 0, 1, 1], [y0 - tick_len, y0, y0, y0 - tick_len], c="black")
        ax.text(
            0.5, y0 + 0.03*span,
            f"{stars} ({label_p})",
            fontsize=BASE_FONTSIZE, va="bottom", ha="center"
        )
        ax.set_ylim(y_min - 0.06*span, y0 + 0.18*span)

    # axes cosmetics
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig, p

# -------------- main --------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # set fonts once (not per-figure)
    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
    })

    hc  = pd.read_csv(FILE_HC)
    scz = pd.read_csv(FILE_SCZ)

    metrics = shared_numeric_columns(hc, scz)
    if not metrics:
        raise RuntimeError("No shared numeric metrics found.")

    # First pass: compute p-values and store data (no plotting)
    results = []
    for m in metrics:
        x = pd.to_numeric(hc[m], errors="coerce").dropna()
        y = pd.to_numeric(scz[m], errors="coerce").dropna()
        if x.empty or y.empty:
            continue
        U, p = mann_whitney(x, y)
        g    = hedges_g(x, y)
        results.append({
            "metric": m,
            "n_HC": len(x), "mean_HC": x.mean(), "sd_HC": x.std(ddof=1),
            "n_SCZ": len(y), "mean_SCZ": y.mean(), "sd_SCZ": y.std(ddof=1),
            "U_mw": U, "p_mw": p, "hedges_g": g,
            "_x": x.values, "_y": y.values,     # stash for plotting in pass 2
        })

    if not results:
        print("No comparable metrics with data to plot.")
        return

    sm = pd.DataFrame(results).sort_values("p_mw").reset_index(drop=True)
    sm["q_bh_fdr"] = bh_fdr(sm["p_mw"].values)
    sm["sig_stars"] = [star_string(p) for p in sm["q_bh_fdr"].values]

    # Save stats summary
    sm_out = sm.drop(columns=["_x","_y"])
    sm_out.to_csv(os.path.join(OUT_DIR, "HC_vs_SCZ_stats_summary.csv"), index=False)

    # Second pass: plot to ONE multi-page PDF (super fast & memory friendly)
    pdf_path = os.path.join(OUT_DIR, OUT_MULTIPAGE_PDF)
    with PdfPages(pdf_path) as pdf:
        for i, row in sm.iterrows():
            fig, _ = build_violin_box(row["metric"], row["_x"], row["_y"])
            pdf.savefig(fig)                    # one page per metric
            plt.close(fig)                      # free memory ASAP
            if (i + 1) % 20 == 0:
                gc.collect()

    print(f"[OK] Wrote multipage PDF -> {pdf_path}")

    # Optionally: also export PNGs for top-K most significant metrics only
    if SAVE_TOPK_PNG and SAVE_TOPK_PNG > 0:
        top = sm.head(SAVE_TOPK_PNG)
        for _, row in top.iterrows():
            base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row["metric"])
            fig, _ = build_violin_box(row["metric"], row["_x"], row["_y"])
            out_png = os.path.join(OUT_DIR, f"{base}.png")
            fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
        print(f"[OK] Also wrote PNGs for top {len(top)} metrics.")

if __name__ == "__main__":
    main()


# %% ---- next notebook cell ----

# hc_scz_violin_style_optimized.py
# Same look & feel, optimized for many metrics (no kernel crashes)

import os, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless, safer for batch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu   # nonparametric two-sample

# ---------------- user paths ----------------
FILE_HC  = "CSV-New-metrics-3-windowed/ADHD_6min_TR2/HC__mtf_windowed__Q12__TR2__min6.csv"
FILE_SCZ = "CSV-New-metrics-3-windowed/ADHD_6min_TR2/ADHD__mtf_windowed__Q12__TR2__min6.csv"
#FILE_HC  = "CSV-New-metrics-3-windowed/ABIDE_5min_TR2/TD__mtf_windowed__Q12__TR2__min5.csv"
#FILE_SCZ = "CSV-New-metrics-3-windowed/ABIDE_5min_TR2/ASD__mtf_windowed__Q12__TR2__min5.csv"
#FILE_HC  = "CSV-New-metrics-3-windowed/SCZ_6min_TR2/HC__mtf_windowed__Q12__TR2__min6.csv"
#FILE_SCZ = "CSV-New-metrics-3-windowed/SCZ_6min_TR2/SCZ__mtf_windowed__Q12__TR2__min6.csv"

OUT_DIR  = "HC_vs_ADHD_violin_style-windown/"
#OUT_DIR  = "HC_vs_SCZ_violin_style-windown/"
#OUT_DIR  = "HC_vs_ASD_violin_style-windown/"
OUT_MULTIPAGE_PDF = "HC_vs_ADHD_all_metrics.pdf"   # one PDF with all pages
#OUT_MULTIPAGE_PDF = "HC_vs_SCZ_all_metrics.pdf"
#OUT_MULTIPAGE_PDF = "HC_vs_ASD_all_metrics.pdf"
SAVE_TOPK_PNG = 20                                # also export PNGs for top-K most significant metrics (set 0 to disable)

# -------------- style colors --------------
BG_WHITE   = "#ffffff"
BLACK      = "#282724"
GREY_DARK  = "#747473"
RED_DARK   = "#850e00"

# “Dark2” palette (for dots)
COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3"]
COLOR_HC, COLOR_SCZ = COLOR_SCALE[0], COLOR_SCALE[2]

# -------------- options --------------
SAVE_DPI = 600                   # 1000 is overkill for batch; 300 is plenty
#GROUPS   = ["HC", "SCZ"]
#GROUPS   = ["HC", "ASD"]
GROUPS   = ["HC", "ADHD"]
BASE_FONTSIZE = 15
IGNORE_COLS = {
    "patient_id","id","subject","group","diagnosis","label","key",
    "n_regions","Q_used"
}

# -------------- helpers --------------
def shared_numeric_columns(a: pd.DataFrame, b: pd.DataFrame):
    inter = [c for c in a.columns if c in b.columns and c not in IGNORE_COLS]
    keep = []
    for c in inter:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            if a[c].notna().any() and b[c].notna().any():
                keep.append(c)
    return sorted(keep)

def mann_whitney(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    U, p = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(U), float(p)

def hedges_g(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    sx2, sy2 = x.var(ddof=1), y.var(ddof=1)
    sp2 = ((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2)
    if sp2 <= 0: return np.nan
    d = (x.mean() - y.mean()) / np.sqrt(sp2)
    J = 1 - (3 / (4*(nx+ny) - 9))
    return float(J * d)

def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p); n = mask.sum()
    q = np.full_like(p, np.nan, dtype=float)
    if n == 0: return q
    idx = np.argsort(p[mask]); ranks = np.arange(1, n+1)
    q_mask = p[mask][idx] * n / ranks
    q_mask = np.minimum.accumulate(q_mask[::-1])[::-1]
    q[mask][idx] = np.clip(q_mask, 0, 1)
    return q

def star_string(p):
    if not np.isfinite(p): return ""
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05:  return "*"
    return "ns"

# -------- single plot (returns a Matplotlib Figure) --------
def build_violin_box(metric, x_vals, y_vals):
    y_data = [np.asarray(x_vals, float), np.asarray(y_vals, float)]
    positions = [0, 1]

    # Mann–Whitney U
    _, p = mann_whitney(x_vals, y_vals)
    label_p = f"p = {p:.3g}" if np.isfinite(p) and p >= 1e-4 else "p < 1e-4"
    stars   = star_string(p)

    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)

    # violin (outline only)
    violins = ax.violinplot(
        y_data, positions=positions, widths=0.5, bw_method="silverman",
        showmeans=False, showmedians=False, showextrema=False
    )
    for body in violins["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor(BLACK)
        body.set_linewidth(1.6)
        body.set_alpha(1)

    # box (median thick, no caps)
    medianprops = dict(linewidth=4, color=GREY_DARK, solid_capstyle="butt")
    boxprops    = dict(linewidth=2, color=GREY_DARK)
    ax.boxplot(
        y_data, positions=positions, showfliers=False, showcaps=False,
        medianprops=medianprops, whiskerprops=boxprops, boxprops=boxprops
    )

    # jittered dots (rasterized to keep PDF small)
    rng = np.random.default_rng(7)
    x_jittered = [positions[i] + rng.normal(0, 0.06, size=len(ys)) for i, ys in enumerate(y_data)]
    for xj, ys, col in zip(x_jittered, y_data, [COLOR_HC, COLOR_SCZ]):
        if len(ys) == 0: 
            continue
        ax.scatter(xj, ys, s=70, color=col, alpha=0.45, zorder=2.5, rasterized=True)

    # means with label/leader line
    means = [np.nan if len(ys)==0 else np.mean(ys) for ys in y_data]
    for i, mu in enumerate(means):
        if not np.isfinite(mu): 
            continue
        ax.scatter(i, mu, s=260, color=RED_DARK, zorder=3)
        ax.plot([i, i + 0.18], [mu, mu], ls="dashdot", color="black", zorder=3)
        ax.text(
            i + 0.20, mu, r"$\hat{\mu}_{\rm{mean}} = $" + f"{mu:.2f}",
            fontsize=BASE_FONTSIZE, va="center",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round", pad=0.18),
            zorder=10
        )

    # comparison bracket with "** (p = ...)"
    y_all = np.concatenate([y for y in y_data if len(y)])
    if len(y_all):
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        span = max(1e-9, y_max - y_min)
        y0 = y_max + 0.08*span
        tick_len = 0.18*span
        ax.plot([0, 0, 1, 1], [y0 - tick_len, y0, y0, y0 - tick_len], c="black")
        ax.text(
            0.5, y0 + 0.03*span,
            f"{stars} ({label_p})",
            fontsize=BASE_FONTSIZE, va="bottom", ha="center"
        )
        ax.set_ylim(y_min - 0.06*span, y0 + 0.18*span)

    # axes cosmetics
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig, p

# -------------- main --------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # set fonts once (not per-figure)
    plt.rcParams.update({
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "legend.fontsize": BASE_FONTSIZE,
    })

    hc  = pd.read_csv(FILE_HC)
    scz = pd.read_csv(FILE_SCZ)

    metrics = shared_numeric_columns(hc, scz)
    if not metrics:
        raise RuntimeError("No shared numeric metrics found.")

    # First pass: compute p-values and store data (no plotting)
    results = []
    for m in metrics:
        x = pd.to_numeric(hc[m], errors="coerce").dropna()
        y = pd.to_numeric(scz[m], errors="coerce").dropna()
        if x.empty or y.empty:
            continue
        U, p = mann_whitney(x, y)
        g    = hedges_g(x, y)
        results.append({
            "metric": m,
            "n_HC": len(x), "mean_HC": x.mean(), "sd_HC": x.std(ddof=1),
            "n_SCZ": len(y), "mean_SCZ": y.mean(), "sd_SCZ": y.std(ddof=1),
            "U_mw": U, "p_mw": p, "hedges_g": g,
            "_x": x.values, "_y": y.values,     # stash for plotting in pass 2
        })

    if not results:
        print("No comparable metrics with data to plot.")
        return

    sm = pd.DataFrame(results).sort_values("p_mw").reset_index(drop=True)
    sm["q_bh_fdr"] = bh_fdr(sm["p_mw"].values)
    sm["sig_stars"] = [star_string(p) for p in sm["q_bh_fdr"].values]

    # Save stats summary
    sm_out = sm.drop(columns=["_x","_y"])
    sm_out.to_csv(os.path.join(OUT_DIR, "HC_vs_SCZ_stats_summary.csv"), index=False)

    # Second pass: plot to ONE multi-page PDF (super fast & memory friendly)
    pdf_path = os.path.join(OUT_DIR, OUT_MULTIPAGE_PDF)
    with PdfPages(pdf_path) as pdf:
        for i, row in sm.iterrows():
            fig, _ = build_violin_box(row["metric"], row["_x"], row["_y"])
            pdf.savefig(fig)                    # one page per metric
            plt.close(fig)                      # free memory ASAP
            if (i + 1) % 20 == 0:
                gc.collect()

    print(f"[OK] Wrote multipage PDF -> {pdf_path}")

    # Optionally: also export PNGs for top-K most significant metrics only
    if SAVE_TOPK_PNG and SAVE_TOPK_PNG > 0:
        top = sm.head(SAVE_TOPK_PNG)
        for _, row in top.iterrows():
            base = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row["metric"])
            fig, _ = build_violin_box(row["metric"], row["_x"], row["_y"])
            out_png = os.path.join(OUT_DIR, f"{base}.png")
            fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
        print(f"[OK] Also wrote PNGs for top {len(top)} metrics.")

if __name__ == "__main__":
    main()


# %% ---- next notebook cell ----


