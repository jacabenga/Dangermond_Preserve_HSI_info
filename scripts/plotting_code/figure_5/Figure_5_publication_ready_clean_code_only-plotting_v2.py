# ============================================================
# Figure_5_publication_ready_clean_code_only-plotting_v2_for_Fabian_v2.py
# ============================================================
#
# date:    01.10.2026   (v3; v2 dated 27.02.2026)
# author:  Jaime C. Revenga
#
# PURPOSE:
#   Plotting-only version of Figure 5 (Dangermond Info Paper).
#   No heavy computation — loads pre-saved .pkl and .csv files.
#   Produces one combined 3×3 figure:
#
#   Row 1 (A–C) — KLPD / ΔShape (ΔG₁₃), selected community vs. PFT mean, coloured by date
#                 A: Shrub (class 20)  |  B: Herbaceous (class 33)  |  C: Woodland / Tree (class 16)
#   Row 2 (D–F) — Mean filter excitation (RCF), coloured by date
#   Row 3 (G–I) — Spectrally-windowed Random Forest, 32 windows
#                 Scatter per date + mean ± SD band
#
# CHANGES vs v2 (Row 3 only; Rows 1–2 unchanged):
#   - Reads the output of RF_windowed_v3_fixed_split.py (one long CSV per task:
#     RF_windowed_{TASK}_all_runs.csv, with columns date, rep, window, lo_nm,
#     hi_nm, OA, balanced_acc, macro_F1, kappa, ...), instead of one CSV per date.
#   - Metric is a global parameter (RF_METRIC); default = balanced accuracy.
#   - SD band is a global parameter (RF_BAND); default = SD across dates.
#   - Excluded windows are derived from wavelengths with the SAME rule as v3
#     (see EXCLUSION RULE below), instead of a hard-coded index list.
#   - Date colours are fixed by position in the 13-date list, so they always
#     match the shared colorbar even if a date is missing.
#
# DATA SOURCES:
#   Row 1 & 2:  .../Figure_5/data_in/{pft}/stored_delta_G_13_vectors/*.pkl
#               .../Figure_5/data_in/{pft}/stored_kernel_excitation_vectors/*.pkl
#   Row 3:      .../Figure_5/data_in/{pft}/{RF_SUBDIR}/RF_windowed_{TASK}_all_runs.csv
#
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ============================================================
# 1. USER PARAMETERS  ← edit these to match your machine / choices
# ============================================================

base_dir = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_5\data_in'

dir_fig_out = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_5\figures_out'
os.makedirs(dir_fig_out, exist_ok=True)

# ---- Row 3 (windowed RF) options ----------------------------------------
RF_SUBDIR = "RF_windowed_v3"        # subfolder of {pft} holding the v3 CSVs

# Metric to plot: 'balanced_acc' | 'OA' | 'macro_F1' | 'kappa'
RF_METRIC = "balanced_acc"

# What the shaded band shows:
#   'dates' -> SD across the 13 dates of the per-date means (temporal spread)
#   'reps'  -> SD across the resampling repetitions, averaged over dates
#   'both'  -> band = SD across dates, plus per-point error bars = SD across reps
RF_BAND = "dates"

SD_DDOF = 1                         # 1 = sample SD (same as pandas in v3); 0 = population SD (v2)
SHOW_FULL_SPECTRUM_REF = True       # dotted line: full-spectrum model (mean across dates)
SHOW_CHANCE_LEVEL = True            # dashed line at 1/n_classes (balanced_acc only)

# ---- EXCLUSION RULE (keep identical to RF_windowed_v3_fixed_split.py) ----
# A spectral window is excluded from the plot if at least EXCLUDE_MIN_OVERLAP
# of its width falls inside GRAY_REGIONS_NM (water vapour / sensor limits).
# RF_windowed_v3_fixed_split.py uses the SAME regions and threshold to set its
# 'excluded_gray' column (and to restrict its resampling-stability statistics).
# If you change either value here, change it in v3 too, and vice versa.
# With 32 windows this excludes Seg01, Seg15, Seg16, Seg22, Seg23, Seg24, Seg32.
GRAY_REGIONS_NM = [
    (0,    450),
    (1325, 1450),
    (1775, 2000),
    (2450, 2510),
]
EXCLUDE_MIN_OVERLAP = 0.5

# Chronological date list (ISO, as in the v3 CSVs) and display labels
DATES_ISO = [
    "2022-02-24", "2022-02-28",
    "2022-03-08", "2022-03-16", "2022-03-22",
    "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
    "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29",
]
date_list = [
    "Feb-24", "Feb-28",
    "March-08", "March-16", "March-22",
    "April-05", "April-12", "April-20", "April-29",
    "May-03", "May-11", "May-17", "May-29",
]

# Spectral x-axis zoom for rows 1 & 2  (nm)
XLIM_ZOOM = (449, 1300)


# ============================================================
# 2. FIXED PARAMETERS
# ============================================================

N_BANDS = 425
WINDOW  = 13

# Band index → wavelength (nm)
wavelengths_nm = [377.8 + 5.01 * b for b in range(N_BANDS)]

# Rolling-window centres (length = 413)
window_centers_nm = np.array([
    377 + 5.01 * ((i + i + WINDOW - 1) / 2)
    for i in range(N_BANDS - WINDOW + 1)
])

# Gray background bands for Rows 1 & 2 (same regions as the exclusion rule)
gray_regions = GRAY_REGIONS_NM

# Colorblind-safe class colours
CLASS_COLORS = {
    5:  "#0072B2",
    11: "#E69F00",
    16: "#009E73",
    20: "#56B4E9",
    31: "#D55E00",
    33: "#CC79A7",
    37: "#F0E442",
    43: "#999999",
}

CLASS_NAMES = {
    5:  "Black sage scrub",
    11: "California sagebrush scrub",
    16: "Coast live oak woodland",
    20: "Coyote brush scrub",
    31: "Menzie's golden bush scrub",
    33: "Non-native grassland",
    37: "Purple sage scrub",
    43: "Semi-natural herbaceous stands",
}

METRIC_LABELS = {
    "balanced_acc": ("Balanced accuracy", "RF balanced accuracy per spectral window"),
    "OA":           ("Overall accuracy (OA)", "RF OA per spectral window"),
    "macro_F1":     ("Macro-F1", "RF macro-F1 per spectral window"),
    "kappa":        ("Cohen's κ", "RF κ per spectral window"),
}

# PFT configuration
# "focus_class" = the single veg class highlighted in rows 1 & 2
# "rf_task"     = TASK name used by RF_windowed_v3_fixed_split.py (CSV filenames)
PFTS = [
    {"folder": "shrub",      "rf_task": "Shrub",      "title": "Shrub",
     "focus_class": 20, "all_classes": [5, 11, 20, 31, 37], "col": 0},
    {"folder": "herbaceous", "rf_task": "Herbaceous", "title": "Herbaceous",
     "focus_class": 33, "all_classes": [33, 43],             "col": 1},
    {"folder": "woodland",   "rf_task": "Tree",       "title": "Woodland (oak vs. other classes)",
     "focus_class": 16, "all_classes": [16],                 "col": 2},
]

ROW_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def shade_gray(ax, xlim=None):
    for r0, r1 in gray_regions:
        if xlim is not None:
            if r1 < xlim[0] or r0 > xlim[1]:
                continue
        ax.axvspan(r0, r1, facecolor="lightgray", alpha=1.0, zorder=20)


def fmt_spectral(ax, xlim=None, ylabel="", ylim=None, yscale="log", skip_every_second=False):
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if xlim:
        ax.set_xlim(*xlim)
        step = 50
        xticks = np.arange(int(xlim[0] / step) * step, xlim[1] + step, step)
    else:
        ax.set_xlim(380, 2500)
        xticks = np.arange(400, 2501, 100)

    ax.set_xticks(xticks)

    if skip_every_second:
        xtick_labels = [str(int(x)) if i % 2 == 0 else "" for i, x in enumerate(xticks)]
        ax.set_xticklabels(xtick_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    shade_gray(ax, xlim)
    ax.set_yscale(yscale)


def get_date_cmap(n):
    """Return n colours from turbo colourmap (early=blue, late=red)."""
    return plt.cm.turbo(np.linspace(0, 1, n))


def gray_overlap_fraction(lo, hi):
    """Fraction of the window [lo, hi] (nm) lying inside GRAY_REGIONS_NM."""
    overlap = sum(max(0.0, min(hi, r1) - max(lo, r0)) for r0, r1 in GRAY_REGIONS_NM)
    return overlap / (hi - lo)


def is_excluded(lo, hi):
    """Exclusion rule shared with RF_windowed_v3_fixed_split.py."""
    return gray_overlap_fraction(lo, hi) >= EXCLUDE_MIN_OVERLAP


# ============================================================
# 4. BUILD FIGURE
# ============================================================

if RF_METRIC not in METRIC_LABELS:
    raise ValueError(f"RF_METRIC must be one of {list(METRIC_LABELS)}")
if RF_BAND not in ("dates", "reps", "both"):
    raise ValueError("RF_BAND must be 'dates', 'reps' or 'both'")

fig = plt.figure(figsize=(18, 15))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.25)

date_colors = get_date_cmap(len(date_list))
DATE_COLOR = {d: date_colors[i] for i, d in enumerate(DATES_ISO)}   # fixed colour per date


# ------------------------------------------------------------------
# ROWS 1 & 2  — iterate over PFTs and dates, loading .pkl per date (unchanged from v2)
# ------------------------------------------------------------------

for pft in PFTS:
    col          = pft["col"]
    focus_cls    = pft["focus_class"]
    dir_klpd     = os.path.join(base_dir, pft["folder"], "stored_delta_G_13_vectors")
    dir_kernels  = os.path.join(base_dir, pft["folder"], "stored_kernel_excitation_vectors")

    klpd_files   = sorted([f for f in os.listdir(dir_klpd)    if f.endswith(".pkl")])
    kernel_files = sorted([f for f in os.listdir(dir_kernels) if f.endswith(".pkl")])

    ax_klpd   = fig.add_subplot(gs[0, col])   # Row 1
    ax_kernel = fig.add_subplot(gs[1, col])   # Row 2

    all_vals_klpd   = []
    all_vals_kernel = []

    for d, (date_tag, c) in enumerate(zip(date_list, date_colors)):

        # ---- Row 1: KLPD / ΔShape ----
        all_delta_G = load_pkl(os.path.join(dir_klpd, klpd_files[d]))

        if focus_cls in all_delta_G:
            delta_G_vec = np.array(all_delta_G[focus_cls])
            all_vals_klpd.append(delta_G_vec)
            ax_klpd.plot(window_centers_nm, delta_G_vec,
                         color=c, linewidth=1.2, alpha=0.85,
                         label=date_tag if col == 0 else "_nolegend_", zorder=2)

        # ---- Row 2: Filter excitation ----
        data_k = load_pkl(os.path.join(dir_kernels, kernel_files[d]))
        mean_per_class = data_k["mean_per_class"]

        if focus_cls in mean_per_class:
            mean_vec = mean_per_class[focus_cls].mean(axis=0)
            if hasattr(mean_vec, "numpy"):
                mean_vec = mean_vec.numpy()

            mean_vec_norm = (mean_vec - mean_vec.mean()) / (mean_vec.std() + 1e-8)
            all_vals_kernel.append(np.abs(mean_vec_norm))

            ax_kernel.plot(window_centers_nm, mean_vec_norm,
                           color=c, linewidth=1.2, alpha=0.85,
                           label=date_tag if col == 0 else "_nolegend_", zorder=2)

    # Fixed ylims per column for Row 1
    _row1_ylims = {0: (1e-10, 0.025), 1: (1e-10, 0.0008), 2: (1e-10, 0.1)}
    ylim_klpd = _row1_ylims[col]

    # ---- Row 1 formatting ----
    panel_label = ROW_LABELS[col]
    ax_klpd.set_title(
        f"{panel_label}  —  {CLASS_NAMES[focus_cls]}, ΔShape (ΔG₁₃)",
        fontsize=12, fontweight='bold', loc='left')
    fmt_spectral(ax_klpd, xlim=None,
                 ylabel=r"Shape (G13)",
                 ylim=ylim_klpd, yscale="linear",
                 skip_every_second=True)

    # ---- Row 2 formatting ----
    panel_label = ROW_LABELS[3 + col]
    ax_kernel.set_title(
        f"{panel_label}  —  {CLASS_NAMES[focus_cls]}, filter bank",
        fontsize=12, fontweight='bold', loc='left')
    fmt_spectral(ax_kernel, xlim=None,
                 ylabel="Mean excitation (z-score)",
                 ylim=(-1.25, 0.25),
                 yscale="linear",
                 skip_every_second=True)


# ------------------------------------------------------------------
# ROW 3  — Spectrally-windowed RF (output of RF_windowed_v3_fixed_split.py)
# ------------------------------------------------------------------

metric_ylabel, metric_title = METRIC_LABELS[RF_METRIC]

for pft in PFTS:
    col   = pft["col"]
    task  = pft["rf_task"]
    ax_rf = fig.add_subplot(gs[2, col])   # Row 3

    f_runs = os.path.join(base_dir, pft["folder"], RF_SUBDIR, f"RF_windowed_{task}_all_runs.csv")
    if not os.path.exists(f_runs):
        ax_rf.text(0.5, 0.5, f"No RF data found:\n{os.path.basename(f_runs)}",
                   ha='center', va='center', transform=ax_rf.transAxes, fontsize=10)
        continue

    runs = pd.read_csv(f_runs)
    is_full = runs["window"].str.startswith("Full")
    full_runs = runs[is_full]
    runs = runs[~is_full]

    # ---- Window table (order, labels, exclusion) ----
    win = (runs[["window", "lo_nm", "hi_nm"]].drop_duplicates()
           .sort_values("lo_nm").reset_index(drop=True))
    win["x"] = np.arange(len(win))
    win["excluded"] = [is_excluded(lo, hi) for lo, hi in zip(win["lo_nm"], win["hi_nm"])]
    x_of = dict(zip(win["window"], win["x"]))

    # ---- Per date and window: mean and SD across repetitions ----
    per_date = (runs.groupby(["date", "window"])[RF_METRIC]
                .agg(mean="mean", sd_reps=lambda s: s.std(ddof=SD_DDOF))
                .reset_index())
    per_date["x"] = per_date["window"].map(x_of)

    # Matrix dates × windows of per-date means (dates in chronological order)
    piv_mean = per_date.pivot(index="date", columns="x", values="mean")
    piv_sdr  = per_date.pivot(index="date", columns="x", values="sd_reps")
    dates_here = [d for d in DATES_ISO if d in piv_mean.index]
    piv_mean, piv_sdr = piv_mean.loc[dates_here], piv_sdr.loc[dates_here]
    missing_dates = [d for d in DATES_ISO if d not in dates_here]
    if missing_dates:
        print(f"[{task}] dates missing from RF output: {missing_dates}")

    xs        = piv_mean.columns.to_numpy()
    valid     = ~win.set_index("x").loc[xs, "excluded"].to_numpy()
    mean_over = piv_mean.mean(axis=0).to_numpy()
    sd_dates  = piv_mean.std(axis=0, ddof=SD_DDOF).to_numpy()
    sd_reps   = piv_sdr.mean(axis=0).to_numpy()          # mean within-date resampling SD

    # ---- Excluded windows: gray spans ----
    for x in win.loc[win["excluded"], "x"]:
        ax_rf.axvspan(x - 0.5, x + 0.5, facecolor='lightgray', alpha=1.0,
                      edgecolor='none', zorder=20)

    # ---- Scatter per date (colour fixed by date) ----
    for d in dates_here:
        y = piv_mean.loc[d].to_numpy()
        if RF_BAND == "both":
            ax_rf.errorbar(xs[valid], y[valid], yerr=piv_sdr.loc[d].to_numpy()[valid],
                           fmt='o', ms=3.5, color=DATE_COLOR[d], ecolor=DATE_COLOR[d],
                           elinewidth=0.6, capsize=0, alpha=0.8)
        else:
            ax_rf.scatter(xs[valid], y[valid], color=DATE_COLOR[d],
                          alpha=0.8, s=12, linewidth=0.5)

    # ---- Mean line + band (NaN at excluded windows so line and band break there) ----
    band = sd_reps if RF_BAND == "reps" else sd_dates
    band_label = "±1 SD (repetitions)" if RF_BAND == "reps" else "±1 SD (dates)"
    m_plot = np.where(valid, mean_over, np.nan)
    b_plot = np.where(valid, band, np.nan)
    ax_rf.plot(xs, m_plot, color='black', linewidth=0.9, alpha=0.5, label='Mean')
    ax_rf.fill_between(xs, m_plot - b_plot, m_plot + b_plot,
                       color='black', alpha=0.10, label=band_label)

    # ---- Reference lines ----
    if SHOW_FULL_SPECTRUM_REF and len(full_runs):
        full_val = full_runs.groupby("date")[RF_METRIC].mean().mean()
        ax_rf.axhline(full_val, color='black', linestyle=':', linewidth=0.9,
                      label='Full spectrum')
    if SHOW_CHANCE_LEVEL and RF_METRIC == "balanced_acc":
        n_classes = sum(c.startswith("recall_") for c in runs.columns)
        if n_classes:
            ax_rf.axhline(1.0 / n_classes, color='gray', linestyle='--', linewidth=0.8,
                          label=f'Chance (1/{n_classes})')

    # ---- Formatting ----
    panel_label = ROW_LABELS[6 + col]
    ax_rf.set_title(f"{panel_label}  —  {pft['title']}\n{metric_title}",
                    fontsize=12, fontweight='bold', loc='left')
    ax_rf.set_xlabel("Spectral window (nm)", fontsize=12)
    ax_rf.set_ylabel(metric_ylabel, fontsize=12)
    ax_rf.set_xticks(win["x"])
    ax_rf.set_xticklabels([f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(win["lo_nm"], win["hi_nm"])],
                          rotation=45, ha='right', fontsize=7)
    ax_rf.set_xlim(-0.5, len(win) - 0.5)
    ax_rf.grid(True, linestyle='--', alpha=0.4)
    ax_rf.legend(fontsize=7, loc='best', framealpha=0.85)


# ------------------------------------------------------------------
# SHARED DATE COLORBAR  (DISCRETE, with white separators + rotated labels)
# ------------------------------------------------------------------

cmap_dates = mcolors.ListedColormap(date_colors)
bounds = np.arange(len(date_list) + 1) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap_dates.N)

sm = cm.ScalarMappable(norm=norm, cmap=cmap_dates)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=fig.axes,
    orientation='vertical',
    fraction=0.015,
    pad=0.02,
    shrink=0.6,
    ticks=np.arange(len(date_list)),
    drawedges=True
)
cbar.dividers.set_color("white")
cbar.dividers.set_linewidth(2.5)
cbar.set_ticklabels(date_list, fontsize=12)
plt.setp(cbar.ax.get_yticklabels(), rotation=45, ha='left', va='center')
cbar.set_label("Date", fontsize=12, fontweight='bold')

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------

out_file = os.path.join(dir_fig_out, f"Figure_5_combined_9panels_v3_{RF_METRIC}_band-{RF_BAND}.png")
plt.tight_layout()
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {out_file}")

