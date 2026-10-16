"""
MI(Delta_G, RCF-excitation): window-size sweep
==================================================

Repeats the population-level MI significance analysis at several window
sizes (26, 40, 60, 100 bands) to test the hypothesis that the windowed
MI test is underpowered at window=26 and that population-level
significance increases as the window widens.

Per-plot figures are SKIPPED during the sweep (121 plots x 4 window
sizes = 484 figures would add substantial time/disk cost for no
analytical benefit here); only the population-level outputs are kept,
since that's what this comparison actually needs. Set
SAVE_PER_PLOT_FIGURES = True to restore them.

Requirements:
    pip install numpy matplotlib scikit-learn scipy pandas
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import binomtest

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DELTAG_PKL = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_deltaG_per-date\deltaG_plot_date_vectors.pkl"
RCF_PKL = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_RCF-excitations\rcf_plot_date_vectors.pkl"
OUT_DIR = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_MI_deltaG-vs-RCF\v_2_window-size_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZES = [26, 40, 60, 100]
N_NEIGHBORS = 5
N_SURROGATES = 100
SIGNIFICANCE_PCTL = 95
FDR_ALPHA = 0.05
NOISE_SEED = 0

SAVE_PER_PLOT_FIGURES = False

rng_gen = np.random.default_rng(NOISE_SEED)

# ----------------------------------------------------------------------
# 1. Load the two saved vector sets (once, shared across all window sizes)
# ----------------------------------------------------------------------

with open(DELTAG_PKL, "rb") as f:
    deltaG_data = pickle.load(f)
with open(RCF_PKL, "rb") as f:
    rcf_data = pickle.load(f)

deltaG_plot_date_curve = deltaG_data["deltaG_plot_date_curve"]
rcf_plot_date_curve = rcf_data["rcf_plot_date_curve"]
window_centers_nm_413 = deltaG_data["window_centers_nm"]
date_tags_all = deltaG_data["date_tags"]

assert len(window_centers_nm_413) == len(rcf_data["window_centers_nm"]), \
    "Delta_G and RCF window_centers_nm differ in length."

n_bands = len(window_centers_nm_413)

all_plots = sorted(set(deltaG_plot_date_curve.keys()) & set(rcf_plot_date_curve.keys()))
print(f"{len(all_plots)} plots present in both vector sets.\n")

# ----------------------------------------------------------------------
# 2. Core functions
# ----------------------------------------------------------------------

def compute_mutual_information(X, Y, n_neighbors=5):
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    return mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=0)[0]


def sliding_window_mi(X, Y, window_size, n_neighbors=5):
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)
    mi_values = []
    for start in range(n - window_size + 1):
        end = start + window_size
        mi_values.append(
            compute_mutual_information(X[start:end], Y[start:end], n_neighbors)
        )
    return np.array(mi_values)


def circular_shift_surrogates(Y, n_surrogates=500, rng=None):
    rng = rng or np.random.default_rng()
    n = len(Y)
    shifts = rng.integers(1, n, size=n_surrogates)
    return np.stack([np.roll(Y, s) for s in shifts])


def mi_significance_threshold(X, Y, window_size, n_neighbors=5,
                               n_surrogates=500, percentile=95, rng=None):
    surrogates = circular_shift_surrogates(Y, n_surrogates, rng)
    null_mi = np.stack([
        sliding_window_mi(X, Y_surr, window_size, n_neighbors)
        for Y_surr in surrogates
    ])
    return np.percentile(null_mi, percentile, axis=0)


def fdr_bh(pvals, alpha=0.05):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if below.any():
        cutoff = ranked[np.max(np.where(below))]
    else:
        cutoff = 0.0
    return pvals <= cutoff, cutoff


# ----------------------------------------------------------------------
# 3. One full analysis pass for a given window size
# ----------------------------------------------------------------------

def run_mi_analysis(window_size, out_dir, save_per_plot_figures=False):
    os.makedirs(out_dir, exist_ok=True)

    mi_window_centers_nm = np.array([
        377.8 + 5.01 * ((i + (i + window_size - 1)) / 2)
        for i in range(n_bands - window_size + 1)
    ])
    print(f"\n=== window_size = {window_size} "
          f"(MI curve length {len(mi_window_centers_nm)}) ===")

    _t0 = time.time()
    _probe_X = next(iter(next(iter(deltaG_plot_date_curve.values())).values()))
    _probe_Y = next(iter(next(iter(rcf_plot_date_curve.values())).values()))
    _ = mi_significance_threshold(_probe_X, _probe_Y, window_size, N_NEIGHBORS,
                                   n_surrogates=N_SURROGATES, percentile=SIGNIFICANCE_PCTL,
                                   rng=rng_gen)
    _dt = time.time() - _t0
    print(f"Timing probe: {_dt:.1f}s/plot -> est. total ~"
          f"{_dt * len(all_plots) / 60:.1f} min for {len(all_plots)} plots.")

    mi_mean_all = {}
    mi_threshold_all = {}

    for pi, plot_name in enumerate(all_plots):
        dates_available = sorted(set(deltaG_plot_date_curve[plot_name].keys())
                                  & set(rcf_plot_date_curve[plot_name].keys()))
        if len(dates_available) == 0:
            continue

        mi_curves = []
        for date_tag in dates_available:
            X = deltaG_plot_date_curve[plot_name][date_tag]
            Y = rcf_plot_date_curve[plot_name][date_tag]
            mi_curves.append(sliding_window_mi(X, Y, window_size, N_NEIGHBORS))

        mi_array = np.array(mi_curves)
        mi_mean = mi_array.mean(axis=0)

        X_mean = np.mean([deltaG_plot_date_curve[plot_name][d] for d in dates_available], axis=0)
        Y_mean = np.mean([rcf_plot_date_curve[plot_name][d] for d in dates_available], axis=0)

        mi_threshold = mi_significance_threshold(
            X_mean, Y_mean, window_size, N_NEIGHBORS,
            n_surrogates=N_SURROGATES, percentile=SIGNIFICANCE_PCTL, rng=rng_gen)

        mi_mean_all[plot_name] = mi_mean
        mi_threshold_all[plot_name] = mi_threshold

        if save_per_plot_figures:
            mi_sd = mi_array.std(axis=0)
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.fill_between(mi_window_centers_nm, mi_mean - mi_sd, mi_mean + mi_sd,
                            color="orange", alpha=0.3)
            ax.plot(mi_window_centers_nm, mi_mean, color="orange", linewidth=2,
                    label=r"MI($\Delta G$, RCF excitation)")
            ax.plot(mi_window_centers_nm, mi_threshold, linestyle=":", color="red",
                    linewidth=1.8, label=f"{SIGNIFICANCE_PCTL}th pct surrogate")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Mutual Information")
            ax.set_title(f"{plot_name} (window={window_size})", fontsize=9)
            ax.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"MI_{plot_name}_w{window_size}.png"), dpi=150)
            plt.close(fig)

        if (pi + 1) % 20 == 0 or (pi + 1) == len(all_plots):
            print(f"  [{pi + 1}/{len(all_plots)}] plots processed")

    # --- Population-level aggregation ---
    plots_with_results = sorted(mi_mean_all.keys())
    n_plots = len(plots_with_results)

    mi_mean_matrix = np.stack([mi_mean_all[p] for p in plots_with_results])
    mi_threshold_matrix = np.stack([mi_threshold_all[p] for p in plots_with_results])

    exceed_matrix = mi_mean_matrix > mi_threshold_matrix
    exceed_fraction = exceed_matrix.mean(axis=0)
    exceed_count = exceed_matrix.sum(axis=0).astype(int)

    null_proportion = (100 - SIGNIFICANCE_PCTL) / 100.0
    p_values = np.array([
        binomtest(int(k), n_plots, null_proportion, alternative="greater").pvalue
        for k in exceed_count
    ])
    significant_fdr, p_fdr_cutoff = fdr_bh(p_values, alpha=FDR_ALPHA)

    n_sig_positions = int(significant_fdr.sum())
    print(f"window={window_size}: {n_sig_positions} / {len(p_values)} positions "
          f"FDR-significant (cutoff p={p_fdr_cutoff:.4g}), "
          f"max exceed fraction = {exceed_fraction.max()*100:.1f}%, "
          f"mean exceed fraction = {exceed_fraction.mean()*100:.1f}%")

    results_df = pd.DataFrame({
        "wavelength_nm": mi_window_centers_nm,
        "exceed_fraction": exceed_fraction,
        "exceed_count": exceed_count,
        "n_plots": n_plots,
        "p_value_binomial": p_values,
        "significant_fdr": significant_fdr,
    })
    results_csv = os.path.join(out_dir, f"population_level_MI_significance_w{window_size}.csv")
    results_df.to_csv(results_csv, index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(mi_window_centers_nm, exceed_fraction * 100, color="darkred", linewidth=1.8,
            label=f"% of {n_plots} plots exceeding null threshold")
    ax.axhline(null_proportion * 100, linestyle="--", color="black", linewidth=1.3,
               label=f"Chance level ({null_proportion*100:.0f}%)")
    if significant_fdr.any():
        idx_sig = np.where(significant_fdr)[0]
        edges = np.where(np.diff(idx_sig) > 1)[0]
        starts = [idx_sig[0]] + [idx_sig[i + 1] for i in edges]
        ends = [idx_sig[i] for i in edges] + [idx_sig[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(mi_window_centers_nm[s], mi_window_centers_nm[e],
                       color="red", alpha=0.15, zorder=0,
                       label="FDR-significant" if (s, e) == (starts[0], ends[0]) else None)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("% of plots exceeding null threshold")
    ax.set_xticks(np.arange(400, 2501, 100))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Population-level MI significance, window={window_size} bands")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"population_level_MI_significance_w{window_size}.png"), dpi=200)
    plt.close(fig)

    return {
        "window_size": window_size,
        "mi_window_centers_nm": mi_window_centers_nm,
        "exceed_fraction": exceed_fraction,
        "significant_fdr": significant_fdr,
        "n_sig_positions": n_sig_positions,
        "n_total_positions": len(p_values),
        "max_exceed_fraction": exceed_fraction.max(),
        "mean_exceed_fraction": exceed_fraction.mean(),
        "n_plots": n_plots,
    }


# ----------------------------------------------------------------------
# 4. Run the sweep
# ----------------------------------------------------------------------

sweep_results = []
for w in WINDOW_SIZES:
    w_out_dir = os.path.join(OUT_DIR, f"window_{w}")
    result = run_mi_analysis(w, w_out_dir, save_per_plot_figures=SAVE_PER_PLOT_FIGURES)
    sweep_results.append(result)

# ----------------------------------------------------------------------
# 5. Comparison across window sizes
# ----------------------------------------------------------------------

summary_df = pd.DataFrame([{
    "window_size": r["window_size"],
    "n_plots": r["n_plots"],
    "n_positions_total": r["n_total_positions"],
    "n_positions_fdr_significant": r["n_sig_positions"],
    "pct_positions_fdr_significant": 100 * r["n_sig_positions"] / r["n_total_positions"],
    "max_exceed_fraction_pct": r["max_exceed_fraction"] * 100,
    "mean_exceed_fraction_pct": r["mean_exceed_fraction"] * 100,
} for r in sweep_results])

summary_csv = os.path.join(OUT_DIR, "window_sweep_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"\nWrote {summary_csv}")
print(summary_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(WINDOW_SIZES)))
for r, col in zip(sweep_results, colors):
    ax.plot(r["mi_window_centers_nm"], r["exceed_fraction"] * 100,
            color=col, linewidth=1.8, label=f"window = {r['window_size']} bands")
ax.axhline(5.0, linestyle="--", color="black", linewidth=1.2, label="Chance level (5%)")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("% of plots exceeding null threshold")
ax.set_xticks(np.arange(400, 2501, 100))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, alpha=0.3)
ax.set_title("MI population-level significance vs. window size")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
comparison_fig = os.path.join(OUT_DIR, "window_sweep_comparison.png")
plt.savefig(comparison_fig, dpi=200)
plt.close(fig)
print(f"Wrote {comparison_fig}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(summary_df["window_size"], summary_df["pct_positions_fdr_significant"],
        marker="o", color="darkred", linewidth=2)
ax.set_xlabel("MI window size (bands)")
ax.set_ylabel("% of spectral positions FDR-significant")
ax.set_title("Does population-level significance increase with window size?")
ax.grid(True, alpha=0.3)
plt.tight_layout()
trend_fig = os.path.join(OUT_DIR, "window_sweep_trend.png")
plt.savefig(trend_fig, dpi=200)
plt.close(fig)
print(f"Wrote {trend_fig}")
