"""
MI(Delta_G, RCF-excitation) per plot, aggregated across dates
==================================================================

For each of the 128 ground-truth plots, computes sliding-window mutual
information (Kraskov et al. 2004, via sklearn's mutual_info_regression)
between that plot's Delta_G_13(lambda) vector and its RCF-excitation
vector, for each date on which BOTH vectors are available. Aggregates
across dates as mean (solid line) +/- 1 SD (shaded ribbon), and overlays
a MI(Delta_G, random noise) baseline for reference.

Inputs: the pickled per-plot, per-date vectors saved by
deltaG_per_plot_per_date.py and rcf_excitation_per_plot_per_date.py.
Both are length-413 vectors (425 bands, KLPD/RCF window = 13), and the
MI sliding window here (26) operates over THIS position axis, exactly
matching the reference template's convention (window_size=26 is an MI
parameter, not a re-windowing of the raw 425-band spectrum).

Requirements:
    pip install numpy matplotlib scikit-learn
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DELTAG_PKL = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_deltaG_per-date\deltaG_plot_date_vectors.pkl"
RCF_PKL = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_RCF-excitations\rcf_plot_date_vectors.pkl"
OUT_DIR = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_MI_deltaG-vs-RCF\plots_aggregated"
os.makedirs(OUT_DIR, exist_ok=True)

MI_WINDOW = 26     # sliding window for the MI computation itself (over
                   # the 413-length position axis), as specified
N_NEIGHBORS = 5
N_SURROGATES = 100       # circular-shift surrogates for the significance threshold
SIGNIFICANCE_PCTL = 95   # percentile of the surrogate null distribution
NOISE_SEED = 0     # for reproducibility of the MI(X, noise) baseline

# ----------------------------------------------------------------------
# 1. Load the two saved vector sets
# ----------------------------------------------------------------------

with open(DELTAG_PKL, "rb") as f:
    deltaG_data = pickle.load(f)
with open(RCF_PKL, "rb") as f:
    rcf_data = pickle.load(f)

deltaG_plot_date_curve = deltaG_data["deltaG_plot_date_curve"]
rcf_plot_date_curve = rcf_data["rcf_plot_date_curve"]
window_centers_nm = deltaG_data["window_centers_nm"]   # length 413, shared

assert len(window_centers_nm) == len(rcf_data["window_centers_nm"]), \
    "Delta_G and RCF window_centers_nm differ in length -- check both " \
    "scripts used the same WINDOW (=13) and N_BANDS."

n_bands = len(window_centers_nm)   # 413, position-axis length (matches
                                    # the reference template's reuse of
                                    # `n_bands` for the ΔG-array length)
print(f"Loaded vectors: {len(deltaG_plot_date_curve)} plots (Delta_G), "
      f"{len(rcf_plot_date_curve)} plots (RCF)")

# ----------------------------------------------------------------------
# 2. Sliding-window MI (as given)
# ----------------------------------------------------------------------

def compute_mutual_information(X, Y, n_neighbors=5):
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    return mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=0)[0]


def sliding_window_mi(X, Y, window_size=26, n_neighbors=5):
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
    """Theiler et al. (1992) circular-shift surrogates: preserves each
    curve's own autocorrelation/smoothness while destroying its
    cross-alignment with X -- the correct null for two smooth,
    autocorrelated curves (unlike i.i.d. noise, which is too weak a null)."""
    rng = rng or np.random.default_rng()
    n = len(Y)
    shifts = rng.integers(1, n, size=n_surrogates)
    return np.stack([np.roll(Y, s) for s in shifts])


def mi_significance_threshold(X, Y, window_size=26, n_neighbors=5,
                               n_surrogates=500, percentile=95, rng=None):
    """95th-percentile MI under the circular-shift null, at each window
    position. Observed MI above this curve is unlikely to arise merely
    from each curve's own smoothness/autocorrelation."""
    surrogates = circular_shift_surrogates(Y, n_surrogates, rng)
    null_mi = np.stack([
        sliding_window_mi(X, Y_surr, window_size, n_neighbors)
        for Y_surr in surrogates
    ])
    return np.percentile(null_mi, percentile, axis=0)


# MI window centres: same formula convention as the reference template,
# applied over the 413-length position axis (n_bands here = 413, not 425).
mi_window_centers_nm = np.array([
    377.8 + 5.01 * ((i + (i + MI_WINDOW - 1)) / 2)
    for i in range(n_bands - MI_WINDOW + 1)
])
print(f"MI curve length: {len(mi_window_centers_nm)} "
      f"({n_bands} - {MI_WINDOW} + 1)")

# ----------------------------------------------------------------------
# 3. Per plot: MI(Delta_G, RCF) and MI(Delta_G, noise) per available
#    date, then aggregate as mean +/- SD across dates
# ----------------------------------------------------------------------

rng = np.random.RandomState(NOISE_SEED)
rng_gen = np.random.default_rng(NOISE_SEED)   # circular_shift_surrogates needs
                                               # the newer Generator API (.integers),
                                               # not RandomState -- kept separate
                                               # from `rng` above for that reason

all_plots = sorted(set(deltaG_plot_date_curve.keys()) & set(rcf_plot_date_curve.keys()))
plots_deltaG_only = sorted(set(deltaG_plot_date_curve.keys()) - set(rcf_plot_date_curve.keys()))
plots_rcf_only = sorted(set(rcf_plot_date_curve.keys()) - set(deltaG_plot_date_curve.keys()))

print(f"\n{len(all_plots)} plots present in BOTH vector sets.")
if plots_deltaG_only:
    print(f"{len(plots_deltaG_only)} plot(s) only in Delta_G vectors "
          f"(skipped, no RCF counterpart): {plots_deltaG_only}")
if plots_rcf_only:
    print(f"{len(plots_rcf_only)} plot(s) only in RCF vectors "
          f"(skipped, no Delta_G counterpart): {plots_rcf_only}")

import time
_t0 = time.time()
_probe_X = next(iter(next(iter(deltaG_plot_date_curve.values())).values()))
_probe_Y = next(iter(next(iter(rcf_plot_date_curve.values())).values()))
_ = mi_significance_threshold(_probe_X, _probe_Y, MI_WINDOW, N_NEIGHBORS,
                               n_surrogates=N_SURROGATES, percentile=SIGNIFICANCE_PCTL,
                               rng=rng_gen)
_dt = time.time() - _t0
print(f"\nSurrogate-threshold timing probe: {_dt:.1f}s for one plot "
      f"({N_SURROGATES} surrogates) -> est. total ~{_dt * len(all_plots) / 60:.1f} "
      f"min for {len(all_plots)} plots (threshold computation only).")
print(f"If this is too slow, lower N_SURROGATES (e.g. 100-200) in CONFIG.\n")

# Accumulators for the population-level (across-plots) aggregation done
# after the main loop -- one mi_mean curve and one mi_threshold curve
# per plot, keyed by plot name.
mi_mean_all = {}
mi_threshold_all = {}

for pi, plot_name in enumerate(all_plots):
    dates_deltaG = set(deltaG_plot_date_curve[plot_name].keys())
    dates_rcf = set(rcf_plot_date_curve[plot_name].keys())
    dates_available = sorted(dates_deltaG & dates_rcf)

    dates_mismatch = (dates_deltaG ^ dates_rcf)  # symmetric difference
    if dates_mismatch:
        print(f"  NOTE: {plot_name} has {len(dates_mismatch)} date(s) "
              f"present in only one of the two vector sets "
              f"(using the {len(dates_available)} dates common to both): "
              f"{sorted(dates_mismatch)}")

    if len(dates_available) == 0:
        print(f"  Skipping {plot_name}: no dates with both vectors available.")
        continue

    mi_curves = []
    mi_noise_curves = []
    for date_tag in dates_available:
        X = deltaG_plot_date_curve[plot_name][date_tag]
        Y = rcf_plot_date_curve[plot_name][date_tag]

        mi = sliding_window_mi(X, Y, MI_WINDOW, N_NEIGHBORS)
        mi_curves.append(mi)

        noise = rng.randn(len(X))
        mi_noise = sliding_window_mi(X, noise, MI_WINDOW, N_NEIGHBORS)
        mi_noise_curves.append(mi_noise)

    mi_array = np.array(mi_curves)             # [n_dates_available, len(mi_window_centers_nm)]
    mi_noise_array = np.array(mi_noise_curves)

    mi_mean = mi_array.mean(axis=0)
    mi_sd = mi_array.std(axis=0)
    mi_noise_mean = mi_noise_array.mean(axis=0)

    # --- Significance threshold via circular-shift surrogates ---
    # Computed ONCE per plot, from the mean Delta_G and mean RCF curves
    # across this plot's available dates (the same curves underlying
    # mi_mean above) -- not per individual date. Running 500 surrogates
    # x 388 window positions PER DATE PER PLOT (as a literal per-date
    # implementation would require) is computationally infeasible at
    # this scale (~300 million MI evaluations total); this reduces it to
    # one surrogate test per plot (~121 x 500 x 388 evaluations), and is
    # directly comparable to mi_mean since both come from the same
    # aggregated curves.
    X_mean = np.mean([deltaG_plot_date_curve[plot_name][d] for d in dates_available], axis=0)
    Y_mean = np.mean([rcf_plot_date_curve[plot_name][d] for d in dates_available], axis=0)

    mi_threshold = mi_significance_threshold(
        X_mean, Y_mean, MI_WINDOW, N_NEIGHBORS,
        n_surrogates=N_SURROGATES, percentile=SIGNIFICANCE_PCTL, rng=rng_gen)

    mi_mean_all[plot_name] = mi_mean
    mi_threshold_all[plot_name] = mi_threshold

    # --- Plot: solid mean line + shaded +/-1 SD ribbon, per plot ---
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.fill_between(mi_window_centers_nm, mi_mean - mi_sd, mi_mean + mi_sd,
                    color="orange", alpha=0.3,
                    label=f"±1 SD (n={len(dates_available)} dates)")
    ax.plot(mi_window_centers_nm, mi_mean, color="orange", linewidth=2,
            label=r"MI($\Delta G_{13}$, RCF excitation)")
    ax.plot(mi_window_centers_nm, mi_noise_mean, linestyle="--", color="black",
            linewidth=1.5, label="MI(X, noise)")
    ax.plot(mi_window_centers_nm, mi_threshold, linestyle=":", color="red",
            linewidth=1.8,
            label=f"{SIGNIFICANCE_PCTL}th pct, circular-shift surrogates "
                  f"(n={N_SURROGATES})")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mutual Information")

    xticks_major = np.arange(400, 2501, 100)
    xticks_minor = np.arange(400, 2501, 50)
    ax.set_xticks(xticks_major)
    ax.set_xticks(xticks_minor, minor=True)
    ax.grid(True, which="major", alpha=0.5)
    ax.grid(True, which="minor", linestyle="--", alpha=0.3)
    ax.set_xlim(mi_window_centers_nm[0], mi_window_centers_nm[-1])

    title = (f"{plot_name}: MI($\\Delta G_{{13}}$, RCF excitation)  "
             f"[{len(dates_available)}/{len(rcf_data['date_tags'])} dates]")
    ax.set_title(title, fontsize=9)

    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, f"MI_deltaG_RCF_{plot_name}.png")
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)

    if (pi + 1) % 10 == 0 or (pi + 1) == len(all_plots):
        print(f"  [{pi + 1}/{len(all_plots)}] wrote {out_fig}")

print(f"\nDone. {len(all_plots)} MI figures written to {OUT_DIR}")

# ----------------------------------------------------------------------
# 4. Population-level aggregation across plots
# ----------------------------------------------------------------------
# Per-plot pointwise significance is a weak, single-shot test (~5% of
# positions cross the 95th-pct threshold by chance even with zero real
# coupling, and adjacent positions are correlated so chance crossings
# cluster into short runs). The much stronger question: at each
# wavelength, what FRACTION of the N plots exceed their own null
# threshold -- and is that fraction itself higher than the 5% expected
# by chance? This aggregates statistical power across the whole plot
# population instead of relying on any single plot's noisy curve.

from scipy.stats import binomtest

plots_with_results = sorted(mi_mean_all.keys())
n_plots = len(plots_with_results)
print(f"\nAggregating across {n_plots} plots with results...")

mi_mean_matrix = np.stack([mi_mean_all[p] for p in plots_with_results])       # [n_plots, n_positions]
mi_threshold_matrix = np.stack([mi_threshold_all[p] for p in plots_with_results])

exceed_matrix = mi_mean_matrix > mi_threshold_matrix    # [n_plots, n_positions] boolean
exceed_fraction = exceed_matrix.mean(axis=0)            # per position, fraction of plots exceeding
exceed_count = exceed_matrix.sum(axis=0).astype(int)    # per position, count of plots exceeding

NULL_PROPORTION = (100 - SIGNIFICANCE_PCTL) / 100.0     # 0.05 for a 95th-pct threshold

p_values = np.array([
    binomtest(int(k), n_plots, NULL_PROPORTION, alternative="greater").pvalue
    for k in exceed_count
])


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (significant_mask, p_fdr_cutoff)."""
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


FDR_ALPHA = 0.05
significant_fdr, p_fdr_cutoff = fdr_bh(p_values, alpha=FDR_ALPHA)
print(f"FDR (Benjamini-Hochberg, alpha={FDR_ALPHA}) p-value cutoff: {p_fdr_cutoff:.4g}")
print(f"{significant_fdr.sum()} / {len(p_values)} positions significant after FDR correction.")

if significant_fdr.any():
    sig_wavelengths = mi_window_centers_nm[significant_fdr]
    # Report as contiguous ranges rather than a long list of individual points
    edges = np.where(np.diff(np.where(significant_fdr)[0]) > 1)[0]
    idx_sig = np.where(significant_fdr)[0]
    starts = [idx_sig[0]] + [idx_sig[i + 1] for i in edges]
    ends = [idx_sig[i] for i in edges] + [idx_sig[-1]]
    print("Significant (FDR-corrected) wavelength region(s):")
    for s, e in zip(starts, ends):
        print(f"    {mi_window_centers_nm[s]:.1f} - {mi_window_centers_nm[e]:.1f} nm "
              f"(exceed fraction {exceed_fraction[s:e+1].mean()*100:.1f}%)")
else:
    print("No positions survived FDR correction -- no reproducible, "
          "population-level coupling detected at this significance level.")

# --- Save results table ---
import pandas as pd

results_df = pd.DataFrame({
    "wavelength_nm": mi_window_centers_nm,
    "exceed_fraction": exceed_fraction,
    "exceed_count": exceed_count,
    "n_plots": n_plots,
    "p_value_binomial": p_values,
    "significant_fdr": significant_fdr,
})
results_csv = os.path.join(OUT_DIR, "population_level_MI_significance.csv")
results_df.to_csv(results_csv, index=False)
print(f"Wrote {results_csv}")

# --- Summary figure: exceed fraction vs wavelength, chance level, FDR-significant regions ---
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(mi_window_centers_nm, exceed_fraction * 100, color="darkred",
        linewidth=1.8, label=f"% of {n_plots} plots exceeding their own "
                              f"{SIGNIFICANCE_PCTL}th-pct null threshold")
ax.axhline(NULL_PROPORTION * 100, linestyle="--", color="black", linewidth=1.3,
           label=f"Chance level ({NULL_PROPORTION*100:.0f}%)")

if significant_fdr.any():
    for s, e in zip(starts, ends):
        ax.axvspan(mi_window_centers_nm[s], mi_window_centers_nm[e],
                   color="red", alpha=0.15, zorder=0,
                   label=f"FDR-significant (BH, alpha={FDR_ALPHA})" if (s, e) == (starts[0], ends[0]) else None)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("% of plots exceeding null threshold")
ax.set_xlim(mi_window_centers_nm[0], mi_window_centers_nm[-1])
ax.set_xticks(np.arange(400, 2501, 100))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, alpha=0.3)
ax.set_title(f"Population-level MI significance: {n_plots} plots, "
             f"{SIGNIFICANCE_PCTL}th-pct circular-shift threshold")
ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
summary_fig = os.path.join(OUT_DIR, "population_level_MI_significance.png")
plt.savefig(summary_fig, dpi=200)
plt.close(fig)
print(f"Wrote {summary_fig}")
