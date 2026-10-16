"""
Extrema co-location test: Delta_G vs RCF-excitation (curvature-based detection)
=================================================================================

Tests whether the strong local extrema (peaks AND troughs, sign-agnostic)
of Delta_G_13(lambda) and RCF-excitation co-occur at the same spectral
positions more often/tightly than expected by chance.

Unlike the windowed-MI test (which asks "does knowing X reduce local
uncertainty about Y in a narrow 26-band window"), this asks a more global
question directly matching visual inspection: "do the LANDMARK features
(strong peaks and dips) of the two curves line up in wavelength?"

Sign-agnostic by design: RCF-excitation is mostly positive except at the
red-edge, which appears as its strongest NEGATIVE excursion. Extrema are
kept only if their local curvature is a statistical outlier relative to
the curve's own turning-point population, so peaks and troughs are
detected and treated identically -- we care about the magnitude of local
deviation, not its sign.

Per (plot, date):
    1. Detect extrema (peaks + troughs) in both curves as genuine turning
       points, kept only if their local curvature (2nd derivative
       magnitude) is a statistical outlier relative to the curve's own
       turning-point population -- a robust modified z-score (Iglewicz &
       Hoaglin, 1993), not a user-chosen prominence-fraction threshold.
    2. Compute a symmetric nearest-neighbor wavelength offset between the
       two extrema sets (a Chamfer-style alignment score, in nm -- lower
       means tighter co-location).
    3. Build a null distribution via circular-shift surrogates of the RCF
       curve (Theiler et al. 1992): shift, re-detect extrema, recompute
       the same offset score, repeat. Circular shift preserves each
       curve's own value distribution exactly, so the SAME curvature
       criterion applies unchanged to every surrogate.
    4. One-sided p-value: fraction of surrogate scores at least as tight
       as the observed score.

Per plot: per-date p-values combined via Fisher's method into one
plot-level p-value. A population-level step pools the actual matched
extrema-pair midpoint wavelengths from every individually-significant
(plot, date) test, to show WHERE in the spectrum tight co-location
concentrates across the whole dataset.

Requirements:
    pip install numpy scipy matplotlib pandas
"""

import os
import pickle
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import combine_pvalues

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

DELTAG_PKL = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_deltaG_per-date\deltaG_plot_date_vectors.pkl"
RCF_PKL = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_RCF-excitations\rcf_plot_date_vectors.pkl"
IN_DIR = r"D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\mosaics_labeled\hdf_file\hsi_SHIFT_plots_intersection"
OUT_DIR = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_extrema_colocation\based_on_curvature_v0"
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_ID_COL = "plot_plot_name"

# Curvature-based extrema detection replaces the prominence-fraction
# threshold used in the original version (0.15/0.30/0.50, swept by hand).
# MODIFIED_Z_THRESH is a single, fixed, literature-standard cutoff
# (Iglewicz & Hoaglin, 1993, "How to Detect and Handle Outliers") applied
# to the modified z-score of each turning point's curvature (2nd
# derivative) magnitude, self-calibrated per curve/segment -- so there is
# no analyst-tuned parameter left to sweep across runs.
MODIFIED_Z_THRESH = 3.5

N_SURROGATES = 500           # cheap here (no k-NN estimator), so the
                             # full 500 from the MI script is affordable
                             # per (plot, date), not just once per plot
NOISE_SEED = 0
FDR_ALPHA = 0.05             # for the population-level pooling step

# Sensor noise / atmospheric absorption bands: extrema here are
# instrumental artifacts, not real spectral features, and must be
# excluded from extrema detection entirely (not just visually flagged).
GRAY_REGIONS = [
    (0, 450),
    (1325, 1450),
    (1775, 2000),
    (2450, 2510),
]

rng_gen = np.random.default_rng(NOISE_SEED)

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
date_tags_all = deltaG_data["date_tags"]

assert len(window_centers_nm) == len(rcf_data["window_centers_nm"]), \
    "Delta_G and RCF window_centers_nm differ in length."

print(f"Loaded vectors: {len(deltaG_plot_date_curve)} plots (Delta_G), "
      f"{len(rcf_plot_date_curve)} plots (RCF)")

all_plots = sorted(set(deltaG_plot_date_curve.keys()) & set(rcf_plot_date_curve.keys()))
print(f"{len(all_plots)} plots present in both vector sets.\n")

# ----------------------------------------------------------------------
# 2. Rebuild plot attributes (for figure titles) from the source parquets
#    -- these were not saved in the vector pickles, only the curves were.
# ----------------------------------------------------------------------

all_files = sorted(glob.glob(os.path.join(IN_DIR, "*.parquet")))
all_files = [f for f in all_files if "ALL_DATES" not in os.path.basename(f)]
date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
date_files = {}
for f in all_files:
    m = date_pattern.search(os.path.basename(f))
    if m:
        date_files[m.group(1)] = f

plot_attrs = {}
for date_tag, fpath in date_files.items():
    df_attr = pd.read_parquet(fpath)
    attr_cols = [c for c in df_attr.columns if c.startswith("plot_") or c == "veg_class"]
    for plot_name, g in df_attr.groupby(PLOT_ID_COL):
        if plot_name not in plot_attrs:
            plot_attrs[plot_name] = g.iloc[0][attr_cols].to_dict()

# ----------------------------------------------------------------------
# 3. Extrema detection and co-location test
# ----------------------------------------------------------------------

def get_valid_segments(x_axis, gray_regions):
    """Indices of x_axis OUTSIDE all gray_regions, split into contiguous
    runs. Precomputed once (depends only on x_axis/gray_regions, not on
    curve values) since it's identical across all curves and surrogates."""
    mask = np.ones(len(x_axis), dtype=bool)
    for lo, hi in gray_regions:
        mask &= ~((x_axis >= lo) & (x_axis <= hi))
    valid_idx = np.flatnonzero(mask)
    if len(valid_idx) == 0:
        return []
    splits = np.where(np.diff(valid_idx) > 1)[0] + 1
    return np.split(valid_idx, splits)


def detect_extrema_curvature(curve, valid_segments, modified_z_thresh=3.5):
    """Peaks AND troughs, sign-agnostic -- detected as genuine local
    turning points (strict sign change in the first difference), then
    kept only if their curvature (2nd derivative magnitude) is a robust
    statistical outlier relative to the curvature of every OTHER turning
    point in this curve's valid segments (modified z-score, Iglewicz &
    Hoaglin 1993). This replaces the prominence-fraction criterion:
    instead of an analyst-chosen fraction of the curve's peak-to-peak
    range, the cutoff self-calibrates to each curve's own curvature
    distribution, using one fixed, literature-standard z-threshold.

    Mirrors the original prominence-based function's structure exactly:
    the reference statistic (there: peak-to-peak range; here: median/MAD
    of curvature) is computed once, GLOBALLY across all valid segments
    combined, then the same threshold is applied within each segment
    independently -- so gray-region values never contribute to the
    reference statistic, and no spurious turning point is created at a
    splice boundary between segments."""
    curve = np.asarray(curve)
    if len(valid_segments) == 0:
        return np.array([], dtype=int)

    # Pass 1: collect every raw turning point (no filtering yet) and its
    # curvature magnitude, from every valid segment.
    seg_turning = []
    all_abs_curv = []

    for seg in valid_segments:
        if len(seg) < 3:
            continue   # too short to contain an interior turning point
        seg_curve = curve[seg]

        d = np.diff(seg_curve)
        is_max = np.zeros(len(seg_curve), dtype=bool)
        is_min = np.zeros(len(seg_curve), dtype=bool)
        is_max[1:-1] = (d[:-1] > 0) & (d[1:] < 0)
        is_min[1:-1] = (d[:-1] < 0) & (d[1:] > 0)
        turning_idx = np.flatnonzero(is_max | is_min)
        if len(turning_idx) == 0:
            continue

        curvature = (seg_curve[turning_idx - 1] - 2 * seg_curve[turning_idx]
                     + seg_curve[turning_idx + 1])
        abs_curv = np.abs(curvature)
        seg_turning.append((seg, turning_idx, abs_curv))
        all_abs_curv.append(abs_curv)

    if not seg_turning:
        return np.array([], dtype=int)

    # Global reference statistic (median/MAD of curvature), computed once
    # across ALL valid segments combined -- same role as rng_span in the
    # prominence-based version.
    all_abs_curv = np.concatenate(all_abs_curv)
    med = np.median(all_abs_curv)
    mad = np.median(np.abs(all_abs_curv - med))

    all_extrema = []
    for seg, turning_idx, abs_curv in seg_turning:
        if mad == 0:
            # Degenerate case (near-identical curvature throughout):
            # still self-calibrated, no fixed user-set fraction -- fall
            # back to strictly-above-median turning points.
            keep = abs_curv > med
        else:
            modified_z = 0.6745 * (abs_curv - med) / mad
            keep = modified_z > modified_z_thresh
        all_extrema.extend(seg[turning_idx[keep]])

    return np.sort(np.array(all_extrema, dtype=int))


def symmetric_nn_offset_nm(idx_a, idx_b, x_axis):
    """Chamfer-style symmetric mean nearest-neighbor distance (nm)
    between two sets of extrema indices. Lower = tighter co-location.
    Returns (score, matched_pairs) where matched_pairs is a list of
    (x_a, x_b, midpoint, offset) for the a->b direction, used later for
    population-level pooling of WHERE matches occur."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan, []
    xa = x_axis[idx_a]
    xb = x_axis[idx_b]

    d_ab = np.abs(xa[:, None] - xb[None, :])   # [len(a), len(b)]
    nearest_b_for_a = d_ab.min(axis=1)
    nearest_a_for_b = d_ab.min(axis=0)

    score = (nearest_b_for_a.mean() + nearest_a_for_b.mean()) / 2.0

    matched_pairs = []
    for i, xa_i in enumerate(xa):
        j = np.argmin(np.abs(xb - xa_i))
        offset = abs(xa_i - xb[j])
        midpoint = (xa_i + xb[j]) / 2.0
        matched_pairs.append((xa_i, xb[j], midpoint, offset))

    return score, matched_pairs


def extrema_colocation_test(G, R, x_axis, valid_segments, modified_z_thresh=3.5,
                             n_surrogates=500, rng=None):
    """Returns dict with observed score, p-value, null distribution
    summary, and the observed matched pairs (for population pooling)."""
    idx_G = detect_extrema_curvature(G, valid_segments, modified_z_thresh)
    idx_R = detect_extrema_curvature(R, valid_segments, modified_z_thresh)

    observed_score, matched_pairs = symmetric_nn_offset_nm(idx_G, idx_R, x_axis)
    if np.isnan(observed_score):
        return None

    rng = rng or np.random.default_rng()
    n = len(R)
    shifts = rng.integers(1, n, size=n_surrogates)

    null_scores = np.empty(n_surrogates)
    for k, s in enumerate(shifts):
        R_shift = np.roll(R, s)
        idx_R_shift = detect_extrema_curvature(R_shift, valid_segments, modified_z_thresh)
        score_k, _ = symmetric_nn_offset_nm(idx_G, idx_R_shift, x_axis)
        null_scores[k] = score_k if not np.isnan(score_k) else np.inf

    # One-sided: is the REAL alignment at least as tight (small) as
    # this fraction of surrogate alignments?
    p_value = float(np.mean(null_scores <= observed_score))

    return {
        "idx_G": idx_G, "idx_R": idx_R,
        "observed_score": observed_score,
        "null_scores": null_scores,
        "p_value": p_value,
        "matched_pairs": matched_pairs,   # (x_G, x_R, midpoint, offset)
    }


VALID_SEGMENTS = get_valid_segments(window_centers_nm, GRAY_REGIONS)
print(f"Valid (non-gray-region) segments: {len(VALID_SEGMENTS)}, "
      f"covering {sum(len(s) for s in VALID_SEGMENTS)} / {len(window_centers_nm)} "
      f"spectral positions.")

# ----------------------------------------------------------------------
# 4. Per plot, per date
# ----------------------------------------------------------------------

cmap = cm.get_cmap("viridis", len(date_tags_all))

plot_level_results = {}       # plot -> {"combined_p", "per_date": {...}}
all_significant_matches = []  # pooled (plot, date, midpoint_nm, offset_nm)

for pi, plot_name in enumerate(all_plots):
    dates_available = sorted(set(deltaG_plot_date_curve[plot_name].keys())
                              & set(rcf_plot_date_curve[plot_name].keys()))
    if len(dates_available) == 0:
        continue

    per_date_results = {}
    p_values_this_plot = []

    for date_tag in dates_available:
        G = deltaG_plot_date_curve[plot_name][date_tag]
        R = rcf_plot_date_curve[plot_name][date_tag]

        result = extrema_colocation_test(
            G, R, window_centers_nm, VALID_SEGMENTS,
            modified_z_thresh=MODIFIED_Z_THRESH,
            n_surrogates=N_SURROGATES, rng=rng_gen)

        if result is None:
            continue

        per_date_results[date_tag] = result
        p_values_this_plot.append(result["p_value"])

        if result["p_value"] < 0.05:
            for (x_G, x_R, midpoint, offset) in result["matched_pairs"]:
                all_significant_matches.append(
                    (plot_name, date_tag, midpoint, offset))

    if len(p_values_this_plot) == 0:
        continue

    # Fisher's method to combine this plot's per-date p-values
    clipped = np.clip(p_values_this_plot, 1e-300, 1.0)  # guard against p=0
    _, combined_p = combine_pvalues(clipped, method="fisher")

    plot_level_results[plot_name] = {
        "combined_p": combined_p,
        "per_date": per_date_results,
        "n_dates": len(p_values_this_plot),
    }

    # --- Figure: two subplots sharing x-axis, ΔG (top) and RCF (bottom),
    # each showing every date's OWN curve (not an across-date mean, which
    # can blur/cancel out real signal) in the same viridis date-color code. ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for date_tag, result in per_date_results.items():
        G = deltaG_plot_date_curve[plot_name][date_tag]
        R = rcf_plot_date_curve[plot_name][date_tag]

        color_idx = date_tags_all.index(date_tag)
        col = cmap(color_idx / max(len(date_tags_all) - 1, 1))
        sig = result["p_value"] < 0.05
        marker = "o" if sig else "x"

        ax1.plot(window_centers_nm, G, color=col, linewidth=1.3, alpha=0.85,
                 label=date_tag, zorder=2)
        ax2.plot(window_centers_nm, R, color=col, linewidth=1.3, alpha=0.85,
                 zorder=2)

        ax1.scatter(window_centers_nm[result["idx_G"]], G[result["idx_G"]],
                    color=col, marker=marker, s=40, zorder=4,
                    edgecolors="black" if sig else "none", linewidths=0.6)
        ax2.scatter(window_centers_nm[result["idx_R"]], R[result["idx_R"]],
                    color=col, marker=marker, s=40, zorder=4,
                    edgecolors="black" if sig else "none", linewidths=0.6)

    for lo, hi in GRAY_REGIONS:
        ax1.axvspan(lo, hi, facecolor="gray", alpha=1.0, zorder=3,
                    label="excluded (sensor noise / atm. absorption)"
                    if (lo, hi) == GRAY_REGIONS[0] else None)
        ax2.axvspan(lo, hi, facecolor="gray", alpha=1.0, zorder=3)

    ax1.set_ylabel(r"$\Delta G_{13}$")
    ax2.set_ylabel("RCF excitation")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_xlim(380, 2500)
    ax2.set_xticks(np.arange(400, 2501, 100))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    ax1.grid(True, alpha=0.3, zorder=0)
    ax2.grid(True, alpha=0.3, zorder=0)

    attrs = plot_attrs.get(plot_name, {})
    veg_class = attrs.get("veg_class", "n/a")
    n_sig = sum(1 for r in per_date_results.values() if r["p_value"] < 0.05)
    title = (f"{plot_name} — {veg_class}\n"
             f"Extrema co-location: combined p = {combined_p:.3g} "
             f"({n_sig}/{len(per_date_results)} dates individually significant)\n"
             f"markers: filled circle = significant date (p<0.05), x = not significant")
    ax1.set_title(title, fontsize=8)

    ax1.legend(loc="upper left", fontsize=7, ncol=4)

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, f"extrema_colocation_{plot_name}.png")
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)

    if (pi + 1) % 10 == 0 or (pi + 1) == len(all_plots):
        print(f"  [{pi + 1}/{len(all_plots)}] wrote {out_fig}")

print(f"\nDone. {len(plot_level_results)} plot-level figures written to {OUT_DIR}")

# ----------------------------------------------------------------------
# 5. Plot-level summary table
# ----------------------------------------------------------------------

summary_rows = []
for plot_name, res in plot_level_results.items():
    summary_rows.append({
        "plot_name": plot_name,
        "combined_p_fisher": res["combined_p"],
        "n_dates_tested": res["n_dates"],
        "n_dates_significant": sum(1 for r in res["per_date"].values() if r["p_value"] < 0.05),
    })
summary_df = pd.DataFrame(summary_rows).sort_values("combined_p_fisher")
summary_csv = os.path.join(OUT_DIR, "plot_level_extrema_colocation_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Wrote {summary_csv}")

n_plots_sig = (summary_df["combined_p_fisher"] < 0.05).sum()
print(f"{n_plots_sig} / {len(summary_df)} plots show significant "
      f"extrema co-location (combined p < 0.05, uncorrected).")

# ----------------------------------------------------------------------
# 6. Population-level: WHERE do significant matches concentrate?
# ----------------------------------------------------------------------
# Pools every matched extrema-pair midpoint wavelength from every
# individually-significant (plot, date) test, and shows their density
# across the spectrum -- this directly answers "which locations activate
# both series," aggregated across the whole plot population.

if all_significant_matches:
    matches_df = pd.DataFrame(all_significant_matches,
                               columns=["plot_name", "date", "midpoint_nm", "offset_nm"])
    matches_csv = os.path.join(OUT_DIR, "significant_matched_extrema.csv")
    matches_df.to_csv(matches_csv, index=False)
    print(f"Wrote {matches_csv}  ({len(matches_df)} matched extrema pairs "
          f"from significant plot-date tests)")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(matches_df["midpoint_nm"], bins=100, range=(380, 2510),
            color="darkred", alpha=0.7, zorder=2)
    for lo, hi in GRAY_REGIONS:
        ax.axvspan(lo, hi, facecolor="gray", alpha=1.0, zorder=1,
                   label="excluded (sensor noise / atm. absorption)"
                   if (lo, hi) == GRAY_REGIONS[0] else None)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Count of significant matched extrema pairs")
    ax.set_xlim(380, 2510)
    ax.set_xticks(np.arange(400, 2501, 100))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Where Delta_G and RCF-excitation extrema co-locate "
                 "(pooled across significant plot-date tests)")
    plt.tight_layout()
    hist_fig = os.path.join(OUT_DIR, "population_extrema_colocation_density.png")
    plt.savefig(hist_fig, dpi=200)
    plt.close(fig)
    print(f"Wrote {hist_fig}")
else:
    print("\nNo individually-significant (plot, date) extrema co-location "
          "tests found -- no population-level concentration to plot.")
