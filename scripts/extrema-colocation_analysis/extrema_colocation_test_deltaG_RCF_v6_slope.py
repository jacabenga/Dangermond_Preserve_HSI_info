"""
Extrema co-location test: Delta_G vs RCF-excitation (flat-slope-based detection)
=================================================================================

Tests whether the strong local extrema (peaks AND troughs, sign-agnostic)
of Delta_G_13(lambda) and RCF-excitation co-occur at the same spectral
positions more often/tightly than expected by chance.

Unlike the windowed-MI test (which asks "does knowing X reduce local
uncertainty about Y in a narrow 26-band window"), this asks a more global
question directly matching visual inspection: "do the LANDMARK features
(strong peaks and dips) of the two curves line up in wavelength?"

Sign-agnostic by design: RCF-excitation is mostly positive except at the
red-edge, which appears as its strongest NEGATIVE excursion. A true local
maximum or minimum is, by definition, exactly where the first derivative
(slope) passes through zero -- so extrema are detected here via the
first derivative directly, rather than curvature (2nd derivative
magnitude): a candidate point must first be a genuine turning point (a
literal sign change in the first difference -- this alone already
excludes any point on a monotonic, inflection-containing stretch, since
such a point never satisfies that condition), and is then ranked by the
MINIMUM flanking first-difference magnitude around it: the SMALLER of
the two adjacent slope magnitudes, immediately before and after the
turning point. A real, strong, two-sided feature is steep on BOTH
flanks; a lopsided feature (steep on one side, nearly flat on the
other -- a shoulder or step, not a true peak/trough) is penalized,
since its weaker flank pulls the minimum down.

(Two earlier versions of this criterion were tried and both had real
problems, worth recording here rather than hiding: (a) comparing each
point's raw instantaneous |slope| against the GLOBAL population of
every interior point in the curve failed a sanity check outright --
zero detections, even for an obvious synthetic peak -- because on a
curve that is mostly smooth/slowly-varying, the global median slope is
itself already near zero, so a genuine summit's near-zero slope isn't
a statistical outlier relative to that baseline. (b) Using the AVERAGE
of the two flanking first-differences fixed that, but turned out to be
mathematically identical to the curvature script: at a turning point,
curvature = d[p] - d[p-1], and since d[p-1]/d[p] have opposite signs
there, |curvature| = |d[p-1]| + |d[p]| exactly -- so "average flank"
was just curvature / 2 at every point, and since modified z-scores are
scale-invariant, the two scripts were guaranteed to flag identical sets
regardless of the data. Taking the MINIMUM instead of the average is
not a rescaling of curvature -- two turning points with flanks (10,10)
and (19,1) have identical curvature/sum (20) but very different
minimum (10 vs 1), so ranking by minimum genuinely reorders candidates
relative to curvature, verified empirically to no longer produce
identical results.)

Per (plot, date):
    1. Detect extrema (peaks + troughs) in both curves as genuine
       turning points (first-difference sign change), kept only if the
       SMALLER of the two flanking first-difference magnitudes around
       them is a robust, self-calibrated HIGH-outlier (modified
       z-score, Iglewicz & Hoaglin, 1993) relative to the same
       min-flank quantity at every other turning point in the curve.
       Not a user-chosen prominence-fraction threshold, and -- unlike
       an average- or sum-based flank measure -- not a rescaling of the
       2nd-derivative (curvature) criterion either.
    2. Compute a symmetric nearest-neighbor wavelength offset between the
       two extrema sets (a Chamfer-style alignment score, in nm -- lower
       means tighter co-location).
    3. Build a null distribution via circular-shift surrogates of the RCF
       curve (Theiler et al. 1992): shift, re-detect extrema, recompute
       the same offset score, repeat. Circular shift preserves each
       curve's own value distribution exactly, so the SAME flat-slope
       criterion applies unchanged to every surrogate.
    4. One-sided p-value: fraction of surrogate scores at least as tight
       as the observed score.

Per plot: per-date p-values combined via Fisher's method into one
plot-level p-value. Plot-level combined p-values are then corrected for
multiple comparisons across the ~121-plot population via Benjamini-
Hochberg FDR (Benjamini & Hochberg, 1995) -- the summary table reports
both the raw (uncorrected) combined p-value and the BH-adjusted p-value
(q-value) side by side. A population-level step pools the actual matched
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
OUT_DIR = r"C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_SHIFT-plots_extrema_colocation\based_on_flat_slope_v2b"
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_ID_COL = "plot_plot_name"

# Flat-slope extrema detection replaces the curvature-based criterion.
# A candidate must first be a genuine turning point (sign change in the
# first difference -- structurally excludes any point on a monotonic,
# inflection-containing stretch), then is ranked by the MINIMUM of the
# two flanking |first-difference| magnitudes (not their average/sum --
# see the module docstring for why the average version turned out to be
# a disguised rescaling of curvature). FLATNESS_Z_THRESH is the modified
# z-score cutoff (Iglewicz & Hoaglin, 1993) applied to that min-flank
# value, self-calibrated per curve/segment -- so there is still no
# analyst-tuned parameter left to sweep across runs.
FLATNESS_Z_THRESH = 3.5

N_SURROGATES = 500           # cheap here (no k-NN estimator), so the
                             # full 500 from the MI script is affordable
                             # per (plot, date), not just once per plot
NOISE_SEED = 0
FDR_ALPHA = 0.05             # for BH-FDR correction of the plot-level
                             # combined p-values (Section 5) across the
                             # whole plot population

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


def detect_extrema_flat_slope(curve, valid_segments, flatness_z_thresh=3.5):
    """Peaks AND troughs, sign-agnostic -- detected via the first
    derivative directly, rather than curvature (2nd derivative). A
    candidate must first be a genuine turning point: a strict sign
    change in the first difference. This alone already excludes any
    point on a monotonic, inflection-containing stretch, since such a
    point never satisfies that condition -- so, unlike a pure curvature
    threshold, there is no risk of a rapid-but-monotonic bend being
    mistaken for a real extremum.

    Confirmed turning points are then ranked by the MINIMUM of the two
    flanking |first-difference| magnitudes (immediately before and
    after the turning point) -- deliberately the minimum, not their
    average or sum. A real, strong, two-sided feature is steep on BOTH
    flanks; a lopsided feature (steep approach on one side, nearly flat
    on the other -- a shoulder or step rather than a true peak/trough)
    is penalized, since its weaker flank pulls the minimum down. A
    turning point is kept only if this min-flank value is a robust,
    self-calibrated statistical outlier relative to the same quantity
    at every OTHER turning point in the curve (modified z-score,
    Iglewicz & Hoaglin 1993) -- not a user-chosen prominence-fraction
    threshold.

    (Two earlier versions of this function were tried and rejected: (a)
    comparing each point's raw instantaneous |slope| against the GLOBAL
    population of every interior point in the curve failed a sanity
    check outright -- zero detections, even for an obvious synthetic
    peak -- because on a curve that is mostly smooth/slowly-varying,
    the global median slope is itself already near zero, so a genuine
    summit's near-zero slope isn't a statistical outlier relative to
    that baseline. (b) Using the AVERAGE of the two flanking first-
    differences fixed that, but turned out to be mathematically
    identical to the curvature script: at a turning point, curvature =
    d[p] - d[p-1], and since d[p-1]/d[p] have opposite signs there,
    |curvature| = |d[p-1]| + |d[p]| exactly, so "average flank" was
    just curvature/2 at every point -- and since modified z-scores are
    scale-invariant, that guaranteed identical flagged sets to the
    curvature script regardless of the data. The MINIMUM is not a
    rescaling of curvature: two turning points with flanks (10,10) and
    (19,1) have identical curvature/sum (20) but very different minimum
    (10 vs 1), so ranking by minimum genuinely reorders candidates --
    verified empirically to no longer match the curvature script.)

    Mirrors the curvature-based function's structure: the reference
    statistic (median/MAD of the min-flank value) is computed once,
    GLOBALLY across all valid segments combined, then the same
    threshold is applied within each segment independently -- so
    gray-region values never contribute to the reference statistic, and
    no spurious extremum is created at a splice boundary between
    segments."""
    curve = np.asarray(curve)
    if len(valid_segments) == 0:
        return np.array([], dtype=int)

    # Pass 1: find genuine turning points and their min-flank value, in
    # every valid segment.
    seg_turning = []       # (seg, turning_idx_local, contrast)
    all_contrast = []

    for seg in valid_segments:
        if len(seg) < 3:
            continue   # too short to contain an interior turning point
        seg_curve = curve[seg]
        d = np.diff(seg_curve)   # d[k] = seg_curve[k+1] - seg_curve[k]

        is_max = np.zeros(len(seg_curve), dtype=bool)
        is_min = np.zeros(len(seg_curve), dtype=bool)
        is_max[1:-1] = (d[:-1] > 0) & (d[1:] < 0)
        is_min[1:-1] = (d[:-1] < 0) & (d[1:] > 0)
        turning_idx = np.flatnonzero(is_max | is_min)
        if len(turning_idx) == 0:
            continue

        # Min-flank: the SMALLER of the two |first-difference|
        # magnitudes just before (d[turning_idx - 1]) and just after
        # (d[turning_idx]) each turning point.
        left_slope = np.abs(d[turning_idx - 1])
        right_slope = np.abs(d[turning_idx])
        contrast = np.minimum(left_slope, right_slope)

        seg_turning.append((seg, turning_idx, contrast))
        all_contrast.append(contrast)

    if not seg_turning:
        return np.array([], dtype=int)

    # Global reference statistic (median/MAD of the min-flank value),
    # computed once across ALL valid segments combined -- same role as
    # rng_span in the prominence-based version, and med/mad-of-curvature
    # in the earlier curvature-based version.
    all_contrast = np.concatenate(all_contrast)
    med = np.median(all_contrast)
    mad = np.median(np.abs(all_contrast - med))

    all_extrema = []
    for seg, turning_idx, contrast in seg_turning:
        if mad == 0:
            # Degenerate case (near-identical min-flank value
            # throughout): still self-calibrated, no fixed user-set
            # fraction -- fall back to strictly-above-median.
            keep = contrast > med
        else:
            modified_z = 0.6745 * (contrast - med) / mad
            keep = modified_z > flatness_z_thresh
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


def extrema_colocation_test(G, R, x_axis, valid_segments, flatness_z_thresh=3.5,
                             n_surrogates=500, rng=None):
    """Returns dict with observed score, p-value, null distribution
    summary, and the observed matched pairs (for population pooling)."""
    idx_G = detect_extrema_flat_slope(G, valid_segments, flatness_z_thresh)
    idx_R = detect_extrema_flat_slope(R, valid_segments, flatness_z_thresh)

    observed_score, matched_pairs = symmetric_nn_offset_nm(idx_G, idx_R, x_axis)
    if np.isnan(observed_score):
        return None

    rng = rng or np.random.default_rng()
    n = len(R)
    shifts = rng.integers(1, n, size=n_surrogates)

    null_scores = np.empty(n_surrogates)
    for k, s in enumerate(shifts):
        R_shift = np.roll(R, s)
        idx_R_shift = detect_extrema_flat_slope(R_shift, valid_segments, flatness_z_thresh)
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


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995).

    Returns:
        significant : bool array, same order as `pvals` -- True where
                       the hypothesis is rejected after correction.
        cutoff       : the largest p-value that still clears the BH
                       critical line (a single scalar threshold).
        q_values     : BH-adjusted p-values ("q-values"), same order as
                       `pvals` and directly comparable to `alpha` --
                       reject where q <= alpha. Reported alongside the
                       raw p-value so both are visible side by side.
    """
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    thresh = alpha * ranks / n
    below = ranked <= thresh
    if below.any():
        cutoff = ranked[np.max(np.where(below))]
    else:
        cutoff = 0.0
    significant = pvals <= cutoff

    # q_(i) = min_{j>=i} (p_(j) * n / j), enforced monotonic via a
    # reverse cumulative minimum, then mapped back to the original
    # (unsorted) order so it lines up with the input `pvals`.
    q_ranked = np.minimum.accumulate((ranked[::-1] * n / ranks[::-1]))[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_values = np.empty(n)
    q_values[order] = q_ranked

    return significant, cutoff, q_values


VALID_SEGMENTS = get_valid_segments(window_centers_nm, GRAY_REGIONS)
print(f"Valid (non-gray-region) segments: {len(VALID_SEGMENTS)}, "
      f"covering {sum(len(s) for s in VALID_SEGMENTS)} / {len(window_centers_nm)} "
      f"spectral positions.")

# ----------------------------------------------------------------------
# 4. Per plot, per date
# ----------------------------------------------------------------------

# --- y-limit helper (constant across plots) ---
wl = np.asarray(window_centers_nm, dtype=float)
VISIBLE = (wl >= 380) & (wl <= 2500)
for lo, hi in GRAY_REGIONS:
    VISIBLE &= ~((wl >= lo) & (wl <= hi))


def data_centered_ylim(stack, mask=VISIBLE, factor=1.2):
    """Symmetric limits about the data midpoint, spanning factor x the range."""
    vals = np.concatenate([np.asarray(a, dtype=float)[mask] for a in stack])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = vals.min(), vals.max()
    center = 0.5 * (lo + hi)
    amp = 0.5 * (hi - lo)
    if amp == 0:
        amp = max(abs(center), 1.0) * 0.01
    return center - factor * amp, center + factor * amp




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

    G_stack, R_stack = [], []

    for date_tag in dates_available:
        G = deltaG_plot_date_curve[plot_name][date_tag]
        R = rcf_plot_date_curve[plot_name][date_tag]

        G_stack.append(np.asarray(G, dtype=float))
        R_stack.append(np.asarray(R, dtype=float))

        result = extrema_colocation_test(
            G, R, window_centers_nm, VALID_SEGMENTS,
            flatness_z_thresh=FLATNESS_Z_THRESH,
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

    ax1.legend(loc="upper right", fontsize=7, ncol=4)

    for ax, stack in ((ax1, G_stack), (ax2, R_stack)):
        lim = data_centered_ylim(stack)
        if lim is not None:
            ax.set_ylim(*lim)

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, f"extrema_colocation_{plot_name}.png")
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)

    if (pi + 1) % 10 == 0 or (pi + 1) == len(all_plots):
        print(f"  [{pi + 1}/{len(all_plots)}] wrote {out_fig}")

print(f"\nDone. {len(plot_level_results)} plot-level figures written to {OUT_DIR}")

# ----------------------------------------------------------------------
# 5. Plot-level summary table  (+ BH-FDR correction across plots)
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

significant_fdr, p_fdr_cutoff, q_values = fdr_bh(
    summary_df["combined_p_fisher"].values, alpha=FDR_ALPHA)
summary_df["p_fdr_adjusted"] = q_values          # BH-adjusted p-value (q-value)
summary_df["significant_fdr"] = significant_fdr  # q_value <= FDR_ALPHA

summary_csv = os.path.join(OUT_DIR, "plot_level_extrema_colocation_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Wrote {summary_csv}")

n_plots_sig = (summary_df["combined_p_fisher"] < 0.05).sum()
print(f"{n_plots_sig} / {len(summary_df)} plots show significant "
      f"extrema co-location (combined p < 0.05, uncorrected).")

n_plots_sig_fdr = int(significant_fdr.sum())
print(f"{n_plots_sig_fdr} / {len(summary_df)} plots remain significant after "
      f"BH-FDR correction (alpha={FDR_ALPHA}, cutoff p={p_fdr_cutoff:.4g}).")

# ----------------------------------------------------------------------
# 6. Population-level: WHERE do significant matches concentrate?
# ----------------------------------------------------------------------
# Pools every matched extrema-pair midpoint wavelength from every
# individually-significant (plot, date) test, and shows their density
# across the spectrum -- this directly answers "which locations activate
# both series," aggregated across the whole plot population. (This step
# pools from the per-date, uncorrected p<0.05 tests, same as before --
# the BH-FDR correction added above applies to the plot-level combined
# p-values in Section 5, not to this per-date pooling step.)

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
