"""
Extrema co-location test: Delta_G_13 vs RCF-excitation -- PFT version
=========================================================================

Adapts the per-plot extrema co-location test (circular-shift surrogate
null, Theiler et al. 1992) to the per-PFT / per-veg-class Delta_G_13 and
RCF-excitation vectors used for Figure 4 (Dangermond Info Paper: Shrub /
Herbaceous / Woodland, with their constituent veg classes).

Tests whether the strong local extrema (peaks AND troughs, sign-agnostic)
of Delta_G_13(lambda) and RCF-excitation co-occur at the same spectral
positions more often/tightly than expected by chance.

Per (veg_class, date):
    1. Detect extrema (peaks + troughs) in both curves via
       scipy.signal.find_peaks with a relative-prominence threshold.
    2. Compute a symmetric nearest-neighbor wavelength offset between the
       two extrema sets (a Chamfer-style alignment score, in nm -- lower
       means tighter co-location).
    3. Build a null distribution via circular-shift surrogates of the RCF
       curve: shift, re-detect extrema, recompute the same offset score,
       repeat.
    4. One-sided p-value: fraction of surrogate scores at least as tight
       as the observed score.

Per veg_class: per-date p-values combined via Fisher's method into one
class-level p-value. UNLIKE the original per-plot template (121 plots,
where multiplicity was not FDR-corrected at the population level), here
the population is only 8 veg classes, so a Benjamini-Hochberg FDR
correction is applied across the 8 class-level combined p-values --
flagged separately (`significant_fdr`) alongside the raw combined
p-value in the summary table. This is an addition beyond the pasted
template, not something the template itself did.

A population-level step pools the actual matched extrema-pair midpoint
wavelengths from every individually-significant (class, date) test
(p < 0.05, uncorrected -- matching the original template), to show WHERE
in the spectrum tight co-location concentrates across the veg classes.

date:    28.07.2026
author:  Jaime C. Revenga

Requirements:
    pip install numpy scipy matplotlib pandas
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import find_peaks
from scipy.stats import combine_pvalues

# ============================================================
# 1. USER PARAMETERS  <- edit these paths to match your machine
# ============================================================

base_dir = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_4\data_in'

OUT_DIR = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_4\figures_out\extrema_colocation\prominence_fraction_0.3'
os.makedirs(OUT_DIR, exist_ok=True)

date_list = [
    "2022-02-24", "2022-02-28",
    "2022-03-08", "2022-03-16", "2022-03-22",
    "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
    "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29"
]

PFTS = [
    {"folder": "shrub",      "veg_classes": [5, 11, 20, 31, 37], "label": "Shrub"},
    {"folder": "herbaceous", "veg_classes": [33, 43],            "label": "Herbaceous"},
    {"folder": "woodland",   "veg_classes": [16],                "label": "Woodland / Tree"},
]

class_names = {
    5:  "sh.2", 11: "sh.1", 16: "oak", 20: "sh.3",
    31: "sh.4", 33: "he.1", 37: "sh.5", 43: "he.2",
}

# Gray background bands (water-vapour absorption + sensor limits)
GRAY_REGIONS = [
    (380,  460),
    (1315, 1460),
    (1765, 2010),
    (2440, 2510),
]

# ============================================================
# 2. FIXED PARAMETERS  (do not change)
# ============================================================

PROMINENCE_FRACTION = 0.3   # extremum must rise/fall by at least this
                             # fraction of the curve's own peak-to-peak
                             # range to count as a "strong deviation"
N_SURROGATES = 500           # cheap here (no k-NN estimator), so the
                             # full 500 is affordable per (class, date)
NOISE_SEED = 0
FDR_ALPHA = 0.05             # for the class-level FDR correction (see
                             # module docstring -- addition beyond the
                             # original per-plot template)
KLPD_WINDOW = 13             # window used by the Delta_G_13 operator
                             # itself, needed to reconstruct the 413-pt
                             # wavelength axis the vectors live on

rng_gen = np.random.default_rng(NOISE_SEED)

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


def detect_extrema(curve, valid_segments, prominence_fraction=0.15):
    """Peaks AND troughs, sign-agnostic, with prominence relative to the
    curve's own peak-to-peak range -- computed from VALID (non-excluded)
    values only, and detected independently within each contiguous valid
    segment, so gray-region values never contribute to the range, the
    prominence threshold, or a spurious peak at a splice boundary."""
    curve = np.asarray(curve)
    if len(valid_segments) == 0:
        return np.array([], dtype=int)

    valid_vals = np.concatenate([curve[seg] for seg in valid_segments])
    rng_span = valid_vals.max() - valid_vals.min()
    if rng_span <= 0:
        return np.array([], dtype=int)
    prom = prominence_fraction * rng_span

    all_extrema = []
    for seg in valid_segments:
        if len(seg) < 3:
            continue
        seg_curve = curve[seg]
        peak_idx, _ = find_peaks(seg_curve, prominence=prom)
        trough_idx, _ = find_peaks(-seg_curve, prominence=prom)
        combined = np.concatenate([peak_idx, trough_idx])
        all_extrema.extend(seg[combined])

    return np.sort(np.array(all_extrema, dtype=int))


def symmetric_nn_offset_nm(idx_a, idx_b, x_axis):
    """Chamfer-style symmetric mean nearest-neighbor distance (nm)
    between two sets of extrema indices. Lower = tighter co-location."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return np.nan, []
    xa = x_axis[idx_a]
    xb = x_axis[idx_b]

    d_ab = np.abs(xa[:, None] - xb[None, :])
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


def extrema_colocation_test(G, R, x_axis, valid_segments, prominence_fraction=0.15,
                             n_surrogates=500, rng=None):
    """Returns dict with observed score, p-value, null distribution
    summary, and the observed matched pairs (for population pooling)."""
    idx_G = detect_extrema(G, valid_segments, prominence_fraction)
    idx_R = detect_extrema(R, valid_segments, prominence_fraction)

    observed_score, matched_pairs = symmetric_nn_offset_nm(idx_G, idx_R, x_axis)
    if np.isnan(observed_score):
        return None

    rng = rng or np.random.default_rng()
    n = len(R)
    shifts = rng.integers(1, n, size=n_surrogates)

    null_scores = np.empty(n_surrogates)
    for k, s in enumerate(shifts):
        R_shift = np.roll(R, s)
        idx_R_shift = detect_extrema(R_shift, valid_segments, prominence_fraction)
        score_k, _ = symmetric_nn_offset_nm(idx_G, idx_R_shift, x_axis)
        null_scores[k] = score_k if not np.isnan(score_k) else np.inf

    p_value = float(np.mean(null_scores <= observed_score))

    return {
        "idx_G": idx_G, "idx_R": idx_R,
        "observed_score": observed_score,
        "null_scores": null_scores,
        "p_value": p_value,
        "matched_pairs": matched_pairs,
    }


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Addition beyond the original
    per-plot template (see module docstring)."""
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


# ============================================================
# 4. LOAD RAW PER-CLASS, PER-DATE VECTORS
# ============================================================

print("=" * 60)
print("Loading Delta_G_13 and RCF mean-excitation vectors ...")
print("=" * 60)

raw_data = {}
class_to_pft = {}
n_bands = None

for pft in PFTS:
    pft_label   = pft["label"]
    veg_classes = pft["veg_classes"]
    dir_kernels = os.path.join(base_dir, pft["folder"], "stored_kernel_excitation_vectors")
    dir_klpd    = os.path.join(base_dir, pft["folder"], "stored_delta_G_13_vectors")

    klpd_files   = sorted([f for f in os.listdir(dir_klpd)    if f.endswith(".pkl")])
    kernel_files = sorted([f for f in os.listdir(dir_kernels) if f.endswith(".pkl")])

    print(f"\n  PFT: {pft_label}")

    for cls in veg_classes:
        raw_data[cls] = {}
        class_to_pft[cls] = pft_label

    for d, date_tag in enumerate(date_list):
        all_delta_G = load_pkl(os.path.join(dir_klpd,    klpd_files[d]))
        data_k      = load_pkl(os.path.join(dir_kernels, kernel_files[d]))
        mean_per_class = data_k["mean_per_class"]

        if n_bands is None:
            n_bands = len(list(all_delta_G.values())[0])

        for cls in veg_classes:
            X = np.array(all_delta_G[cls])
            Y = mean_per_class[cls].mean(axis=0)
            assert len(X) == n_bands and len(Y) == n_bands, \
                f"Vector length mismatch for class {cls}, date {date_tag}"
            raw_data[cls][date_tag] = {"X": X, "Y": Y}

    print(f"    loaded {len(date_list)} dates for classes {veg_classes}")

all_classes = sorted(raw_data.keys())
print(f"\n{len(all_classes)} veg classes loaded across {len(PFTS)} PFTs.\n")

# Reconstruct the wavelength axis the 413-pt Delta_G_13 / RCF vectors
# live on: the KLPD operator (window=13) applied to the raw 425-band
# AVIRIS-NG grid produces exactly n_bands = 425 - 13 + 1 positions.
raw_n_bands = n_bands + KLPD_WINDOW - 1
window_centers_nm = np.array([
    377.8 + 5.01 * ((i + (i + KLPD_WINDOW - 1)) / 2)
    for i in range(raw_n_bands - KLPD_WINDOW + 1)
])
assert len(window_centers_nm) == n_bands, \
    "Reconstructed wavelength axis does not match vector length."

VALID_SEGMENTS = get_valid_segments(window_centers_nm, GRAY_REGIONS)
print(f"Valid (non-gray-region) segments: {len(VALID_SEGMENTS)}, "
      f"covering {sum(len(s) for s in VALID_SEGMENTS)} / {len(window_centers_nm)} "
      f"spectral positions.")

# ============================================================
# 5. PER VEG CLASS, PER DATE
# ============================================================

_t0 = time.time()

cmap = cm.get_cmap("viridis", len(date_list))

class_level_results = {}      # veg_class -> {"combined_p", "per_date": {...}}
all_significant_matches = []  # pooled (veg_class, date, midpoint_nm, offset_nm)

for ci, cls in enumerate(all_classes):
    dates_available = sorted(raw_data[cls].keys())

    per_date_results = {}
    p_values_this_class = []

    for date_tag in dates_available:
        G = raw_data[cls][date_tag]["X"]
        R = raw_data[cls][date_tag]["Y"]

        result = extrema_colocation_test(
            G, R, window_centers_nm, VALID_SEGMENTS,
            prominence_fraction=PROMINENCE_FRACTION,
            n_surrogates=N_SURROGATES, rng=rng_gen)

        if result is None:
            continue

        per_date_results[date_tag] = result
        p_values_this_class.append(result["p_value"])

        if result["p_value"] < 0.05:
            for (x_G, x_R, midpoint, offset) in result["matched_pairs"]:
                all_significant_matches.append(
                    (cls, date_tag, midpoint, offset))

    if len(p_values_this_class) == 0:
        continue

    clipped = np.clip(p_values_this_class, 1e-300, 1.0)
    _, combined_p = combine_pvalues(clipped, method="fisher")

    class_level_results[cls] = {
        "combined_p": combined_p,
        "per_date": per_date_results,
        "n_dates": len(p_values_this_class),
    }

    # --- Figure: two subplots sharing x-axis, ΔG (top) and RCF (bottom),
    # each showing every date's OWN curve in viridis date-color code. ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for date_tag, result in per_date_results.items():
        G = raw_data[cls][date_tag]["X"]
        R = raw_data[cls][date_tag]["Y"]

        color_idx = date_list.index(date_tag)
        col = cmap(color_idx / max(len(date_list) - 1, 1))
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

    cls_name = class_names.get(cls, f"class {cls}")
    n_sig = sum(1 for r in per_date_results.values() if r["p_value"] < 0.05)
    title = (f"{class_to_pft[cls]} — {cls_name} (veg class {cls})\n"
             f"Extrema co-location: combined p = {combined_p:.3g} "
             f"({n_sig}/{len(per_date_results)} dates individually significant)\n"
             f"markers: filled circle = significant date (p<0.05), x = not significant")
    ax1.set_title(title, fontsize=8)

    ax1.legend(loc="upper left", fontsize=7, ncol=4)

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, f"extrema_colocation_{cls_name}.png")
    plt.savefig(out_fig, dpi=200)
    plt.close(fig)

    print(f"  [{ci + 1}/{len(all_classes)}] wrote {out_fig}")

_dt = time.time() - _t0
print(f"\nDone. {len(class_level_results)} class-level figures written to {OUT_DIR} "
      f"({_dt/60:.1f} min).")

# ============================================================
# 6. CLASS-LEVEL SUMMARY TABLE  (+ FDR correction across classes)
# ============================================================

summary_rows = []
for cls, res in class_level_results.items():
    summary_rows.append({
        "veg_class": cls,
        "class_name": class_names.get(cls, str(cls)),
        "pft": class_to_pft[cls],
        "combined_p_fisher": res["combined_p"],
        "n_dates_tested": res["n_dates"],
        "n_dates_significant": sum(1 for r in res["per_date"].values() if r["p_value"] < 0.05),
    })
summary_df = pd.DataFrame(summary_rows).sort_values("combined_p_fisher").reset_index(drop=True)

significant_fdr, p_fdr_cutoff = fdr_bh(summary_df["combined_p_fisher"].values, alpha=FDR_ALPHA)
summary_df["significant_fdr"] = significant_fdr

summary_csv = os.path.join(OUT_DIR, "class_level_extrema_colocation_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print(f"Wrote {summary_csv}")

n_classes_sig_raw = (summary_df["combined_p_fisher"] < 0.05).sum()
n_classes_sig_fdr = int(significant_fdr.sum())
print(f"{n_classes_sig_raw} / {len(summary_df)} classes show significant "
      f"extrema co-location (combined p < 0.05, uncorrected).")
print(f"{n_classes_sig_fdr} / {len(summary_df)} classes remain significant after "
      f"BH-FDR correction (alpha={FDR_ALPHA}, cutoff p={p_fdr_cutoff:.4g}).")

# ============================================================
# 7. POPULATION-LEVEL: WHERE DO SIGNIFICANT MATCHES CONCENTRATE?
# ============================================================
# Pools every matched extrema-pair midpoint wavelength from every
# individually-significant (veg_class, date) test (p < 0.05,
# uncorrected -- matching the original template), and shows their
# density across the spectrum, aggregated across all veg classes.

if all_significant_matches:
    matches_df = pd.DataFrame(all_significant_matches,
                               columns=["veg_class", "date", "midpoint_nm", "offset_nm"])
    matches_csv = os.path.join(OUT_DIR, "significant_matched_extrema.csv")
    matches_df.to_csv(matches_csv, index=False)
    print(f"Wrote {matches_csv}  ({len(matches_df)} matched extrema pairs "
          f"from significant class-date tests)")

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
    ax.set_title("Where Delta_G_13 and RCF-excitation extrema co-locate "
                 "(pooled across significant veg-class/date tests)")
    plt.tight_layout()
    hist_fig = os.path.join(OUT_DIR, "population_extrema_colocation_density.png")
    plt.savefig(hist_fig, dpi=200)
    plt.close(fig)
    print(f"Wrote {hist_fig}")
else:
    print("\nNo individually-significant (veg_class, date) extrema co-location "
          "tests found -- no population-level concentration to plot.")
