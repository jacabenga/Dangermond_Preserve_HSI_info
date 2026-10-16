
# ============================================================
# Figure_4_publication_ready_clean_code_only-plotting_v2.py
# ============================================================
#
# date:    9.04.2026
# author:  Jaime C. Revenga
#
# PURPOSE:
#   Plotting-only version of Figure 4 (Dangermond Info Paper).
#   Loads pre-computed delta_G and kernel excitation vectors for
#   all 3 PFTs (Shrub, Herbaceous, Woodland/Tree) and plots
#   two combined figures — each with 3 stacked panels (one per PFT):
#
#     Figure 4a:  MI(ΔG, mean kernel excitation)     — 3 panels stacked
#     Figure 4b:  MI(ΔG, variance kernel excitation) — 3 panels stacked
#
# DATA SOURCES (pre-computed, stored as .pkl files):
#
#    1) Delta_G vector, at 122 locations per class (there are 8 veg_classes),
#    2) RCF-excitations vector, at 122 locations per class (there are 8 veg_classes),
#
# PKL structure is:
'''
deltaG_vectors
│
├── class5_loc000
│       │
│       ├── 2022-02-24 : array(413,)
│       ├── 2022-05-29 : array(413,)
│       ├── ...
│       └── 13 dates total
│
├── class5_loc001
│       └── ...
│
└── class43_loc121

HIERARCHY IS:
            location (N=122) → date (N=13) → spectral vector

'''
#
#
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import pickle

from sklearn.feature_selection import mutual_info_regression

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import re
from collections import defaultdict


# ============================================================
# 1. USER PARAMETERS  ← edit these paths to match your machine
# ============================================================

NOISE_PAIRS_PER_CLASS = 100     # set to None to use ALL pairs (slow)
NOISE_SEED            = 42

# Base directory for Figure 4 data
base_dir = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_WRA_MI_122-per-class-vectors\in'

# Output directory for saved figures
dir_fig_out = rf'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\hsi_WRA_MI_122-per-class-vectors\out\MI_deltaG_RCF-excitations_noise{NOISE_PAIRS_PER_CLASS}'
os.makedirs(dir_fig_out, exist_ok=True)

# Dates (chronological order)
date_list = [
    "2022-02-24", "2022-02-28",
    "2022-03-08", "2022-03-16", "2022-03-22",
    "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
    "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29"
]


# ============================================================
# 1.1. Inspect .pkl file structure
# ============================================================


delta_G_pkl_file = os.path.join(base_dir, 'delta_G', r'deltaG_class_date_vectors_S2withinPFT_fixedLocations_adjacentGroups.pkl')
RCF_pkl_file = os.path.join(base_dir, 'RCF_excitations', r'rcf_class_date_vectors_S2withinPFT_fixedLocations_adjacentGroups.pkl')

pkl_files = [delta_G_pkl_file,RCF_pkl_file]

for (i, pkl_file) in enumerate(pkl_files):
    print('\npkl_fle is\n', pkl_files[i])

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    print("Top-level type:")
    print(type(data))

    if isinstance(data, dict):
        print("\nDictionary keys:")
        for k in data.keys():
            print("  ", k)

        print("\nContents:")
        for k, v in data.items():
            print(f"\nKEY: {k}")
            print(" type:", type(v))

            if isinstance(v, (list, tuple, np.ndarray)):
                print(" shape:", np.shape(v))

            elif isinstance(v, dict):
                print(" dict keys:", list(v.keys())[:10])

    else:
        print("\nObject shape:")
        print(np.shape(data))


# ============================================================
# 1.2. Load both files - inspect their structure
# ============================================================


with open(delta_G_pkl_file, "rb") as f:
    deltaG_data = pickle.load(f)

with open(RCF_pkl_file, "rb") as f:
    rcf_data = pickle.load(f)


deltaG_vectors = deltaG_data["deltaG_plot_date_curve"]
rcf_vectors    = rcf_data["rcf_plot_date_curve"]

date_tags = deltaG_data["date_tags"]
window_centers_nm = deltaG_data["window_centers_nm"]

print("Number of locations:", len(deltaG_vectors))
print("Dates:", len(date_tags))

example_key = list(deltaG_vectors.keys())[0]

print(example_key)
print("ΔG shape:", np.shape(deltaG_vectors[example_key]))
print("RCF shape:", np.shape(rcf_vectors[example_key]))


# ============================================================
# 1.3. Let's inspect one actual entry.
# ============================================================

example_key = list(deltaG_vectors.keys())[0]

print("KEY:", example_key)

print("\nΔG object:")
print(type(deltaG_vectors[example_key]))
print(deltaG_vectors[example_key])

print("\nRCF object:")
print(type(rcf_vectors[example_key]))
print(rcf_vectors[example_key])

print('...')

print("\nΔG length:")
try:
    print(len(deltaG_vectors[example_key]))
except Exception as e:
    print(e)

print("\nRCF length:")
try:
    print(len(rcf_vectors[example_key]))
except Exception as e:
    print(e)


# ============================================================
# 1.3. Have a look at missing dates
# ============================================================

missing_dates = {}

for loc_key in deltaG_vectors.keys():

    available = set(deltaG_vectors[loc_key].keys())

    missing = set(date_tags) - available

    if missing:
        missing_dates[loc_key] = sorted(missing)


print("Locations with missing dates:", len(missing_dates))

for k,v in list(missing_dates.items())[:20]:
    print(k, "missing:", v)

# ============================================================
# 1.4. Quantify the exact missing dates
# ============================================================
# intersection of available locations:

common_locations = sorted(
    set(deltaG_vectors.keys()) &
    set(rcf_vectors.keys())
)

print(
    "Common locations:",
    len(common_locations)
)

# quantify exactly what is missing

all_locations = set(deltaG_vectors.keys()) | set(rcf_vectors.keys())

print("ΔG locations:", len(deltaG_vectors))
print("RCF locations:", len(rcf_vectors))
print("Common:", len(common_locations))
missing_locations = all_locations - set(common_locations)
print(
    "Missing locations:",
    len(missing_locations)
)
print(missing_locations)

#
missing_date_count = 0

for loc in common_locations:
    missing = set(date_tags) - (
        set(deltaG_vectors[loc].keys())
        &
        set(rcf_vectors[loc].keys())
    )

    if missing:
        missing_date_count += len(missing)

print(
    "Missing location-date pairs:",
    missing_date_count
)

# ============================================================
# 2. FIXED PARAMETERS  (do not change)
# ============================================================

WINDOW_SIZE = 26      # sliding-window width for MI computation
N_NEIGHBORS = 5       # k-NN parameter for mutual_info_regression

# PFT definitions — folder name, veg class IDs, display label
PFTS = [
    {
        "folder":      "shrub",
        "veg_classes": [5, 11, 20, 31, 37],
        "label":       "Shrub",
        "color_mean":  "orange",
        "color_var":   "green",
    },
    {
        "folder":      "herbaceous",
        "veg_classes": [33, 43],
        "label":       "Herbaceous",
        "color_mean":  "orange",
        "color_var":   "green",
    },
    {
        "folder":      "woodland",
        "veg_classes": [16],
        "label":       "Woodland / Tree",
        "color_mean":  "orange",
        "color_var":   "green",
    },
]
'''
class_names = {
    5:  "Black sage scrub",
    11: "California sagebrush scrub",
    16: "Coast live oak woodland",
    20: "Coyote brush scrub",
    31: "Menzie's golden bush scrub",
    33: "Non-native grassland",
    37: "Purple sage scrub",
    43: "Semi-natural herbaceous stands",
}
'''
# shorter names to facilitate label in Figure
class_names = {
    5:  "sh.2",
    11: "sh.1",
    16: "oak",
    20: "sh.3",
    31: "sh.4",
    33: "he.1",
    37: "sh.5",
    43: "he.2",
}


# Gray background bands (water-vapour absorption + sensor limits)
gray_regions = [
    (0,    460),
    (1315, 1460),
    (1765, 2010),
    (2440, 2510),
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def compute_mutual_information(X, Y, n_neighbors=5):
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    return mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=0)[0]


def sliding_window_mi(X, Y, window_size=26, n_neighbors=5):
    """Compute MI between X and Y using a rolling spectral window."""
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


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def shade_gray_regions(ax):
    for r_start, r_end in gray_regions:
        ax.axvspan(r_start, r_end,
                   facecolor="lightgray",  # ← changed
                   alpha=1.0,              # ← fully opaque
                   zorder=10)


def format_mi_ax(ax, window_centers_nm, ylabel="Mutual Information", ylim=None, x_axis_title = None):
    xticks_major = np.arange(400, 2501, 100)
    xticks_minor = np.arange(400, 2501, 50)

    if x_axis_title is not None:
        ax.set_xlabel(x_axis_title, fontsize=12)  # "Wavelength (nm)", fontsize=12 --> but only if subplot = 2
    else:
        ax.set_xlabel("") # <- truly no label and no space reserved for it
        #ax.xaxis.label.set_visible(False) # even better for tight layouts

    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xticks(xticks_major)
    ax.set_xticks(xticks_minor, minor=True)

    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', linestyle='--', alpha=0.3)

    ax.set_xlim(window_centers_nm[0], window_centers_nm[-1])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)  # ↑ bigger
    ax.tick_params(axis='y', labelsize=9)  # ↑ y-ticks bigger

    shade_gray_regions(ax)

    if ylim is not None:
        ax.set_ylim(ylim)



# ============================================================
# 4. COMPUTE MI OVER ALL VALID LOCATION-DATE PAIRS
# ============================================================

print("=" * 60)
print("Computing MI spectra for all valid locations and dates ...")
print("=" * 60)

results = {}

n_calculated = 0
n_missing_dates = 0

for loc_key in common_locations:

    print(f"Processing {loc_key}")

    results[loc_key] = {}

    available_dates = (
        set(deltaG_vectors[loc_key].keys())
        &
        set(rcf_vectors[loc_key].keys())
        &
        set(date_tags)
    )

    missing_dates_here = set(date_tags) - available_dates

    if missing_dates_here:
        n_missing_dates += len(missing_dates_here)
        print(
            f"  Missing dates: {sorted(missing_dates_here)}"
        )

    for date_tag in sorted(available_dates):

        X = np.asarray(
            deltaG_vectors[loc_key][date_tag],
            dtype=float
        )

        Y = np.asarray(
            rcf_vectors[loc_key][date_tag],
            dtype=float
        )

        results[loc_key][date_tag] = sliding_window_mi(
            X,
            Y,
            WINDOW_SIZE,
            N_NEIGHBORS
        )

        n_calculated += 1


print("=" * 60)
print("MI calculation finished")
print("Calculated MI spectra:", n_calculated)
print("Missing location-date pairs:", n_missing_dates)
print("=" * 60)

# ============================================================
# 4.1 Verify structure of results after the loop
# ============================================================

example_loc = list(results.keys())[0]
example_date = date_tags[0]

print(example_loc)
print(example_date)

print(results[example_loc][example_date].shape)

''' it should be something like this:
results
│
├── class5_loc000
│     ├── 2022-02-24 → MI spectrum
│     ├── 2022-02-28 → MI spectrum
│     └── ...
├── class5_loc001
├── class11_loc000
└── ...
'''

# ============================================================
# 4.1 Save results before plotting
# ============================================================

with open("MI_results_all_122locations_13dates.pkl", "wb") as f:
    pickle.dump(
        {
            "results": results,
            "date_tags": date_tags,
            "window_centers_nm": window_centers_nm,
            "window_size": WINDOW_SIZE,
            "n_neighbors": N_NEIGHBORS,
            "pft_mapping": deltaG_data["pft_mapping"],
            "veg_classes": deltaG_data["veg_classes"],
        },
        f
    )

print("Saved MI results")

# ============================================================
# REPLACEMENT for sections 4b / 5 / 6
# Figure 4 — MI(ΔG, RCF) from the 122 fixed locations per class
# ============================================================
#
# Assumes the following already exist in the namespace (from your
# sections 0-4 of the new script):
#
#   results            : dict  loc_key -> date_tag -> MI curve (n_windows,)
#   deltaG_vectors     : dict  loc_key -> date_tag -> ΔG vector (413,)
#   common_locations   : sorted list of loc_key
#   date_tags          : list of 13 date strings
#   window_centers_nm  : 1-D array (the 413-band input axis — section 4c
#                        derives the shorter 388-pt MI axis `mi_axis_nm`)
#   PFTS, class_names, gray_regions, format_mi_ax, sliding_window_mi
#   WINDOW_SIZE, N_NEIGHBORS, dir_fig_out
#
# KEY DIFFERENCE vs the old script
# --------------------------------
#   OLD:  results[pft_label][veg_class]["mi_mean" | "mi_var" | "mi_noise"]
#         → 3 MI variants, each a list of 13 date-level curves built from
#           class-MEAN vectors (~15 000 pixels averaged → oversmoothed).
#
#   NEW:  results[loc_key][date_tag] → ONE MI curve per (location, date).
#         ΔG and RCF now come from the SAME single pixel location, so the
#         "mean filter bank" / "variance filter bank" distinction no longer
#         exists: there is exactly one MI(ΔG, RCF) curve per pair.
#         Sample size for the ± SD envelope is now
#           n = n_locations(class) × n_dates   (≈ 122 × 13)
#         instead of n = 13 dates.
# ============================================================

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# 4b. NOISE BASELINE  (subsampled — the full grid is expensive)
# ============================================================
#
# MI(ΔG, white noise) gives the floor of the estimator. Computing it for
# all ~1 586 location-date pairs doubles an already heavy loop, so it is
# estimated from a random subsample of pairs per class. Increase
# NOISE_PAIRS_PER_CLASS if you want a tighter floor.


def class_from_loc_key(loc_key):
    """'class37_loc088' -> 37"""
    m = re.match(r"class(\d+)_loc(\d+)$", loc_key)
    if m is None:
        raise ValueError(f"Unexpected location key format: {loc_key!r}")
    return int(m.group(1))


def loc_index_from_loc_key(loc_key):
    """'class37_loc088' -> 88"""
    return int(re.match(r"class(\d+)_loc(\d+)$", loc_key).group(2))


# --- build the full (class -> list of (loc_key, date_tag)) index ----------
pairs_by_class = defaultdict(list)
for loc_key, date_dict in results.items():
    cls = class_from_loc_key(loc_key)
    for date_tag in date_dict.keys():
        pairs_by_class[cls].append((loc_key, date_tag))

print("=" * 60)
print("Pairs available per class")
print("=" * 60)
for cls in sorted(pairs_by_class):
    n_loc = len({lk for lk, _ in pairs_by_class[cls]})
    print(f"  class {cls:>2}  ({class_names.get(cls, '?'):>5}) : "
          f"{n_loc:>3} locations × dates = {len(pairs_by_class[cls]):>5} pairs")

# --- noise MI ------------------------------------------------------------
print("\n" + "=" * 60)
print("Computing noise baseline MI(ΔG, white noise) ...")
print("=" * 60)

rng_pick  = np.random.default_rng(NOISE_SEED)
noise_by_class = defaultdict(list)

for cls in sorted(pairs_by_class):
    pairs = pairs_by_class[cls]

    if NOISE_PAIRS_PER_CLASS is not None and len(pairs) > NOISE_PAIRS_PER_CLASS:
        idx   = rng_pick.choice(len(pairs), NOISE_PAIRS_PER_CLASS, replace=False)
        chosen = [pairs[i] for i in idx]
    else:
        chosen = pairs

    print(f"  class {cls:>2}: {len(chosen)} noise pairs")

    for j, (loc_key, date_tag) in enumerate(chosen):
        X = np.asarray(deltaG_vectors[loc_key][date_tag], dtype=float)
        # deterministic but different noise realisation per pair
        Y_noise = np.random.default_rng(NOISE_SEED + 1000 * cls + j).standard_normal(len(X))
        noise_by_class[cls].append(
            sliding_window_mi(X, Y_noise, WINDOW_SIZE, N_NEIGHBORS)
        )

print("Noise baseline finished.\n")


# ============================================================
# 4c. BUILD THE WAVELENGTH AXIS FOR THE MI CURVES
# ============================================================
#
# The `window_centers_nm` stored in the .pkl is the axis of the INPUT
# vectors (413 bands). The sliding-window MI output is shorter:
#     n_windows = 413 - WINDOW_SIZE + 1 = 388
# Plotting the 413-band axis against a 388-point curve raises
#     ValueError: 'x' has size 413, but 'y1' has an unequal size of 388
#
# So derive the window-centre axis: centre of window i is the mean of
# the input wavelengths it spans. This preserves the true (possibly
# non-uniform) band spacing instead of assuming the 377.8 + 5.01·i grid.

_example_curve = results[list(results.keys())[0]][
    sorted(results[list(results.keys())[0]].keys())[0]
]
_n_windows = len(_example_curve)
_src_axis  = np.asarray(window_centers_nm, dtype=float)

if len(_src_axis) == _n_windows:
    # already a window axis — use as is
    mi_axis_nm = _src_axis

elif len(_src_axis) == _n_windows + WINDOW_SIZE - 1:
    # full band axis — collapse each window to its mean wavelength
    mi_axis_nm = np.array([
        _src_axis[i:i + WINDOW_SIZE].mean()
        for i in range(_n_windows)
    ])

else:
    # fallback: analytic AVIRIS-NG grid used in the original script
    print(f"  WARNING: axis length {len(_src_axis)} matches neither "
          f"{_n_windows} nor {_n_windows + WINDOW_SIZE - 1}; "
          f"falling back to the analytic 377.8 + 5.01·i grid.")
    mi_axis_nm = np.array([
        377.8 + 5.01 * ((i + (i + WINDOW_SIZE - 1)) / 2)
        for i in range(_n_windows)
    ])

print(f"MI curve length : {_n_windows}")
print(f"Stored axis     : {len(_src_axis)} points "
      f"({_src_axis[0]:.1f} - {_src_axis[-1]:.1f} nm)")
print(f"MI plot axis    : {len(mi_axis_nm)} points "
      f"({mi_axis_nm[0]:.1f} - {mi_axis_nm[-1]:.1f} nm)\n")

assert len(mi_axis_nm) == _n_windows

# ============================================================
# REVISED sections 5 / 6
#   - all three SD modes computed & stored (+ variance diagnostics)
#   - per-class figure: real panel spacing, short y-label
#   - optional shared y-limits so panels/figures are comparable
# ============================================================
#
# Requires from earlier sections:
#   results, pairs_by_class, noise_by_class, mi_axis_nm,
#   PFTS, class_names, format_mi_ax, WINDOW_SIZE, N_NEIGHBORS,
#   dir_fig_out, date_tags, window_centers_nm
# ============================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# 5. AGGREGATE — per class, all three SD definitions
# ============================================================
#
# SD_MODE controls what the shaded envelope means:
#
#   "pooled"    → SD across ALL location-date pairs of a class.
#                 Widest band; mixes spatial + temporal variability.
#
#   "locations" → per location, average the MI curve over its dates;
#                 then take SD across the 122 location-means.
#                 Isolates SPATIAL variability (recommended: it is the
#                 quantity that matches the 122-pixel subsample claim).
#
#   "dates"     → per date, average the MI curve over locations;
#                 then take SD across the 13 date-means.
#                 This is the closest analogue of the OLD figure's
#                 "±1 SD across dates".
#
# The three envelopes are NOT arbitrary alternatives; they are related
# by the law of total variance (for balanced groups):
#
#   Var_pooled  =  Var(location means)  +  mean(within-location Var)
#               =  Var(date means)      +  mean(within-date Var)
#
# So POOLED SD is mathematically >= both of the others, always.
# If the "pooled" figure did not *look* widest, it is almost certainly
# the autoscaled y-axis: with ylim=None each figure rescales to its own
# data, so a wider band drags the limits out with it and the band
# occupies the same fraction of the panel. Set MI_YLIM (below) to a
# fixed range and re-run the three modes to compare them honestly.
# Section 5b prints the numbers so you don't have to trust the eye.

SD_MODE = "dates" # "locations" # "pooled"          # "pooled" | "locations" | "dates"
assert SD_MODE in ("pooled", "locations", "dates")

aggregated = {}

for cls in sorted(pairs_by_class):

    by_loc  = defaultdict(list)
    by_date = defaultdict(list)
    pooled  = []

    for loc_key, date_tag in pairs_by_class[cls]:
        curve = np.asarray(results[loc_key][date_tag], dtype=float)
        pooled.append(curve)
        by_loc[loc_key].append(curve)
        by_date[date_tag].append(curve)

    pooled_arr = np.array(pooled)                                          # (n_pairs, n_win)
    loc_means  = np.array([np.mean(v, axis=0) for v in by_loc.values()])   # (n_loc,   n_win)
    date_means = np.array([np.mean(v, axis=0) for v in by_date.values()])  # (n_date,  n_win)

    # within-group variances (the parts that pooled adds on top)
    within_loc_var  = np.mean([np.var(v, axis=0) for v in by_loc.values()],  axis=0)
    within_date_var = np.mean([np.var(v, axis=0) for v in by_date.values()], axis=0)

    sd_pooled    = pooled_arr.std(axis=0)
    sd_locations = loc_means.std(axis=0)
    sd_dates     = date_means.std(axis=0)

    noise_arr = np.array(noise_by_class[cls])

    aggregated[cls] = {
        "mi_avg":          pooled_arr.mean(axis=0),
        "mi_sd_pooled":    sd_pooled,
        "mi_sd_locations": sd_locations,
        "mi_sd_dates":     sd_dates,
        "mi_sd":           {"pooled":    sd_pooled,
                            "locations": sd_locations,
                            "dates":     sd_dates}[SD_MODE],
        "mi_sem":          sd_pooled / np.sqrt(pooled_arr.shape[0]),
        "within_loc_var":  within_loc_var,
        "within_date_var": within_date_var,
        "mi_noise_avg":    noise_arr.mean(axis=0),
        "mi_noise_p95":    np.percentile(noise_arr, 95, axis=0),
        "loc_means":       loc_means,
        "date_means":      date_means,
        "n_pairs":         pooled_arr.shape[0],
        "n_locations":     loc_means.shape[0],
        "n_dates":         date_means.shape[0],
    }


# ============================================================
# 5b. DIAGNOSTIC — why 'pooled' should be the widest band
# ============================================================

print("=" * 78)
print(f"Aggregation finished   (plotting SD_MODE = '{SD_MODE}')")
print("=" * 78)
print(f"{'class':>6} {'n_loc':>6} {'n_date':>7} {'n_pairs':>8} "
      f"{'SD pooled':>10} {'SD locs':>9} {'SD dates':>9}  check")
print("-" * 78)

for cls in sorted(aggregated):
    a = aggregated[cls]

    sp = a["mi_sd_pooled"].mean()
    sl = a["mi_sd_locations"].mean()
    sd = a["mi_sd_dates"].mean()

    # law of total variance, band-wise (exact only for balanced groups)
    lhs = a["mi_sd_pooled"] ** 2
    rhs = a["mi_sd_locations"] ** 2 + a["within_loc_var"]
    rel_err = np.abs(lhs - rhs).mean() / max(lhs.mean(), 1e-12)

    flag = "ok" if (sp >= sl - 1e-12 and sp >= sd - 1e-12) else "!! pooled NOT widest"

    print(f"{cls:>6} {a['n_locations']:>6} {a['n_dates']:>7} {a['n_pairs']:>8} "
          f"{sp:>10.4f} {sl:>9.4f} {sd:>9.4f}  {flag} "
          f"(var decomp rel.err {rel_err:.1e})")

print("-" * 78)
print("SD pooled >= SD locs and >= SD dates is guaranteed by the variance")
print("decomposition. If the plotted band looks narrower, the cause is the")
print("autoscaled y-axis, not the statistics — set MI_YLIM below.\n")


# ============================================================
# 6. PLOTTING
# ============================================================

# --- aesthetics -------------------------------------------------------
Y_LABEL          = "MI (nats)"   # short — was "Mutual Information"
MI_YLIM          = None          # e.g. (0.0, 0.45) to compare SD modes fairly
SHARE_YLIM       = True          # common y-range across panels of a figure
YLIM_PAD         = 0.6          # fractional headroom when auto-sharing

PANEL_H_COMBINED = 6.0           # inches per PFT panel
PANEL_H_PERCLASS = 4.6           # inches per class panel
HSPACE_COMBINED  = 0.42
HSPACE_PERCLASS  = 1.0          # was 0.35 — titles were landing on the
                                 # rotated x tick labels of the panel above


def _auto_ylim(curves, pad=YLIM_PAD):
    """Common y-limits from every curve/envelope that will be drawn."""
    lo = min(np.min(c) for c in curves)
    hi = max(np.max(c) for c in curves)
    span = hi - lo
    return (lo - pad * span, hi + pad * span)


def make_combined_figure(filename_tag,
                         sd_mode=None,
                         show_class_lines=True,
                         show_noise_p95=True,
                         ylim=MI_YLIM):
    """One panel per PFT: PFT mean, +/-1 SD envelope, noise floor, class lines."""
    sd_mode = sd_mode or SD_MODE
    sd_key  = f"mi_sd_{sd_mode}"

    fig, axes = plt.subplots(len(PFTS), 1,
                             figsize=(8, PANEL_H_COMBINED * len(PFTS)),
                             gridspec_kw={"hspace": HSPACE_COMBINED})
    axes = np.atleast_1d(axes)

    panel_data = []
    for pft in PFTS:
        veg_classes = [c for c in pft["veg_classes"] if c in aggregated]
        if not veg_classes:
            panel_data.append(None)
            continue

        all_avg   = [aggregated[c]["mi_avg"]       for c in veg_classes]
        all_sd    = [aggregated[c][sd_key]         for c in veg_classes]
        noise_avg = [aggregated[c]["mi_noise_avg"] for c in veg_classes]
        noise_p95 = [aggregated[c]["mi_noise_p95"] for c in veg_classes]

        panel_data.append({
            "veg_classes": veg_classes,
            "all_avg":     all_avg,
            "avg":         np.mean(all_avg,   axis=0),
            "sd":          np.mean(all_sd,    axis=0),
            "noise":       np.mean(noise_avg, axis=0),
            "noise_p95":   np.mean(noise_p95, axis=0),
            "n_pairs":     sum(aggregated[c]["n_pairs"]     for c in veg_classes),
            "n_locs":      sum(aggregated[c]["n_locations"] for c in veg_classes),
        })

    if ylim is None and SHARE_YLIM:
        pool = []
        for p in panel_data:
            if p is None:
                continue
            pool += [p["avg"] - p["sd"], p["avg"] + p["sd"], p["noise"]]
            pool += p["all_avg"]
            if show_noise_p95:
                pool.append(p["noise_p95"])
        ylim = _auto_ylim(pool)

    for i, (ax, pft, p) in enumerate(zip(axes, PFTS, panel_data)):
        if p is None:
            ax.set_visible(False)
            continue

        color = pft["color_mean"]

        ax.fill_between(mi_axis_nm, p["avg"] - p["sd"], p["avg"] + p["sd"],
                        color=color, alpha=0.25, label=f"±1 SD ({sd_mode})")

        ax.plot(mi_axis_nm, p["avg"], color=color, linewidth=2.0,
                label="MI(ΔG₁₃, RCF)")

        ax.plot(mi_axis_nm, p["noise"], linestyle='--', color='black',
                linewidth=1.2, label="MI(ΔG₁₃, noise)")

        if show_noise_p95:
            ax.plot(mi_axis_nm, p["noise_p95"], linestyle=':', color='black',
                    linewidth=1.0, label="noise p95")

        if show_class_lines and len(p["veg_classes"]) > 1:
            for cls, avg_cls in zip(p["veg_classes"], p["all_avg"]):
                ax.plot(mi_axis_nm, avg_cls, linewidth=0.8, alpha=0.45,
                        label=class_names.get(cls, f"class {cls}"))

        format_mi_ax(ax, mi_axis_nm,
                     ylabel=Y_LABEL,
                     ylim=ylim,
                     x_axis_title="Wavelength (nm)" if i == len(axes) - 1 else None)

        ax.set_title(
            f"{['A','B','C'][i]}  —  {pft['label']}  |  MI(ΔG₁₃, RCF)\n"
            f"[{p['n_locs']} locations, {p['n_pairs']} location-date pairs]",
            fontsize=14, fontweight='bold', loc='left', pad=10
        )

        leg = ax.legend(loc='upper right', fontsize=12, framealpha=0.7,
                        ncol=3, columnspacing=0.8, handletextpad=0.5)
        leg.set_zorder(20)

    # NOTE: no tight_layout() — it recomputes spacing and silently undoes
    # the hspace we just set. subplots_adjust + bbox_inches='tight' keeps
    # the gaps and still trims the outer margins.
    fig.subplots_adjust(hspace=HSPACE_COMBINED)

    out_file = os.path.join(dir_fig_out, f"Figure_4_{filename_tag}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


def make_per_class_figure(filename_tag, sd_mode=None, ylim=MI_YLIM):
    """One panel per veg class."""
    sd_mode = sd_mode or SD_MODE
    sd_key  = f"mi_sd_{sd_mode}"

    classes = sorted(aggregated)
    n = len(classes)

    fig, axes = plt.subplots(n, 1,
                             figsize=(8, PANEL_H_PERCLASS * n),
                             gridspec_kw={"hspace": HSPACE_PERCLASS})
    axes = np.atleast_1d(axes)

    if ylim is None and SHARE_YLIM:
        pool = []
        for cls in classes:
            a = aggregated[cls]
            pool += [a["mi_avg"] - a[sd_key],
                     a["mi_avg"] + a[sd_key],
                     a["mi_noise_avg"]]
        ylim = _auto_ylim(pool)

    for i, (ax, cls) in enumerate(zip(axes, classes)):
        a = aggregated[cls]

        ax.fill_between(mi_axis_nm,
                        a["mi_avg"] - a[sd_key],
                        a["mi_avg"] + a[sd_key],
                        color="orange", alpha=0.25,
                        label=f"±1 SD ({sd_mode})")

        ax.plot(mi_axis_nm, a["mi_avg"], color="orange", linewidth=2.0,
                label="MI(ΔG₁₃, RCF)")

        ax.plot(mi_axis_nm, a["mi_noise_avg"], linestyle='--', color='black',
                linewidth=1.2, label="MI(ΔG₁₃, noise)")

        format_mi_ax(ax, mi_axis_nm,
                     ylabel=Y_LABEL,
                     ylim=ylim,
                     x_axis_title="Wavelength (nm)" if i == len(axes) - 1 else None)

        ax.set_title(
            f"class {cls}  ({class_names.get(cls, '?')})  |  "
            f"{a['n_locations']} locations, {a['n_pairs']} pairs",
            fontsize=13, fontweight='bold', loc='left', pad=10
        )

        leg = ax.legend(loc='upper right', fontsize=10, framealpha=0.7, ncol=3)
        leg.set_zorder(20)

    fig.subplots_adjust(hspace=HSPACE_PERCLASS)

    out_file = os.path.join(dir_fig_out, f"Figure_4_{filename_tag}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


# ------------------------------------------------------------
# 6c. DIAGNOSTIC FIGURE — the three SD modes on the SAME axes
#     This is the honest way to compare them: one panel per PFT,
#     identical y-range, three nested envelopes.
# ------------------------------------------------------------

def make_sd_comparison_figure(filename_tag="MI_SD_mode_comparison"):
    modes  = ["pooled", "locations", "dates"]
    alphas = [0.18, 0.28, 0.40]
    cols   = ["tab:red", "tab:blue", "tab:green"]

    fig, axes = plt.subplots(len(PFTS), 1,
                             figsize=(8, PANEL_H_COMBINED * len(PFTS)),
                             gridspec_kw={"hspace": HSPACE_COMBINED})
    axes = np.atleast_1d(axes)

    pool_for_ylim = []
    panels = []
    for pft in PFTS:
        veg_classes = [c for c in pft["veg_classes"] if c in aggregated]
        if not veg_classes:
            panels.append(None)
            continue
        avg = np.mean([aggregated[c]["mi_avg"] for c in veg_classes], axis=0)
        sds = {m: np.mean([aggregated[c][f"mi_sd_{m}"] for c in veg_classes], axis=0)
               for m in modes}
        panels.append((veg_classes, avg, sds))
        for m in modes:
            pool_for_ylim += [avg - sds[m], avg + sds[m]]

    ylim = MI_YLIM or _auto_ylim(pool_for_ylim)

    for i, (ax, pft, p) in enumerate(zip(axes, PFTS, panels)):
        if p is None:
            ax.set_visible(False)
            continue
        _, avg, sds = p

        # widest first so narrower ones stay visible on top
        for m, alpha, col in zip(modes, alphas, cols):
            ax.fill_between(mi_axis_nm, avg - sds[m], avg + sds[m],
                            color=col, alpha=alpha,
                            label=f"±1 SD {m} (mean {sds[m].mean():.3f})")

        ax.plot(mi_axis_nm, avg, color='black', linewidth=1.8, label="MI mean")

        format_mi_ax(ax, mi_axis_nm,
                     ylabel=Y_LABEL,
                     ylim=ylim,
                     x_axis_title="Wavelength (nm)" if i == len(axes) - 1 else None)

        ax.set_title(f"{['A','B','C'][i]}  —  {pft['label']}  |  SD comparison",
                     fontsize=14, fontweight='bold', loc='left', pad=10)

        leg = ax.legend(loc='upper right', fontsize=10, framealpha=0.7, ncol=2)
        leg.set_zorder(20)

    fig.subplots_adjust(hspace=HSPACE_COMBINED)

    out_file = os.path.join(dir_fig_out, f"Figure_4_{filename_tag}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

make_combined_figure(
    filename_tag=f"MI_deltaG_RCF_all_PFTs_122_pixel_locations_SD-{SD_MODE}"
)

make_per_class_figure(
    filename_tag=f"MI_deltaG_RCF_per_class_122_pixel_locations_SD-{SD_MODE}"
)

make_sd_comparison_figure()


# ------------------------------------------------------------
# 6d. Save aggregated arrays (light — no raw per-location curves)
# ------------------------------------------------------------

agg_out = {
    cls: {k: v for k, v in a.items() if k not in ("loc_means", "date_means")}
    for cls, a in aggregated.items()
}

with open(os.path.join(dir_fig_out, "MI_aggregated_122locations.pkl"), "wb") as f:
    pickle.dump(
        {
            "aggregated":        agg_out,
            "sd_mode":           SD_MODE,
            "mi_axis_nm":        mi_axis_nm,
            "window_centers_nm": window_centers_nm,
            "window_size":       WINDOW_SIZE,
            "n_neighbors":       N_NEIGHBORS,
            "date_tags":         date_tags,
            "class_names":       class_names,
        },
        f
    )

print("\nDone. Figures + aggregated arrays saved.")


