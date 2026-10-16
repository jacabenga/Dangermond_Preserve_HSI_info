

# ============================================================
# Figure_3_publication_ready_clean_code_only-plotting_v4.py
# ============================================================
#
# date:    27.02.2026
# author:  Jaime C. Revenga
#
# PURPOSE:
#   Plotting-only version of Figure 3 (Dangermond Info Paper).
#   No heavy computation (no KLPD, no RCF).
#   Loads pre-computed data vectors from disk and plots directly.
#   Designed to be fully reproducible and shareable.
#
# FIGURE STRUCTURE (combined figure):
#
#   Left column  — Spectral regions with KLPD (ΔShape, 380–2500 nm) on a given date
#       A: Shrub
#       B: Herbaceous
#       C: Woodland / Tree
#
#   Right column — Mean filter excitations (380–2500 nm) on a given date
#       A: Shrub
#       B: Herbaceous
#       C: Woodland / Tree
#
# DATA SOURCES (pre-computed, stored as .pkl files):
#
#   Left column  (delta_G vectors):
#     Shrub:       .../Figure_3/data_in/shrub/stored_delta_G_13_vectors/
#     Herbaceous:  .../Figure_3/data_in/herbaceous/stored_delta_G_13_vectors/
#     Woodland:    .../Figure_3/data_in/woodland/stored_delta_G_13_vectors/
#
#   Right column (kernel excitation vectors):
#     Shrub:       .../Figure_3/data_in/shrub/stored_kernel_excitation_vectors/
#     Herbaceous:  .../Figure_3/data_in/herbaceous/stored_kernel_excitation_vectors/
#     Woodland:    .../Figure_3/data_in/woodland/stored_kernel_excitation_vectors/
#
# ============================================================

# Notes : now 'scale' is an argument of format_spectral_ax.
            # You can call this function with either 'log' or 'linear' scaling.



# ============================================================
# 0. IMPORTS
# ============================================================

import os
import pickle
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter


# ============================================================
# 1. USER PARAMETERS  ← edit these paths to match your machine
# ============================================================

# ---- Base directory containing all Figure_3 data ----
base_dir = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_3\data_in'

# ---- Date of interest ----
date_list = ["2022-02-28",
            "2022-03-08", "2022-03-16", "2022-03-22",
            "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
            "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29" ]

date_tag = "2022-05-11"   # select it manually
#date_tag = date_list[0]   # or choose from date list

# ---- Output directory for the saved figure ----
dir_fig_out = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_3\figures_out'
os.makedirs(dir_fig_out, exist_ok=True)


# ============================================================
# 2. FIXED SPECTRAL PARAMETERS  (do not change)
# ============================================================

N_BANDS   = 425
WINDOW    = 13

# Band index → wavelength (nm)
wavelengths_nm = [377.8 + 5.01 * b for b in range(N_BANDS)]

# Rolling window centre → wavelength (nm)   (length = N_BANDS - WINDOW + 1 = 413)
window_centers_nm = [
    377 + 5.01 * ((i + i + WINDOW - 1) / 2)
    for i in range(N_BANDS - WINDOW + 1)
]

# Gray background bands (water-vapour absorption + sensor limits)
gray_regions = [
    (0,    460),    # sensor blue-edge artefact
    (1315, 1460),   # water-vapour, CO2 absorption
    (1765, 2010),   # water-vapour, CO2 absorption
    (2440, 2510),   # sensor SWIR artefact
]


# ============================================================
# 3. COLOUR PALETTE  (colorblind-safe)
# ============================================================

colors = {
                                                        # shrub classes
    5:  "#0072B2",   # Black sage scrub
    11: "#E69F00",   # California sagebrush scrub
    20: "#009E73",   # Coyote brush scrub
    31: "#56B4E9",   # Menzie's golden bush scrub
    37: "#D55E00",   # Purple sage scrub
                                                        # herbaceous classes
    33: "#CC79A7",   # Non-native grassland
    43: "#F0E442",   # Semi-natural herbaceous stands
                                                        # tree class
    16: "#009E73",   # Coast live oak woodland
}

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


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def load_delta_G(pft_folder, date_tag, pft_label):
    """
    Load a pre-computed delta_G dictionary from a .pkl file.

    Parameters
    ----------
    pft_folder : str   Path to .../stored_delta_G_13_vectors/
    date_tag   : str   e.g. "2022-05-11"
    pft_label  : str   e.g. "Tree", "Shrub", "Herbaceous"

    Returns
    -------
    dict  {veg_class_int: np.ndarray of shape (413,)}
    """
    fname = os.path.join(pft_folder, f"delta_G_{date_tag}_{pft_label}.pkl")
    with open(fname, "rb") as f:
        data = pickle.load(f)
    print(f"  [delta_G]    loaded: {fname}")
    return data


def load_kernel_stats(pft_folder, date_tag, pft_label):
    """
    Load pre-computed RCF kernel-bank statistics from a .pkl file.

    Parameters
    ----------
    pft_folder : str   Path to .../stored_kernel_excitation_vectors/
    date_tag   : str   e.g. "2022-05-11"

    Returns
    -------
    dict with keys including 'mean_over_kernels_per_class'
         → {veg_class_int: np.ndarray of shape (413,)}
    """
    fname = os.path.join(pft_folder, f"RCF_kernel_bank_stats_{date_tag}_{pft_label}.pkl")
    with open(fname, "rb") as f:
        data = pickle.load(f)
    print(f"  [kernel]     loaded: {fname}")
    return data


def shade_gray_regions(ax):
    """Add gray shading for absorption / artefact wavelength bands."""
    for r_start, r_end in gray_regions:
        #ax.axvspan(r_start, r_end, facecolor="white", alpha=1.0, zorder=10)  # to completely cancel all those regions
	    ax.axvspan(r_start, r_end, facecolor="lightgray", alpha=1.0, zorder=10)  # if you want to see through, to pick those awkward values


def format_spectral_ax(ax, ylabel="", ylim=None, scale= 'linear'):   # , autoscale_y = True,  here the scale can be changed, 'log', 'linear'
    """Apply common formatting to a spectral axis."""
    ax.set_xlabel("Wavelength (nm)", fontsize=12)   # ↑ bigger
    ax.set_ylabel(ylabel, fontsize=12)              # ↑ bigger
    ax.set_xlim(380, 2500)

    # ticks handled below (see section 4)
    xticks = np.arange(400, 2501, 100)
    ax.set_xticks(xticks)

    # Show only every second label
    xtick_labels = [str(int(x)) if i % 2 == 0 else "" for i, x in enumerate(xticks)]
    ax.set_xticklabels(xtick_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    ax.tick_params(axis='y', labelsize=9)  # ↑ y-ticks bigger

    ax.grid(True, alpha=0.3)

    #if autoscale_y:
    #    ax.autoscale(axis='y')
    #    pass

    shade_gray_regions(ax)
    ax.set_yscale(scale)


# ============================================================
# 5. LOAD ALL PRE-COMPUTED DATA
# ============================================================

print("Loading pre-computed data ...")
print("-" * 50)

# ---- Left column: delta_G ----

delta_G_shrub = load_delta_G(
    pft_folder = os.path.join(base_dir, "shrub",       "stored_delta_G_13_vectors"),
    date_tag   = date_tag,
    pft_label  = "Shrub"
)

delta_G_herb = load_delta_G(
    pft_folder = os.path.join(base_dir, "herbaceous",  "stored_delta_G_13_vectors"),
    date_tag   = date_tag,
    pft_label  = "Herbaceous"
)

delta_G_tree = load_delta_G(
    pft_folder = os.path.join(base_dir, "woodland",    "stored_delta_G_13_vectors"),
    date_tag   = date_tag,
    pft_label  = "Tree"
)

# ---- Right column: kernel excitations ----

kernel_shrub = load_kernel_stats(
    pft_folder = os.path.join(base_dir, "shrub",       "stored_kernel_excitation_vectors"),
    date_tag   = date_tag,
    pft_label  = "Shrub"
)

kernel_herb = load_kernel_stats(
    pft_folder = os.path.join(base_dir, "herbaceous",  "stored_kernel_excitation_vectors"),
    date_tag   = date_tag,
    pft_label  = "Herbaceous"
)

kernel_tree = load_kernel_stats(
    pft_folder = os.path.join(base_dir, "woodland",    "stored_kernel_excitation_vectors"),
    date_tag   = date_tag,
    pft_label  = "Tree"
)

print("-" * 50)
print("All data loaded successfully.\n")


# ============================================================
# 6. PLOT  — combined 3 × 2 figure
# ============================================================

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(
    3, 2,
    figure=fig,
    hspace=0.35,   # ↓ tighter vertically
    wspace=0.18    # ↓ tighter horizontally
)

#row_labels  = ["A", "B", "C"]
panel_labels = [["A", "B"],
                ["C", "D"],
                ["E", "F"]]
pft_titles  = ["Shrub", "Herbaceous", "Woodland"] # Woodland / Tree

delta_G_all = [delta_G_shrub, delta_G_herb,  delta_G_tree]
kernel_all  = [kernel_shrub,  kernel_herb,   kernel_tree]


# ------------------------------------------------------------------
# LEFT COLUMN  — ΔShape (KLPD rolling window, width=13 bands)
# ------------------------------------------------------------------

for row, (delta_G_dict, pft_label) in enumerate(zip(delta_G_all, pft_titles)):

    ax = fig.add_subplot(gs[row, 0])

    for cls, delta_G_vec in delta_G_dict.items():
        c     = colors.get(cls, "gray")
        label = class_names.get(cls, f"class {cls}")
        ax.plot(window_centers_nm, np.clip(delta_G_vec, 1e-10, None),   # -> we add this to avoid attempting to plot negative log
                color=c, linewidth=1.5, alpha=0.85, label=label)

    if row == 0:
        ylim = (1e-5, 2.2e-2)  # Shrub
    elif row == 1:
        ylim = (1e-6, 3e-4)  # Herbaceous

    elif row == 2:
        ylim = (1e-5, 1.1e-2)  # Tree

    format_spectral_ax(
        ax,
        ylabel=r"$\Delta$Shape ($\Delta G_{13}$)",
        #ylabel = r"$\Delta$Shape ($\Delta G_{13}$) [$\times 10^{-2}$]"
    )
    ax.set_ylim(ylim)

    if row ==0:
        ax.set_ylabel(r"$\Delta$Shape ($\Delta G_{13}$) [$\times 10^{-2}$]")
    elif row ==1:
        ax.set_ylabel(r"$\Delta$Shape ($\Delta G_{13}$) [$\times 10^{-4}$]")
    if row ==2:
        ax.set_ylabel(r"$\Delta$Shape ($\Delta G_{13}$) [$\times 10^{-2}$]")


    # scientific notation in ticks
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset = False)
    ax.yaxis.get_offset_text().set_visible(False)

    ax.set_title(
        f"{panel_labels[row][0]}  —  {pft_label}  |  $\\Delta$Shape  ({date_tag})",
        fontsize=13,  # ↑ from 10
        fontweight='bold',
        loc='left'
    )

    leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
    leg.set_zorder(40)

# ------------------------------------------------------------------
# RIGHT COLUMN  — Mean kernel-bank excitation (normalised)
# ------------------------------------------------------------------

for row, (kernel_dict, pft_label) in enumerate(zip(kernel_all, pft_titles)):

    ax = fig.add_subplot(gs[row, 1])

    mean_over_kernels_per_class = kernel_dict["mean_over_kernels_per_class"]

    for cls, mean_vec in mean_over_kernels_per_class.items():
        # z-score normalise (same as in the computation script)
        mean_vec_norm = (mean_vec - mean_vec.mean()) / (mean_vec.std() + 1e-8)

        c     = colors.get(int(cls), "gray")
        label = class_names.get(int(cls), f"class {cls}")
        ax.plot(window_centers_nm, np.clip(np.abs(mean_vec_norm), 1e-10, None),  # -> avoids attempt to plot negative log
                color=c, linewidth=1.5, alpha=0.85, label=label)

    if row == 0:
        ylim = (1e-3, 1.75)  # Shrub
    elif row == 1:
        ylim = (5e-3, 1.0)  # Herbaceous
    elif row == 2:
        ylim = (1e-1, 0.7)  # Tree

    format_spectral_ax(
        ax,
        ylabel="Filter excitation (|z-score|)",

    )
    ax.set_ylim(ylim)

    ax.set_title(
        f"{panel_labels[row][1]}  —  {pft_label}  |  Filter excitation  ({date_tag})",
        fontsize=13,
        fontweight='bold',
        loc='left'
    )


    leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.7)
    leg.set_zorder(40)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
plt.tight_layout()

out_file = os.path.join(dir_fig_out, f"Figure_3_{date_tag}_v3.png")
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {out_file}")