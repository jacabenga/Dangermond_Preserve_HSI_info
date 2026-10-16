# ============================================================
# Figure_6_publication_ready_clean_code_only-plotting_class_mean_v1.py
# ============================================================
#
# date:    27.02.2026
# author:  Jaime C. Revenga
#
# PURPOSE:
#   Plotting-only version of Figure 6 (Dangermond Info Paper).
#
#   "Left panels (a, b, c) show mutual information over time between
#    ΔG₁₃(λ) and class mean of the filter excitations within
#    individual plant functional types:
#       (a) shrubland   (b) grassland   (c) woodland"
#
#   Three panels stacked vertically, one per PFT.
#   Each panel shows global MI(ΔG₁₃, class mean of RCF) over the
#   13 acquisition dates — one line per vegetation community.
#
# DATA SOURCES (pre-computed, stored as .pkl files):
#   .../Figure_6/data_in/{pft}/stored_delta_G_13_vectors/*.pkl
#   .../Figure_6/data_in/{pft}/stored_kernel_excitation_vectors/*.pkl
#
#   where {pft} is one of:  shrub | herbaceous | woodland
#
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression


# ============================================================
# 1. USER PARAMETERS  ← edit these paths to match your machine
# ============================================================

base_dir = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_6\data_in'

dir_fig_out = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_6\figures_out'
os.makedirs(dir_fig_out, exist_ok=True)

date_list = [
    "2022-02-24", "2022-02-28",
    "2022-03-08", "2022-03-16", "2022-03-22",
    "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
    "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29"
]


# ============================================================
# 2. FIXED PARAMETERS
# ============================================================

N_NEIGHBORS = 5

PFTS = [
    {
        "folder":      "shrub",
        "panel_label": "A",
        "pft_title":   "Shrubland",
        "veg_classes": [5, 11, 20, 31, 37],
        "label_map": {
            5:  "Black sage scrub",
            11: "California sagebrush scrub",
            20: "Coyote brush scrub",
            31: "Menzies golden bush scrub",
            37: "Purple sage scrub",
        },
    },
    {
        "folder":      "herbaceous",
        "panel_label": "B",
        "pft_title":   "Grassland",
        "veg_classes": [33, 43],
        "label_map": {
            33: "Non-native grassland",
            43: "Semi-natural herbaceous stands",
        },
    },
    {
        "folder":      "woodland",
        "panel_label": "C",
        "pft_title":   "Woodland",
        "veg_classes": [16],
        "label_map": {
            16: "Coast live oak woodland",
        },
    },
]

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

CLASS_MARKERS = {
    5:  "o",
    11: "s",
    16: "D",
    20: "^",
    31: "v",
    33: "P",
    37: "X",
    43: "*",
}


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_mi(X, Y, n_neighbors=5):
    """Global MI between two 1-D signals."""
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    return mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=0)[0]


# ============================================================
# 4. COMPUTE GLOBAL MI OVER ALL DATES FOR ALL PFTs
# ============================================================

print("Computing global MI over time for all PFTs ...")
print("=" * 60)

# pft_data[pft_title][veg_class] = list of float, one value per date

pft_data = {}

for pft in PFTS:
   print(f"\n  PFT: {pft['pft_title']}")
    dir_klpd    = os.path.join(base_dir, pft["folder"], "stored_delta_G_13_vectors")
    dir_kernels = os.path.join(base_dir, pft["folder"], "stored_kernel_excitation_vectors")

    klpd_files   = [f for f in os.listdir(dir_klpd)    if f.endswith(".pkl")]
    kernel_files = [f for f in os.listdir(dir_kernels) if f.endswith(".pkl")]

    # Match files to dates by checking whether the date string appears in the filename.
    # This is robust to different sort orders across PFT folders.
    def find_file(file_list, date_tag):
        matches = [f for f in file_list if date_tag in f]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 file matching '{date_tag}', found {len(matches)}: {matches}"
            )
        return matches[0]

    class_store = {cls: [] for cls in pft["veg_classes"]}

    for d, date_tag in enumerate(date_list):
        print(f"    {d+1:02d}/{len(date_list)}  {date_tag}")

        all_delta_G    = load_pkl(os.path.join(dir_klpd,    find_file(klpd_files,   date_tag)))
        data_k         = load_pkl(os.path.join(dir_kernels, find_file(kernel_files, date_tag)))
        mean_per_class = data_k["mean_per_class"]

        for veg_class in pft["veg_classes"]:
            X      = np.array(all_delta_G[veg_class])
            Y_mean = mean_per_class[veg_class].mean(axis=0)
            if hasattr(Y_mean, "numpy"):
                Y_mean = Y_mean.numpy()

            class_store[veg_class].append(compute_mi(X, Y_mean, N_NEIGHBORS))

    pft_data[pft["pft_title"]] = class_store

print("\nMI computation complete.\n")


# ============================================================
# 5. PLOT — 3 vertically stacked panels
# ============================================================

x_pos = np.arange(len(date_list))

fig, axes = plt.subplots(3, 1, figsize=(12, 13),
                         gridspec_kw={"hspace": 0.35})

for i, (ax, pft) in enumerate(zip(axes, PFTS)):  # enumerate() gets you both index and element in iterable
    pft_title   = pft["pft_title"]
    veg_classes = pft["veg_classes"]
    label_map   = pft["label_map"]
    store       = pft_data[pft_title]

    for cls in veg_classes:
        c      = CLASS_COLORS.get(cls, "gray")
        marker = CLASS_MARKERS.get(cls, "o")

        ax.plot(x_pos, np.array(store[cls]),
                color=c, marker=marker,
                linewidth=1.8, markersize=5,
                label=label_map[cls])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(date_list, rotation=45, ha='right', fontsize=9)
    if i == 2:  # only bottom panel
        ax.set_xlabel("Acquisition date", fontsize=13)  # ↑ bigger
    else:
        ax.set_xlabel("")
    ax.set_ylabel(
        #"Global MI  (ΔG₁₃ ↔ class mean\nof RCF excitation)",
        "Global MI",
        fontsize=12
    )
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title(
        f"{pft['panel_label']} — {pft_title} | MI(ΔG₁₃, class mean of filter bank excitations)",
        fontsize=13, fontweight='bold', loc='left'
    )
    leg = ax.legend(
        fontsize=12,  # ↑ bigger
        loc='best',
        framealpha=0.85,
        ncol=2 if len(veg_classes) > 2 else 1
    )
    leg.set_zorder(20)

#fig.suptitle(
#    "Figure 6 — Global mutual information over time between ΔG₁₃(λ) and\n"
#    "class mean of filter excitations within each PFT",
#    fontsize=13, fontweight='bold', y=1.01
#)

out_file = os.path.join(dir_fig_out, "Figure_6_MI_over_time_3panels_class-mean_v2.png")

plt.tight_layout()
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {out_file}")