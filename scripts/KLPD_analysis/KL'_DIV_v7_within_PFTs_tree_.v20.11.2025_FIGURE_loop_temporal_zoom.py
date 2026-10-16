

# date: 14.10.2025

'''
... a mano y despacio

V6  --> same as V5, but her ewe include SWIR (425 bands).

V5 --> same as v4, but for 3 different plant functional types.

V4 is --> same as V3, but here, we include the gaps corresponding to the removed bands [194-208] U [319-354] idx
    , corresponding to [1350-1420] U [1800-1975] nm

'''


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.special import kl_div

from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# create dir_res_out

dir_res_out = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\results'
os.makedirs(dir_res_out, exist_ok=True)

# ----------------------------------------------------------------
# Load parquet
# ----------------------------------------------------------------

dir_parquet_files = r'D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\mosaics_labeled\df_parquet_streamed'

date_list = ['2022-02-24','2022-02-28',
            '2022-03-08', '2022-03-16', '2022-03-22',
            '2022-04-05', '2022-04-12', '2022-04-20', '2022-04-29',
            '2022-05-03', '2022-05-11', '2022-05-17', '2022-05-29']

# ----------------------------------------------------------------
# USER PARAMS
# ----------------------------------------------------------------

pft_class = 'Tree'  # 'Herbaceous' #  choose between: 'Tree' #

#veg_class_tag =  43 #  33 #                FOR HERBACEOUS    43: semi-natural herbaceous stands, 33: non-native Grassland
#veg_class_tag = 20 # 5 # 11 #20 #31 #37 #   FOR SHRUB 5:black sage scrub, 11:California sagebrush scrub, 20:Coyote brush scrub, 31:Menzie's golden brush scrub, 37: purple sage scrub
veg_class_tag = 16

# ---------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------

def kl_divergence(p, q):
    """Compute symmetric Kullback-Leibler divergence between two signals."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = np.nan_to_num(p, nan=1e-10)
    q = np.nan_to_num(q, nan=1e-10)
    p /= np.sum(p)
    q /= np.sum(q)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))


def get_energy(signal):
    """Compute energy (area under curve) using trapezoidal integration."""
    return np.trapezoid(signal)


def delta_shape(signal_1, signal_2):
    """Compute ΔShape between two reflectance signals."""
    k1 = get_energy(signal_1)
    k2 = get_energy(signal_2)
    KL_pq = kl_divergence(signal_1, signal_2)
    KL_qp = kl_divergence(signal_2, signal_1)
    delta_G = k1 * KL_pq + k2 * KL_qp
    return delta_G


def delta_energy(signal_1, signal_2):
    """Compute ΔEnergy between two reflectance signals (robust version)."""
    k1 = max(get_energy(signal_1), 1e-10)
    k2 = max(get_energy(signal_2), 1e-10)
    delta_W = (k1 - k2) * np.log((k1 + 1e-10) / (k2 + 1e-10))
    return delta_W

# ----------------------------------------------------------------
# Loop through dates
# ----------------------------------------------------------------


list_df_delta_G = []
parquet_files = sorted(os.listdir(dir_parquet_files))

print('looping through dates\n')
for date_tag, file_name in zip(date_list, parquet_files):

    print(f'looping through date {date_tag}\n')

    parquet_file = os.path.join(dir_parquet_files, file_name)

    # ---- Load ----
    df = pd.read_parquet(parquet_file)
    print(df.head())
    print()

    # Remove sentinel rows
    df_clean = df[~(df == -9999.0).any(axis=1)]

    # ---- Add PFT ----
    pft_mapping = {
        16: "Tree",
        20: "Shrub",
        5: "Shrub",
        11: "Shrub",
        37: "Shrub",
        31: "Shrub",
        33: "Herbaceous",
        43: "Herbaceous"
    }

    df_clean["PFT"] = df_clean["veg_class"].map(pft_mapping).astype("category")
    df_clean.drop(df_clean[df_clean["PFT"].isna()].index, inplace=True)

    # ---- Subsample ----
    n_samples = min(500_000, len(df_clean))
    df_sub = df_clean.sample(n=n_samples, random_state=42)

    # ---- Compute signal of reference before filtering PFT ----        # --> this is the special case of tree where n_veg = n_PFT = 1

    signal_all = df_sub[band_columns].values
    signal_ref_all = np.nanmean(signal_all, axis=0)

    s2 = signal_ref_all

    # filter for selected PFT
    df_clean = df_clean[df_clean["PFT"] == pft_class]   # --> the risk is that we filter down to 500_000 and the nagain by PFT --> we have less samples here.


    # ---- Bands ----
    band_columns = [c for c in df_sub.columns if c.startswith("band_")]

    # ---- Compute signal of interest ----
    s1 = np.nanmean(df_sub[df_sub["veg_class"] == veg_class_tag][band_columns].values, axis=0)

    # ---- Rolling ΔShape ----
    window = 13
    n_bands = len(band_columns)

    delta_G_rolling = []
    window_names = []

    for i in range(n_bands - window + 1):
        s1w = s1[i:i + window]
        s2w = s2[i:i + window]
        delta_G_val = delta_shape(s1w, s2w)
        delta_G_rolling.append(delta_G_val)
        window_names.append(f"{band_columns[i]}_{band_columns[i + window - 1]}")

    df_delta_G = pd.DataFrame([delta_G_rolling],
                              columns=[f"delta_G_{w}" for w in window_names])

    list_df_delta_G.append(df_delta_G)

    print(f'{date_tag} date is finished\n')

print(f'All dates are finished\n')


# --------------------------------------------------------------
# Plot — bands: 0-425
# --------------------------------------------------------------

# Convert band numbers → wavelength (nm)
wavelengths_nm = [380 + 5.02 * b for b in range(len(band_columns))]


# Compute ΔShape window centers (in wavelength nm)
window_centers_nm = [
    377 + 5.01 * ((i + i + window - 1) / 2)
    for i in range(len(band_columns) - window + 1)
]


# --- Plot ---
fig, ax1 = plt.subplots(figsize=(12, 5))

# include gray shaded region in certain wavelengths

# Add gray background regions for specific wavelength ranges
gray_regions = [
    (0, 450),       # start
    (1325, 1450),   # First region: 1350-1420 nm
    (1775, 2000),    # Second region: 1800-1975 nm
    (2450, 2510)    # end
]

#colors = plt.cm.tab20(np.linspace(0, 1, len(list_df_delta_G)))

colors = plt.cm.turbo(np.linspace(0, 1, len(list_df_delta_G)))

for idx, df_i in enumerate(list_df_delta_G):
    # ΔShape (left y-axis)
    ax1.plot(window_centers_nm,
             df_i.iloc[0].values[:len(window_centers_nm)],
             color= colors[idx],
             linestyle='-',
             linewidth = 2,
             label=f'{date_list[idx]}'
             )

ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("ΔShape (ΔG)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 0.05)  # fix y-scale for comparability

# Set x-ticks and grid lines every 50 nm
ax1.set_xlim(449, 1201)
x_ticks = np.arange(450, 1200, 50)  # Adjust range based on your wavelength range
ax1.set_xticks(x_ticks) # , rotation = 45
ax1.grid(True, which='major', alpha=0.3)

# Rotate x-ticks 45 degrees
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# Median reflectance (right y-axis)

# Create twin axis
ax2 = ax1.twinx()
ax2.patch.set_alpha(0)            # Make ax2 background invisible
#ax2.set_zorder(ax1.get_zorder() - 1)  # Push ax2 behind ax1

'''
ax2.plot(wavelengths_nm,
         s1,
         color='tab:red', linestyle='-', label=f'Mean refl. veg. type ({veg_class_tag})')
ax2.plot(wavelengths_nm,
         s2,
         color='tab:orange', linestyle='-', label=f'Mean Reflectance {pft_class}')
ax2.set_ylabel("Mean Reflectance", color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')
ax2.set_ylim(0, 0.55)  # fix y-scale for comparability
'''
'''
for region_start, region_end in gray_regions:
    ax1.axvspan(region_start, region_end, alpha=0.3, color='gray', label='Atmospheric absorption\n (or noisy bands)' if region_start == 1350 else "") 
'''
for region_start, region_end in gray_regions:
    ax1.axvspan(region_start, region_end,
                facecolor='white', edgecolor='gray', linewidth=0.5, alpha=1.0, zorder=100,       # <-- zorder forces spans on top
                label='Atmospheric absorption\n (or noisy bands)' if region_start == 1350 else "")

for region_start, region_end in gray_regions:
    ax2.axvspan(region_start, region_end,
                facecolor='white', edgecolor='gray', linewidth=0.5, alpha=1.0,  zorder=100,      # <-- zorder forces spans on top
                label='Atmospheric absorption\n (or noisy bands)' if region_start == 1350 else "")


# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()

leg = ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc= 'center right', frameon=True, ncol = 4)  #  loc = (0.45,0.65)
leg.set_zorder(200)                       # <-- ensure legend is on top
leg.get_frame().set_facecolor('white')    # optional
leg.get_frame().set_alpha(1.0)            # optional

# Title and limits
plt.title(f"ΔShape Rolling Window={window}, veg. type {veg_class_tag}_PFT_{pft_class}_temporal_zoom")

#plt.xlim(wavelengths_nm[0], wavelengths_nm[-1])  # -50, + 50

plt.tight_layout(rect=[0, 0.02, 1, 1])  # leave extra space at bottom (2%)

# Save figure
plt.savefig(os.path.join(dir_res_out, f'plot_delta_G_veg_type_{veg_class_tag}_PFT_{pft_class}_temporal_zoom.png'))
plt.close()


print(f'finished (all bands (unfiltered), {pft_class})\n')
print('plot finished and saved')





 # # # # # # template code here below