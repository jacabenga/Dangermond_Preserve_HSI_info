

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
file_date = os.listdir(dir_parquet_files)[10]        # list items in dir, index the first one [0]
parquet_file = os.path.join(dir_parquet_files, file_date)

# ----------------------------------------------------------------
# USER PARAMS
# ----------------------------------------------------------------

pft_class = 'Herbaceous' # 'Shrub'  # choose between: 'Tree' #
date_tag = '2022-05-11'  # choose between:  2022-05-11, ...


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
# Have a look
# ----------------------------------------------------------------

df = pd.read_parquet(parquet_file)
print(df.head)

'''
print(df[:][-3:-1])
print(len(df.columns))
print(df.columns)
'''

# Remove rows where any column has the sentinel value -9999.0
df_clean = df[~(df == -9999.0).any(axis=1)]

# ----------------------------------------------------------------
# Inspect veg_class
# ----------------------------------------------------------------

print(df_clean['veg_class'].unique())
print(' ')
print(len(df_clean['veg_class'].unique()))   # 51 classes

# ----------------------------------------------------------------
# Define PFT mapping --- for our (8) veg classes of interest
# ----------------------------------------------------------------

pft_mapping = {
    16: "Tree",
    20: "Shrub",
    5:  "Shrub",
    11: "Shrub",
    37: "Shrub",
    31: "Shrub",
    33: "Herbaceous",
    43: "Herbaceous"
}

# Add PFT column
df_clean['PFT'] = df_clean['veg_class'].map(pft_mapping).astype('category')

# Check
df_clean[['veg_class', 'PFT']].head()

# ----------------------------------------------------------------
# Keep only (8) classes of interest , remove the rest
# ----------------------------------------------------------------

df_clean.drop(df_clean[df_clean['PFT'].isna()].index, inplace = True)

# ----------------------------------------------------------------
# Filter further to keep only one single PFT of interest ('Shrub', 'Tree' or 'Herbaceous')
# ----------------------------------------------------------------

df_clean.drop(df_clean[df_clean['PFT'] != pft_class].index, inplace=True)

# have a look
print(df_clean.PFT.unique())
#

veg_class_tag =  43 #  33 # FOR HERBACEOUS   43: semi-natural herbaceous stands ; 33: non-native Grassland

'''
pft_mapping = {
    16: "Tree",
    20: "Shrub",
    5:  "Shrub",
    11: "Shrub",
    37: "Shrub",
    31: "Shrub",
    33: "Herbaceous",
    43: "Herbaceous"
}
'''

# ----------------------------------------------------------------
# Subsample to 0.5M pixels
# ----------------------------------------------------------------

# If cleaned DataFrame has fewer than 500_000 rows, this will take all available rows

n_samples = min(500_000, len(df_clean))                         # subsample to make things faster
#n_samples = max(500_000, len(df_clean))                        # if we don't want to subsample
df_subsampled = df_clean.sample(n=n_samples, random_state=42)   # random_state for reproducibility

# ----------------------------------------------------------------
# have a quick look at the median reflectance across bands
# ----------------------------------------------------------------
# Get band columns (assuming they're named 'band_1', 'band_2', etc.)
band_columns = [col for col in df_subsampled.columns if col.startswith('band_')]

# Calculate median reflectance for each band
median_reflectance = df_subsampled[band_columns].median()

# Plot median reflectance across bands
plt.figure(figsize=(12, 6))
plt.plot(median_reflectance.index, median_reflectance.values, linewidth=2) # marker='o'

plt.xlabel('Band Number')
plt.xticks(range(0, len(median_reflectance) + 1, 50))  # x-ticks every 50
plt.ylabel('Median Reflectance')
plt.title('Median Reflectance Across Bands')


plt.grid(True, alpha=0.6)
#plt.show()
plt.savefig(os.path.join(dir_res_out, f'median_reflectance_{date_tag}_{pft_class}.png'))
plt.close()

# Print some statistics
print("Median Reflectance Statistics:")
print(f"Number of bands: {len(band_columns)}")
print(f"Overall median: {median_reflectance.mean():.4f}")
print(f"Min median: {median_reflectance.min():.4f} (Band {median_reflectance.argmin() + 1})")
print(f"Max median: {median_reflectance.max():.4f} (Band {median_reflectance.argmax() + 1})")
print(f"Range: {median_reflectance.max() - median_reflectance.min():.4f}")


# ----------------------------------------------------------------
# define the signals
# ----------------------------------------------------------------

signal_all = df_subsampled[band_columns].values
signal_ref_all = np.nanmean(signal_all, axis=0)     # MEAN of that PFT

# ---------------------------------------------------------------------
# . global delta_G between  s1 (df_date_median_PFT_date_veg_class_i) and s2 (df_date_median_PFT_date)
# ---------------------------------------------------------------------

# Convert median DataFrames to 1D numpy arrays
'''
s1 = df_date_median_veg_class[band_columns].iloc[0].to_numpy()
s2 = df_date_mean[band_columns].iloc[0].to_numpy()
'''

s1 =  np.nanmean((df_subsampled[df_subsampled['veg_class']== veg_class_tag][band_columns].values), axis = 0)     # MEAN of that veg class
s2 = signal_ref_all

'''
# have a look
df_subsampled.columns
df_subsampled.head()
print(len(s1), len(s2))
#
'''


# Define rolling window parameters

window = 13
n_bands = len(band_columns)
delta_G_rolling = []  # store results
window_names = []


# Loop over rolling windows

for i in range(n_bands - window + 1):
    # get current window of bands
    s1_window = s1[i:i + window]
    s2_window = s2[i:i + window]

    # compute delta_G for this window
    delta_G_val = delta_shape(s1_window, s2_window)

    # store
    delta_G_rolling.append(delta_G_val)
    window_names.append(f"{band_columns[i]}_{band_columns[i + window - 1]}")

print('finished!')


# Convert to DataFrame

df_delta_G = pd.DataFrame([delta_G_rolling], columns=[f"delta_G_{w}" for w in window_names])
print(df_delta_G)


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


# ΔShape (left y-axis)
ax1.plot(window_centers_nm,
         df_delta_G.iloc[0].values[:len(window_centers_nm)],
         color='tab:blue', linestyle='-', label='ΔShape (ΔG)')

ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("ΔShape (ΔG)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 0.003)  # fix y-scale for comparability

# Set x-ticks and grid lines every 50 nm
x_ticks = np.arange(400, 2501, 50)  # Adjust range based on your wavelength range
ax1.set_xticks(x_ticks) # , rotation = 45
ax1.grid(True, which='major', alpha=0.3)

# Rotate x-ticks 45 degrees
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# Median reflectance (right y-axis)

# Create twin axis
ax2 = ax1.twinx()
ax2.patch.set_alpha(0)            # Make ax2 background invisible
#ax2.set_zorder(ax1.get_zorder() - 1)  # Push ax2 behind ax1


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

leg = ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc= (0.05,0.65), frameon=True)
leg.set_zorder(200)                       # <-- ensure legend is on top
leg.get_frame().set_facecolor('white')    # optional
leg.get_frame().set_alpha(1.0)            # optional

# Title and limits
plt.title(f"ΔShape Rolling Window={window}, shrub type {veg_class_tag} vs Mean Reflectance on {date_tag} (bands 0–425 / {wavelengths_nm[0]}–{wavelengths_nm[-1]} nm)")

plt.xlim(wavelengths_nm[0], wavelengths_nm[-1])  # -50, + 50

plt.tight_layout(rect=[0, 0.02, 1, 1])  # leave extra space at bottom (2%)

# Save figure
plt.savefig(os.path.join(dir_res_out, f'plot_delta_G_{date_tag}_veg_type_{veg_class_tag}_PFT_{pft_class}_bands0_425_unfiltered.png'))
plt.close()

print(f'finished (all bands (unfiltered), {pft_class})')


# # # template code here below

