


# date is: 27.11.2025


# this code works well

# "shrub_v4.2"  -> this is an adaptation of the case "herbaceous", which works fine, but this time for the shrub classes



'''
    per-veg-community MI spectra with mean ± SD shading over dates. Essentially:

    For each wavelength (or sliding window center), compute MI for all 13 dates.

    Then compute the mean and standard deviation across dates for that wavelength.

    Plot the spectrum with shaded SD.
'''

# date: 27.11.2025

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression

# ---------------------------------------------------------
# DIRECTORIES
# ---------------------------------------------------------
dir_kernels = r'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\mutual_information_KLPD_and_kernels\shrubs\stored_kernel_excitation_vectors'
dir_klpd    = r'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\mutual_information_KLPD_and_kernels\shrubs\stored_delta_G_13_vectors'
dir_out     = r'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\mutual_information_KLPD_and_kernels\shrubs\MI_results\spectra_MI_over_time'
os.makedirs(dir_out, exist_ok=True)

# ---------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------
window_size = 26
n_neighbors = 5
date_list = ["2022-02-24","2022-02-28","2022-03-08","2022-03-16","2022-03-22",
             "2022-04-05","2022-04-12","2022-04-20","2022-04-29",
             "2022-05-03","2022-05-11","2022-05-17","2022-05-29"]

# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def compute_mutual_information(X, Y, n_neighbors=5):
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    return mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=0)[0]

def sliding_window_mi(X, Y, window_size=13, n_neighbors=5):
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

# ---------------------------------------------------------
# INITIALIZE STORAGE  (5 shrub classes)
# ---------------------------------------------------------

veg_classes = [5, 11, 20, 31, 37]
mi_mean_over_time = {5: [], 11: [], 20: [], 31: [], 37: []} # sliding-window MI for mean kernel
mi_var_over_time  = {5: [], 11: [], 20: [], 31: [], 37: []}  # sliding-window MI for variance kernel

# Initialize noise storage
mi_noise_over_time = {5: [], 11: [], 20: [], 31: [], 37: []}



# ---------------------------------------------------------
# MAIN LOOP OVER DATES AND VEG_CLASSES
# ---------------------------------------------------------

for d, date_tag in enumerate(date_list):
    print("Processing date:", date_tag)

    # Load ΔG_13
    klpd_pkl = [f for f in os.listdir(dir_klpd) if f.endswith(".pkl")][d]
    with open(os.path.join(dir_klpd, klpd_pkl), "rb") as f:
        all_delta_G = pickle.load(f)

    # Load kernel excitation
    kernel_pkl = [f for f in os.listdir(dir_kernels) if f.endswith(".pkl")][d]
    with open(os.path.join(dir_kernels, kernel_pkl), "rb") as f:
        data_k = pickle.load(f)

    mean_per_class       = data_k["mean_per_class"]
    inter_class_var_norm = data_k["inter_class_var_norm"]

    # Wavelength axis
    n_bands = len(list(all_delta_G.values())[0])
    window_centers_nm = np.array([
        377.8 + 5.01 * ((i + (i + window_size - 1)) / 2)
        for i in range(n_bands - window_size + 1)
    ])


    for veg_class in veg_classes:
        X = np.array(all_delta_G[veg_class])
        Y_mean = mean_per_class[veg_class].mean(axis=0)
        Y_var  = inter_class_var_norm

        # Compute sliding-window MI
        mi_mean = sliding_window_mi(X, Y_mean, window_size, n_neighbors)
        mi_var  = sliding_window_mi(X, Y_var, window_size, n_neighbors)
        mi_noise = sliding_window_mi(X, np.random.randn(len(X)), window_size, n_neighbors)

        # Store MI spectra
        mi_mean_over_time[veg_class].append(mi_mean)
        mi_var_over_time[veg_class].append(mi_var)
        mi_noise_over_time[veg_class].append(mi_noise)

print('MI spectra stored for all dates. \n')

# ---------------------------------------------------------
# COMPUTE MEAN ± SD AND PLOT
# ---------------------------------------------------------

for veg_class in veg_classes:
    mi_mean_array  = np.array(mi_mean_over_time[veg_class])
    mi_var_array   = np.array(mi_var_over_time[veg_class])
    mi_noise_array = np.array(mi_noise_over_time[veg_class])

    mi_mean_avg  = mi_mean_array.mean(axis=0)
    mi_mean_sd   = mi_mean_array.std(axis=0)

    mi_var_avg   = mi_var_array.mean(axis=0)
    mi_var_sd    = mi_var_array.std(axis=0)

    mi_noise_avg = mi_noise_array.mean(axis=0)

    # Common x settings
    xticks_major = np.arange(400, 2501, 100)
    xticks_minor = np.arange(400, 2501, 50)

    # --- Plot MI(deltaG, mean kernel) ---
    plt.figure(figsize=(12,6))
    plt.fill_between(window_centers_nm, mi_mean_avg - mi_mean_sd, mi_mean_avg + mi_mean_sd,
                     color='orange', alpha=0.3)
    plt.plot(window_centers_nm, mi_mean_avg, color='orange', linewidth=2,
             label='MI(deltaG, mean kernel)')
    plt.plot(window_centers_nm, mi_noise_avg, linestyle='--', color='black', linewidth=1.5,
             label='MI(X, noise)')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mutual Information")
    plt.title(f"Veg community {veg_class}: MI(deltaG, mean kernel)")

    # GRID: major (100 nm) solid, minor (50 nm) dashed
    plt.xticks(xticks_major)
    plt.gca().set_xticks(xticks_minor, minor=True)
    plt.grid(True, which='major', alpha=0.5)
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)

    plt.xlim(window_centers_nm[0], window_centers_nm[-1])
    # ⭐ LEGEND UPPER LEFT
    plt.legend(loc= 'upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_out, f"MI_mean_kernel_class{veg_class}.png"), dpi=300)
    plt.close()


    # --- Plot MI(deltaG, variance kernel) ---
    plt.figure(figsize=(12,6))
    plt.fill_between(window_centers_nm, mi_var_avg - mi_var_sd, mi_var_avg + mi_var_sd,
                     color='green', alpha=0.3)
    plt.plot(window_centers_nm, mi_var_avg, color='green', linewidth=2,
             label='MI(deltaG, variance kernel)')
    plt.plot(window_centers_nm, mi_noise_avg, linestyle='--', color='black', linewidth=1.5,
             label='MI(X, noise)')

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mutual Information")
    plt.title(f"Veg community {veg_class}: MI(deltaG, variance kernel)")

    # GRID: major (100 nm) solid, minor (50 nm) dashed
    plt.xticks(xticks_major)
    plt.gca().set_xticks(xticks_minor, minor=True)
    plt.grid(True, which='major', alpha=0.5)
    plt.grid(True, which='minor', linestyle='--', alpha=0.3)

    plt.xlim(window_centers_nm[0], window_centers_nm[-1])
    # ⭐ LEGEND UPPER LEFT
    plt.legend(loc= 'upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_out, f"MI_variance_kernel_class{veg_class}.png"), dpi=300)
    plt.close()




# # # # test code here below
