

# date is : 26 Nov 2025

# here we do this:

'''

# "..._V1.py" ... first attempt.

# ..."_herbaceous_"    -> focus on the 2 herbaceous classes we have

# we calculate the mutual information (MI) between delta_G_13 (KLPD_shape) and either,

        # - mean kernel bank excitation
        # - variance of excitations


# X: delta_G_13   --> we borrow this metric from  -->  "KL'_DIV_v7_within_PFTs_shrubs_v.21.11.2025_FIGURE_several_classes.py"

# Y: kernel excitation     --> we borrow this metric from  "Ankit_RCF_on_hyperspectral_v5_loop_no_RF_v.20.11.2025_FIGURE.py"


# --> There is no universal upper limit for mutual information.

    Rule-of-thumb interpretation for MI in sliding windows of size (13, 26):

| MI value      | Interpretation (continuous signals, 13-point window)  |
| ------------- | ----------------------------------------------------- |
| **0.00–0.05** | Essentially no relationship (independence).           |
| **0.05–0.15** | Very weak shared structure.                           |
| **0.15–0.30** | Weak but reliable dependence.                         |
| **0.30–0.50** | Clear nonlinear or mixed dependence.                  |
| **0.50–0.80** | Strong shared information.                            |
| **> 0.80**    | Very strong or near-deterministic local relationship. |

Mutual information (MI) does not have a fixed upper bound, so an MI value like 0.8 can be “large” or “small” depending on:

        - the entropy of the signals,

        - the noise,

        - the window size,

        - the estimator used,

        - and whether your variables are continuous or discrete.


'''

date_str = '2022-05-29'

#-----------------------------------------------------------
# import libraries
#-----------------------------------------------------------

import numpy as np
from sklearn.feature_selection import mutual_info_regression

#-----------------------------------------------------------
# define directories
#-----------------------------------------------------------

#dir_res_out = fr'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\mutual_information_KLPD_and_kernels\herbaceous\herbaceous_veg_class_{veg_class_target}'



#-----------------------------------------------------------
# auxiliary functions
#-----------------------------------------------------------


def compute_mutual_information(X, Y, n_neighbors=5):
    """Compute MI between two 1D continuous vectors using Kraskov estimator."""
    X = np.asarray(X).reshape(-1, 1)
    Y = np.asarray(Y)
    return mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=0)[0]


def sliding_window_mi(X, Y, window_size=13, n_neighbors=5):
    """
    Computes sliding-window mutual information between X and Y.

    Parameters
    ----------
    X, Y : array-like of shape (n_samples,)
        Input vectors of equal length.
    window_size : int
        Size of the sliding window (default 13).
    n_neighbors : int
        Number of neighbors for MI estimator.

    Returns
    -------
    mi_values : np.array
        Array of MI values for each window (length = len(X) - window_size + 1)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)

    mi_values = []

    for start in range(n - window_size + 1):
        end = start + window_size

        Xw = X[start:end]
        Yw = Y[start:end]

        mi = compute_mutual_information(Xw, Yw, n_neighbors=n_neighbors)
        mi_values.append(mi)

    return np.array(mi_values)


# ---------------------------
# # Plug-in vectors  X: delta_G
# ---------------------------

for q in range(0,2):

    print(q)

    # -----------------------------------------------------------
    # define directories
    # -----------------------------------------------------------

    # dir_res_out = fr'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\mutual_information_KLPD_and_kernels\herbaceous\herbaceous_veg_class_{veg_class_target}'



    veg_classes_klpd = list(all_delta_G.keys())    #  this dictionary (i.e. all_delta_G) comes from -->  "KL'_DIV_v7_within_PFTs_shrubs_v.21.11.2025_FIGURE_several_classes.py"
    veg_class_target = veg_classes_klpd[q]

    print('veg_class_target is: ', veg_class_target)

    X = all_delta_G[veg_class_target]

    # ---------------------------
    # # Plug-in vectors  Y_1: mean_kernel_excitation ; Y_2: variance of kernel excitation
    # ---------------------------

    #   both Y_1 and Y_2 come from --> "Ankit_RCF_on_hyperspectral_v5_loop_no_RF_v.20.11.2025_FIGURE.py"

    veg_classes_kernel = list(mean_per_class.keys())
    veg_class_target = veg_classes_kernel[q]

    Y_1 = mean_per_class[veg_class_target].mean(dim=0)
    Y_2 = inter_class_var_norm.numpy()

    #Y_2 = np.abs(inter_class_var_norm.numpy())          # or inter_class_var_norm.numpy()  #

    # ----------
    # have a quick look
    # ----------
    '''
    # have a look
    
    print(len(X))   # -> delta_G_13
    
    print(len(mean_per_class[cls].mean(dim=0)))     # -> mean of kernel bank excitation
    print(len(mean_per_class))
    print(len(Y_1))
    
    print(len(inter_class_var_norm.numpy()))        # -> inter-class variance of kernel bank excitation
    print(len(Y_2))
    
    '''
    # ---------------------------
    # Mutual information (MI) for the entire series
    # ---------------------------

    print('the MI along the entire spectrum (window_size = 13) between the 2 methods is:\n', compute_mutual_information(X, Y_1, n_neighbors=21))   # 5)


    # ---------------------------
    # Implementation of mutual information; MI(X, Y_1)
    # ---------------------------

    # Compute sliding-window MI
    mi_series_mean = sliding_window_mi(X, Y_1, window_size=13)

    print("MI values shape:", mi_series_mean.shape)
    print("First 10 MI values:", mi_series_mean[:10])


    # --------------------------------------------------------------
    # Map band indices (0–424) to wavelengths (nm)
    # --------------------------------------------------------------

    # L1: wavelength per band (your calibration)
    wavelengths_nm = np.array([377.8 + 5.01 * b for b in range(425)])

    # ΔShape window length
    window = 13

    # Compute ΔShape window center wavelengths (nm)
    # A window starting at index i spans [i, i+12]; center = (i + i+12)/2
    window_centers_nm = np.array([
        377.8 + 5.01 * ((i + (i + window - 1)) / 2)
        for i in range(425 - window + 1)
    ])

    print("len of wavelengths_nm:", len(wavelengths_nm))
    print(' ')
    print("len of window_centers_nm:", len(window_centers_nm))


    # -----------
    # generate x_linspace for mi_series_mean
    # -----------

    x_linspace = window_centers_nm[6:-6]
    len(x_linspace)

    # -----------------------------------------------------------
    # Plot X, Y_1, and MI in aligned subplots
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # Compute MI(X, noise) as a null distribution reference
    # -----------------------------------------------------------
    '''
    GOOD IDEA: Compare to MI(X, noise)  --> Noise should give MI close to 0.0–0.05.
    '''

    noise = np.random.randn(len(X))
    mi_series_noise = sliding_window_mi(X, noise, window_size=13)


    # -----------------------------------------------------------
    # Plot X, Y_1, and MI in aligned subplots
    # -----------------------------------------------------------

    mean_or_variance = 'mean'

    plt.figure(figsize=(12, 10))

    # Get the overall x-range that covers all data
    all_x_data = np.concatenate([window_centers_nm, window_centers_nm, x_linspace])
    x_min = np.min(all_x_data)
    x_max = np.max(all_x_data)

    # Create a function to set consistent grid
    def set_consistent_grid():
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
        plt.xlim(x_min, x_max)

    # ---- 1. X ----
    plt.subplot(3, 1, 1)
    plt.plot(window_centers_nm, X, linewidth=1.5)
    plt.title("X : delta_G_13")
    plt.ylabel("X value")
    set_consistent_grid()

    # ---- 2. Y_1 (mean kernel excitation) ----
    plt.subplot(3, 1, 2)
    plt.plot(window_centers_nm, Y_1, linewidth=1.5)
    plt.title("Y₁ : Mean Kernel Excitation")
    plt.ylabel("Y₁ value")
    set_consistent_grid()

    # ---- 3. MI(X, Y₁) + MI(X, noise) ----
    plt.subplot(3, 1, 3)
    plt.plot(x_linspace, mi_series_mean, linewidth=1.5, label="MI(X, Y₁)")
    plt.plot(x_linspace, mi_series_noise, linewidth=1.5, linestyle="--", label="MI(X, noise)")

    plt.title("Sliding-Window Mutual Information : MI(X, Y₁) vs MI(X, noise)")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("MI (nats)")

    plt.legend()
    set_consistent_grid()

    plt.tight_layout()

    plt.savefig(os.path.join(
        dir_res_out,
        f"mutual_information_delta_G_13_kernel-bank_{mean_or_variance}_{date_str}_{veg_class_target}.png"
    ), dpi=300)

    plt.close()


    # ---------------------------
    # Implementation of mutual information; MI(X, Y_2)
    # ---------------------------

    # Compute sliding-window MI for Y_2
    mi_series_var = sliding_window_mi(X, Y_2, window_size=13)

    print("MI values shape:", mi_series_var.shape)
    print("First 10 MI values:", mi_series_var[:10])

    # ---------------------------
    # Compute MI(X, noise) baseline
    # ---------------------------
    noise = np.random.randn(len(X))
    mi_series_noise = sliding_window_mi(X, noise, window_size=13)



    mean_or_variance = 'variance'

    plt.figure(figsize=(12, 10))

    # Get the overall x-range that covers all data
    all_x_data = np.concatenate([window_centers_nm, window_centers_nm, x_linspace])
    x_min = np.min(all_x_data)
    x_max = np.max(all_x_data)

    # Create a function to set consistent grid
    def set_consistent_grid():
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
        plt.xlim(x_min, x_max)

    # ---- 1. X ----
    plt.subplot(3, 1, 1)
    plt.plot(window_centers_nm, X, linewidth=1.5)
    plt.title("X : delta_G_13")
    plt.ylabel("X value")
    set_consistent_grid()

    # ---- 2. Y_2 (inter-class variance kernel excitation) ----
    plt.subplot(3, 1, 2)
    plt.plot(window_centers_nm, Y_2, linewidth=1.5)
    plt.title("Y₂ : Inter-Class Variance of Kernel Excitation")
    plt.ylabel("Y₂ value")
    set_consistent_grid()

    # ---- 3. MI(X, Y₂) + MI(X, noise) ----
    plt.subplot(3, 1, 3)
    plt.plot(x_linspace, mi_series_var, linewidth=1.5, label="MI(X, Y₂)")
    plt.plot(x_linspace, mi_series_noise, linewidth=1.5, linestyle="--", label="MI(X, noise)")

    plt.title("Sliding-Window Mutual Information : MI(X, Y₂) vs MI(X, noise)")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("MI (nats)")
    plt.legend()
    set_consistent_grid()

    plt.tight_layout()

    plt.savefig(os.path.join(
        dir_res_out,
        f"mutual_information_delta_G_13_kernel-bank_{mean_or_variance}_{date_str}_{veg_class_target}.png"
    ), dpi=300)

    plt.close()


    print('mutual information plots: finished.')


# # # # test code here below
