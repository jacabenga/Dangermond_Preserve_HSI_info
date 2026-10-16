

# date: 14.10.2025

'''

# saves as pickle files the delta_G_13 vectors

# same as before, but here we modified the code to include all 5 shrub vegetation classes in the combined plot.

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

import pickle


# ----------------------------------------------------------------
# USER PARAMS
# ----------------------------------------------------------------

pft_class = 'Herbaceous' # 'Shrub'  # choose between: 'Tree' #

date_list = ["2022-02-24","2022-02-28",
         "2022-03-08", "2022-03-16", "2022-03-22",
        "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
        "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29" ]                      # already done ,  ,



#date_tag = '2022-05-17'

for d in np.arange(len(date_list)):

    date_tag = date_list[d]
    print(d,', ',date_tag)
    print(' ')
    # ----------------------------------------------------------------
    #  OUTPUT DIRECTORY
    # ----------------------------------------------------------------

    #dir_res_out = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\results'
    dir_res_out = fr'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\mutual_information_KLPD_and_kernels\herbaceous\stored_delta_G_13_vectors'

    os.makedirs(dir_res_out, exist_ok=True)

    # ----------------------------------------------------------------
    # Load parquet
    # ----------------------------------------------------------------

    dir_parquet_files = r'D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\mosaics_labeled\df_parquet_streamed'
    file_date = os.listdir(dir_parquet_files)[d]        # list items in dir, index the first one [0]
    parquet_file = os.path.join(dir_parquet_files, file_date)

    print('parque_file selected is: ', parquet_file, '\n')

    df = pd.read_parquet(parquet_file)

    # ----------------------------------------------------------------
    # Have a look
    # ----------------------------------------------------------------

    print(df.head)
    '''
    print(len(df.columns))
    print(df.columns)
    '''


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
    # Remove rows where any column has the sentinel value -9999.0
    # ----------------------------------------------------------------

    df_clean = df[~(df == -9999.0).any(axis=1)]

    print('df_cleaned \n')

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

    veg_class_tag = 33 # 43          # FOR HERBACEOUS



    # ----------------------------------------------------------------
    # Subsample to 0.5M pixels
    # ----------------------------------------------------------------

    # If cleaned DataFrame has fewer than 500k rows, this will take all available rows
    n_samples = min(500_000, len(df_clean))
    #n_samples = max(500_000, len(df_clean))                        # if we don't want to subsample
    df_subsampled = df_clean.sample(n=n_samples, random_state=42)   # random_state for reproducibility



    # ----------------------------------------------------------------
    # define columns of interest
    # ----------------------------------------------------------------

    # Get band columns (assuming they're named 'band_1', 'band_2', etc.)
    band_columns = [col for col in df_subsampled.columns if col.startswith('band_')]


    # ----------------------------------------------------------------
    # have a quick look at the median reflectance across bands
    # ----------------------------------------------------------------

    '''
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
    '''


    # ----------------------------------------------------------------
    # define the signals
    # ----------------------------------------------------------------

    signal_all = df_subsampled[band_columns].values
    signal_ref_all = np.nanmean(signal_all, axis=0)     # MEAN of that PFT

    # ---------------------------------------------------------------------
    # . global delta_G between  s1 (df_date_median_PFT_date_veg_class_i) and s2 (df_date_median_PFT_date)
    # ---------------------------------------------------------------------
    '''
    # Convert median DataFrames to 1D numpy arrays
    s1 = df_date_median_veg_class[band_columns].iloc[0].to_numpy()
    s2 = df_date_mean[band_columns].iloc[0].to_numpy()
    '''

    s1 =  np.nanmean((df_subsampled[df_subsampled['veg_class']== veg_class_tag][band_columns].values), axis = 0)     # MEAN of that veg_class
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




    # # # # # # # # # Now we can plot delta_G of all 5 classes of shrubs

    # ----------------------------------------------------------------
    # step-0 : define some parameters
    # ----------------------------------------------------------------

    signal_all = df_subsampled[band_columns].values
    signal_ref_all = np.nanmean(signal_all, axis=0)     # MEAN of that PFT
    window = 13
    n_bands = len(band_columns)

    # Convert band numbers → wavelength (nm)
    wavelengths_nm = [380 + 5.02 * b for b in range(len(band_columns))]

    # Compute ΔShape window centers (in wavelength nm)
    window_centers_nm = [
        377 + 5.01 * ((i + i + window - 1) / 2)
        for i in range(len(band_columns) - window + 1)
    ]

    # Add gray background regions for specific wavelength ranges
    gray_regions = [
        (0, 450),       # start
        (1325, 1450),   # First region: 1325-1450nm
        (1775, 2000),    # Second region: 1775-2000 nm
        (2450, 2510)    # end
    ]



    # ----------------------------------------------------------------
    # step-1
    # ----------------------------------------------------------------

    #shrub_classes = [5, 11, 20, 31, 37]
    herbaceous_classes = [33, 43]

    # step-1.2 verify how many shrub classes remain in the df

    print(df_subsampled['veg_class'].value_counts())

    # ----------------------------------------------------------------
    # step-2
    # ----------------------------------------------------------------

    def compute_delta_G_for_class(df_subsampled, veg_class_tag, band_columns, window):
        # s1 = mean reflectance of this vegetation class
        s1 = np.nanmean(
            df_subsampled[df_subsampled["veg_class"] == veg_class_tag][band_columns].values,
            axis=0
        )

        # s2 = mean reflectance of all shrubs (already computed above)
        s2 = signal_ref_all

        n_bands = len(band_columns)
        delta_G_rolling = []

        for i in range(n_bands - window + 1):
            delta_G_rolling.append(
                delta_shape(s1[i:i + window], s2[i:i + window])
            )

        return np.array(delta_G_rolling)

    # ----------------------------------------------------------------
    # step-3
    # ----------------------------------------------------------------

    all_delta_G = {}   # dictionary: veg_class_tag -> delta_G array

    for cls in herbaceous_classes:
        print(f"Processing herbaceous class: {cls}")
        delta_G = compute_delta_G_for_class(
            df_subsampled=df_subsampled,
            veg_class_tag=cls,
            band_columns=band_columns,
            window=window
        )
        all_delta_G[cls] = delta_G


    # ----------------------------------------------------------------
    # checkpoint - save files
    # ----------------------------------------------------------------


    save_path = os.path.join(dir_res_out, fr"delta_G_{date_tag}_{pft_class}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(all_delta_G, f)

    print("Saved delta_G dictionary to:", save_path)


    # ----------------------------------------------------------------
    # plot 4: plot all
    # ----------------------------------------------------------------

    colors = ["red", "blue", "green", "orange", "purple"]

    fig, ax = plt.subplots(figsize=(12,6))

    for i, cls in enumerate(herbaceous_classes):
        ax.plot(
            window_centers_nm,
            all_delta_G[cls],
            label=f"Herbaceous class {cls}",
            linewidth=2,
            alpha=0.8,
            color=colors[i]
        )

    # Same formatting as before
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("ΔShape (ΔG)")
    ax.set_ylim(-0.0005, 0.021)

    ax.set_xticks(np.arange(400, 2501, 50))
    ax.set_xlim(380, 2500)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)

    # Gray regions (optional)
    for region_start, region_end in gray_regions:
        ax.axvspan(region_start, region_end, facecolor="gray", alpha=0.2)

    ax.legend()
    plt.title(f"ΔShape for all herbaceous classes — {date_tag}")

    plt.tight_layout()
    plt.savefig(os.path.join(dir_res_out, rf"all_herbaceous_deltaG_{date_tag}_v2.png"), dpi=300)
    plt.close()

    print('(delta_G_13) fig saved in: ', dir_res_out)



    # ----------------------------------------------------------------
    # FINAL MEMORY CLEANUP (per date)
    # ----------------------------------------------------------------


    del df, df_clean, df_subsampled
    del signal_all, signal_ref_all, s1, s2
    del delta_G_rolling, df_delta_G
    del all_delta_G, delta_G

    # Delete intermediate function variables if they exist
    try:
        del s1_window, s2_window
    except:
        pass

    # Cleanup large containers
    import gc
    gc.collect()

    print(f"✔ Memory cleaned up for {date_tag}\n")

