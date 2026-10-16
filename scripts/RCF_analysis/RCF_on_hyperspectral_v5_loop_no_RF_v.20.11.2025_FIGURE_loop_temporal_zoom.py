

# date is: 2.11.2025

# This is a “kitchen sinks” model, specifically a random convolutional feature extractor + RandomForest,
    #  which is an instance of the Random Kitchen Sinks idea adapted for spectral data.

'''
Concept and Purpose:

    Simulate a “shallow CNN” over the spectral axis to capture local spectral patterns (e.g. absorption features) — but with random, fixed weights instead of trained ones.
    So instead of learning filters, it uses randomly sampled spectral kernels.

'''

'''
_loop_temporal_zoom_ -> loop over dates in order to show evolution of mean of 1 veg class w.r.t. time 

...v.20.11.2025_FIGURE  -> prepared for manuscript

v5 ... --> one same combined plot of (i) intra-class mean and of (ii) inter-class variance.

v4 ... --> here we save two separate plots of (i) intra-class mean and of (ii) inter-class variance.

same as v1 ... --> but within a loop, to obtain results in 13 different dates.

    1.Loads your .parquet dataset from disk.

    2.Extracts features and labels.

    3.Uses HSI_RCF (empirical mode) to create random convolutional features.
        -> If mode='empirical', the filters are not purely random — they are drawn from real pixel spectra
            ->  (?): So each random kernel corresponds to a short spectral snippet from some pixel.
            
        -> self._normalize(patches) : applies zero-centering, normalization, and ZCA whitening — to decorrelate the random filters.
            -> That ensures filters are orthogonal and balanced, preventing redundancy.

    4.Splits into train/test (75/25).

    5.Trains a RandomForestClassifier.

    6.Prints accuracy and plots the confusion matrix.
'''


import os
import numpy as np

import pandas as pd
import pyarrow.dataset as ds

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import gc


# define mode for convolutional random kernel bank
mode_type = "empirical"                                         # "gaussian" # "gaussian"


# ---- OUTPUT DIRECTORY ----
dirout = fr'H:\LandCRAFT_year-2\communication and results\communication with Ankit\mode_{mode_type}'

# ---- DATES ----
data_dir = r"D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\mosaics_labeled\df_parquet_streamed"

dates = ["2022-02-24", "2022-02-28",
         "2022-03-08", "2022-03-16", "2022-03-22",
        "2022-04-05", "2022-04-12", "2022-04-20", "2022-04-29",
        "2022-05-03", "2022-05-11", "2022-05-17", "2022-05-29" ]                       # already done

# dates = ["2022-04-29"]  # if interested in only 1 target date


# ---- Class mapping ----
class_names = {
    5:  "Black sage scrub",
    11: "California sagebrush scrub",
    16: "Coast live oak woodland",
    20: "Coyote brush scrub",
    31: "Menzie's golden bush scrub",
    33: "Non-native grassland",
    37: "Purple sage scrub",
    43: "Semi-natural herbaceous stands"
}


# ---- Colorblind-safe palette and symbols ----
colors = [
    "#0072B2", "#E69F00", "#009E73", "#56B4E9",
    "#D55E00", "#CC79A7", "#F0E442", "#999999"
]

markers = ["o", "s", "^", "v", "D", "P", "X", "*"]



# ---- Random Convolutional Feature Extractor ----
class HSI_RCF(nn.Module):
    def __init__(self, in_channels=1, in_features=377, out_channels=4096, kernel_size=13,
                 bias=-1.0, avg_pool=True, seed=None, mode="empirical", dataset=None):   # "empirical", "gaussian"
        super().__init__()
        assert mode in ["empirical", "gaussian"]
        if mode == "empirical" and dataset is None:
            raise ValueError("dataset must be provided when mode='empirical'")
        assert out_channels % 2 == 0
        num_patches = out_channels // 2

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        self.avg_pool = avg_pool
        self.register_buffer("weights", torch.randn(num_patches, in_channels, kernel_size, generator=generator))
        self.register_buffer("biases", torch.zeros(num_patches) + bias)

        if mode == "empirical":
            num_channels, length = dataset[0]["data"].shape
            patches = np.zeros((num_patches, num_channels, kernel_size), dtype=np.float32)
            idxs = torch.randint(0, len(dataset), (num_patches,), generator=generator).numpy()
            xs = torch.randint(0, max(1, length - kernel_size), (num_patches,), generator=generator).numpy()
            for i in range(num_patches):
                hsi = dataset[idxs[i]]["data"].numpy()
                patches[i] = hsi[:, xs[i]: xs[i] + kernel_size]
            print("Assigning RCF empirically")
            patches = self._normalize(patches)
            self.weights = torch.tensor(patches)

    def _normalize(self, patches, min_divisor=1e-8, zca_bias=0.001):
        n_patches = patches.shape[0]
        orig_shape = patches.shape
        patches = patches.reshape(n_patches, -1)
        patches -= np.mean(patches, axis=1, keepdims=True)
        patch_norms = np.linalg.norm(patches, axis=1)
        patch_norms[patch_norms < min_divisor] = 1
        patches /= patch_norms[:, np.newaxis]
        cov = 1.0 / n_patches * patches.T.dot(patches)
        E, V = np.linalg.eig(cov)
        E += zca_bias
        inv_sqrt = np.diag(1.0 / np.sqrt(E))
        zca = V.dot(inv_sqrt).dot(V.T)
        patches = patches.dot(zca).dot(zca.T)
        print("Performed ZCA transformation!")
        return patches.reshape(orig_shape).astype(np.float32)

    def forward(self, x):
        x1a = F.relu(F.conv1d(x, self.weights, bias=self.biases))
        x1b = F.relu(-F.conv1d(x, self.weights, bias=self.biases))
        if self.avg_pool:
            x1a = F.adaptive_avg_pool1d(x1a, 1)
            x1b = F.adaptive_avg_pool1d(x1b, 1)
        return torch.cat((x1a, x1b), dim=1)


# ---- PyTorch Dataset Wrapper ----
class HSIDataset(Dataset):
    def __init__(self, input_data, labels):
        self.input_data = torch.tensor(input_data).unsqueeze(1)
        self.labels = torch.tensor(labels).long()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {"data": self.input_data[idx], "label": self.labels[idx]}



# --------------------------------------------------------------
# Plot — bands: 0-425   --> translate this first to wavelengths
# --------------------------------------------------------------

# Convert band numbers → wavelength (nm)
wavelengths_nm = [377.8 + 5.01 * b for b in range(425)]

# Compute ΔShape window centers (in wavelength nm)
window = 13
window_centers_nm = [
    377 + 5.01 * ((i + i + window - 1) / 2)
    for i in range(425 - window + 1)
]

print('len of wavelength_window_centers_nm: ', len(window_centers_nm))


# ------------------------------------------------
# ---- data preparation before main loop ---------
# ------------------------------------------------

kernel_excitation_mean_list = []   # will hold 1-D numpy arrays (L,)
date_tags = []

parquet_files = sorted(os.listdir(data_dir))  # ensure consistent ordering (optional)




# ------------------------------------------------
# ---- MAIN LOOP OVER DATES ----
# ------------------------------------------------




for date_str in dates:
    print("="*60)
    print(f"Processing dataset: {date_str}")
    print("="*60)

    file_path = os.path.join(data_dir, f"{date_str}.parquet")
    dataset = ds.dataset(file_path, format="parquet")

    #target_classes = list(class_names.keys())
    #target_classes = [list(class_names.keys())[i] for i in [5, -1]]  # only * HERBACEOUS * classes
    #target_classes = [list(class_names.keys())[i] for i in [0,1,3,4,6]]  # only * SHRUB * classes
    #target_classes = [list(class_names.keys())[i] for i in [2]]  # only * TREE * (1 only class)

   #target_classes = [list(class_names.keys())[i] for i in [-1]]  # only * HERBACEOUS * classes   # ONLY 1 to show evolution over time # -1: semi-natural herbaceous stands

    target_classes = [list(class_names.keys())[i] for i in [1]]  # only * SHRUB * classes   # ONLY 1 to show evolution over time # 1: California Sagescrub

    table = dataset.to_table(filter=(ds.field('veg_class').isin(target_classes)))
    df = table.to_pandas()
    print("Original shape:", df.shape)

    df = df[df['veg_class'].isin(target_classes)]
    if len(df) > 100_000:
        df = df.sample(n=100_000, random_state=42)
    print(f"After selecting {len(df)} samples")

    numeric_cols = [c for c in df.columns if c.startswith("band_")]
    X = df[numeric_cols].to_numpy(dtype=np.float32)
    y = df["veg_class"].to_numpy()
    del df

    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= 2]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]
    print(f"Filtered dataset: {len(y)} samples, {len(valid_classes)} valid classes")

    # ------------------------------------------------
    # ---- REDUCE X ONLY FOR ZCA WHITENING ----
    # ------------------------------------------------

    print("X shape before ZCA:", X.shape)
    print(' ')

    # Downscale for ZCA whitening only (keep full X for RCF)
    ZCA_sample_size = min(15_000, len(X))
    print(f"Using {ZCA_sample_size} samples for ZCA whitening")

    indices = np.random.choice(len(X), size=ZCA_sample_size, replace=False)
    X_small = X[indices]

    # Apply ZCA to the reduced sample
    print("Performing ZCA on reduced sample...")
    X_small_tensor = torch.tensor(X_small, dtype=torch.float32)
    zca_mean = X_small_tensor.mean(0, keepdim=True)
    X_centered = X_small_tensor - zca_mean

    # Covariance
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eps = 1e-5
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + eps))

    # Whitening matrix
    zca_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T

    print("ZCA whitening ready (computed from reduced sample).")

    # Save ZCA params for later use inside RCF
    zca_params = {
        "mean": zca_mean,
        "matrix": zca_matrix
    }

    # ------------------------------------------------
    # ---- CONSTRUCT RCF (using FULL X dataset) ----
    # ------------------------------------------------

    dt = HSIDataset(X, y)  # FULL DATASET, unchanged
    n_features = X.shape[1]
    del X_small_tensor, X_small, X_centered, cov, eigvals, eigvecs, D_inv_sqrt
    gc.collect()

    rcf = HSI_RCF(in_channels=1, in_features=n_features, out_channels=4096,
                  kernel_size=13, avg_pool=True, mode= mode_type, dataset=dt)  # "empirical" , "gaussian"

    batch_size = 256
    loader = DataLoader(dt, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcf = rcf.to(device)
    target_classes_unique = np.unique(y)

# from here


    # --- prepare ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rcf = rcf.to(device)

    target_classes_unique = np.unique(y)

    # Initialize accumulators (lazy init for tensor shapes)
    sum_per_class = {int(cls): None for cls in target_classes_unique}  # tensors on GPU
    count_per_class = {int(cls): 0 for cls in target_classes_unique}

    with torch.no_grad():
        for batch in loader:
            x = batch["data"].to(device)  # shape: [B, 1, L]
            y_batch = batch["label"].numpy()  # CPU labels

            # Compute activations on GPU
            x1a = F.relu(F.conv1d(x, rcf.weights, bias=rcf.biases))
            x1b = F.relu(-F.conv1d(x, rcf.weights, bias=rcf.biases))
            activations = torch.cat((x1a, x1b), dim=1)  # [B, C_act, L_out]

            # Accumulate class-wise sums on GPU
            for cls in target_classes_unique:
                mask = (y_batch == cls)
                if mask.any():
                    mask_idx = np.nonzero(mask)[0].tolist()
                    s = activations[mask_idx].sum(dim=0)

                    if sum_per_class[int(cls)] is None:
                        sum_per_class[int(cls)] = s.clone()
                    else:
                        sum_per_class[int(cls)] += s

                    count_per_class[int(cls)] += int(mask.sum())

    # ---- Compute per-class means (small CPU tensors) ----
    mean_per_class = {}
    for cls in target_classes_unique:
        cls = int(cls)
        mean_per_class[cls] = (sum_per_class[cls] / count_per_class[cls]).cpu()

    # Cleanup GPU buffers
    del activations, x1a, x1b
    torch.cuda.empty_cache()
    gc.collect()


# until here

    # ----  Compute mean per class ----
    #mean_per_class = {cls: sum_per_class[cls] / count_per_class[cls] for cls in target_classes_unique}

    # ---- Compute inter-class variance ----
    # Stack all class mean activations → tensor of shape [num_classes, num_kernels, num_bands]
    stacked_means = torch.stack(list(mean_per_class.values()), dim=0)  # [C, K, L]

    # Mean over kernels → average activation per class along spectral axis
    mean_over_kernels_per_class = stacked_means.mean(dim=1)  # [C, L]

    # compute mean across classes → 1-D (L,)
    mean_across_classes = mean_over_kernels_per_class.mean(dim=0)  # torch tensor shape [L]                #... --> ok, but there is only one class

    # move to CPU numpy (detach if on GPU), store
    kernel_excitation_mean_list.append(mean_across_classes.cpu().numpy())

    # keep the date string for legend
    date_tags.append(date_str)

    print("🎉 All datasets processed successfully!")

    # memory cleanup

    del mean_across_classes
    torch.cuda.empty_cache()
    gc.collect()

    print('memory cleaned')

    # Variance across classes at each band
    #inter_class_var = mean_over_kernels_per_class.var(dim=0)  # [L]

    # Normalize for better visualization
    #inter_class_var_norm = (inter_class_var - inter_class_var.mean()) / inter_class_var.std()


# # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # #

# the plot is out of the loop to plot mean kernel excitation of veg class over time



# ---- Plot intraclass MEAN over time----


fig, ax1 = plt.subplots(figsize=(12, 6))

# choose turbo colormap over the number of dates
colors = plt.cm.turbo(np.linspace(0, 1, len(kernel_excitation_mean_list)))


for k_np, dt, c in zip(kernel_excitation_mean_list, date_tags, colors):

    # normalize before plotting

    k_np_mean = k_np.mean()
    k_np = (k_np - k_np_mean) / k_np.std()
    # k_np is 1-D array length L
    ax1.plot(window_centers_nm, k_np, color=c, linewidth=1.8, alpha=0.9, label=dt)

ax1.set_xlabel("Wavelength (nm)")
ax1.set_ylabel("Kernel Bank Excitation (class mean)")

ax1.set_ylim(-1.3, 0.75)   # or whatever range you prefer
ax1.set_xlim(449, 1300)   # or whatever range you prefer

# decorations
x_ticks = np.arange(450, 1201, 50)
ax1.set_xticks(x_ticks)
ax1.grid(True, which='major', alpha=0.3)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

# legend: 4 columns (or change ncol)
leg = ax1.legend(title="Date", loc='upper center', bbox_to_anchor=(0.25, 0.8), ncol=4, fontsize=8, frameon=True)
leg.set_zorder(200)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_alpha(1.0)

plt.title("Mean kernel excitation (class mean) — all dates")
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

out_fname = os.path.join(dirout, "RCF_mean_excitation_SHRUB_veg_class_11california_all_dates_NORM_v1.png")
plt.savefig(out_fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved combined plot to {out_fname}")



 # # # # # test code below


