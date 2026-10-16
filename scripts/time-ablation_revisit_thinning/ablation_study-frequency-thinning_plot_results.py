

# ablation_study-frequency-thinning_plot_results.py

# date is: 25.11.2025


# we plot the results of the frequency thinning experiment.


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter

# ------------------------------------------------------------
# Load all summaries
# ------------------------------------------------------------
dir_summaries = r"E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\KNN_ET_frequency_thinning\dfs_results_summaries"

files = glob.glob(os.path.join(dir_summaries, "*.csv"))
print(f"Found {len(files)} summary files")

dfs = [pd.read_csv(f) for f in files]
print("Loaded all CSVs")

# ------------------------------------------------------------
# Define the frequency order (x-axis)
# MUST match your scenario names
# ------------------------------------------------------------
frequency_weeks = {
    "1-week" : 1,
    "2-weeks": 2,
    "3-weeks": 3,
    "4-weeks": 4,
    "6-weeks": 6,
    "8-weeks": 8,
    "10-weeks": 10,
    "12-weeks": 12
}

ordered_scenarios = list(frequency_weeks.keys())
x = [frequency_weeks[s] for s in ordered_scenarios]

# ------------------------------------------------------------
# Collect accuracy values across all iterations
# accuracy_matrix shape: (n_iterations, n_scenarios)
# ------------------------------------------------------------

acc_matrix = []

for df in dfs:
    # ensure correct scenario order
    df_sorted = df.set_index("scenario").loc[ordered_scenarios]
    acc_matrix.append(df_sorted["accuracy"].values)

acc_matrix = np.array(acc_matrix)

# Mean & std
acc_mean = acc_matrix.mean(axis=0)

acc_mean = acc_mean / acc_mean.max() * 100

acc_std  = acc_matrix.std(axis=0)

print("Mean accuracy per frequency:", acc_mean)
print("Std accuracy per frequency:", acc_std)

# ------------------------------------------------------------
# Plot: mean accuracy ± std band
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

# Mean accuracy line
ax.plot(
    x, acc_mean,
    linewidth=2,
    color="black", alpha=0.8,
    label="Mean Accuracy"
)

# STD shading
ax.fill_between(
    x, acc_mean - acc_std, acc_mean + acc_std,
    color="gray", alpha=0.3,
    label="±1 SD"
)

# Labels and title
ax.set_xlabel("Revisit period (weeks)", fontsize=12)
ax.set_ylabel("Acc. (%) relative to 1 week revisit time", fontsize=12)
ax.set_title("KNN Frequency Thinning — Accuracy vs Revisit Period", fontsize=14)

# X-axis ticks
ax.set_xticks(x)

# Y-axis limits
ax.set_ylim(80, 101)

# ---------- major grid ----------
ax.grid(True, alpha=0.3)

# ---------- minor ticks every 1 unit (no labels) ----------
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_formatter(NullFormatter())

# ---------- minor grid ----------
ax.grid(which="minor", linestyle="--", alpha=0.2)

# Legend
ax.legend()

plt.tight_layout()

# Save
out_fig = os.path.join(dir_summaries, "plot_accuracy_vs_frequency.png")
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure:", out_fig)