

# ablation_study-frequency_thinning_KNN_v1_aux_for_loop.py

# date: 25.11.2025

# here we tak the results of KNN and plot them


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# Load results
# ---------------------------------------



dir_in = r'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\KNN_ET_date-ablation\retake_25.11.2025'

records = []

for fname in os.listdir(dir_in):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(dir_in, fname))

    # Assumes each CSV has multiple rows or one row of results
    for _, row in df.iterrows():
        records.append({
            "n_train_dates": row["n_train_dates"],
            "accuracy": row["accuracy"]
        })

df_all = pd.DataFrame(records)

print('all accuracy records are loaded')

# ---------------------------------------
# Compute mean and std grouped by number of training dates
# ---------------------------------------


summary = df_all.groupby("n_train_dates")["accuracy"].agg(["mean", "std"])
summary = summary.sort_index()

x = summary.index.values
y = summary["mean"].values
y = (summary["mean"].values / summary["mean"].values.max()) * 100         # as a percentage

y_std = summary["std"].values
y_std = (summary["std"].values / summary["mean"].values.max()) * 100  # as a percentage

# ---------------------------------------
# Plot: line + std band
# ---------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(x, y, color="black", linewidth=2, alpha = 0.6, label="Mean accuracy")

# STD band
ax.fill_between(
    x, y - y_std, y + y_std,
    color="gray", alpha=0.3, label="± 1 SD"
)

ax.set_xlabel("Number of training dates", fontsize=12)
ax.set_ylabel("Accuracy (%) relative to 13 dates", fontsize=12)
ax.set_title("KNN Date Ablation — Accuracy vs Number of Training Dates", fontsize=14)

# y-axis ticks every 2 units
#ymin, ymax = ax.get_ylim()
#ax.set_yticks(np.arange(np.floor(ymin), np.ceil(ymax) + 1, 5))
ax.set_yticks(np.arange(75, 125, 5))
ax.set_xticks(x)


# ----- minor ticks every 1 unit, but no labels -----
from matplotlib.ticker import MultipleLocator, NullFormatter

ax.yaxis.set_minor_locator(MultipleLocator(1))   # 1-unit spacing
ax.yaxis.set_minor_formatter(NullFormatter())    # hide labels

# minor grid
ax.grid(which="minor", linestyle="--", alpha=0.2)

ax.grid(True, alpha=0.3)

ax.legend()

plt.ylim(75, 101)

plt.tight_layout()


out_path = os.path.join(dir_in, 'plots/records_accuracy_KNN_summary_v-test.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Figure saved to: {out_path}')


# # # # test code here below


