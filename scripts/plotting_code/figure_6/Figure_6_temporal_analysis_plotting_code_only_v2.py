


# Figure_6_temporal_analysis_plotting_code_only_v2.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullFormatter

# ------------------------------------------------------------
# INPUT DIRECTORIES
# ------------------------------------------------------------
dir_freq = r"E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\KNN_ET_frequency_thinning\dfs_results_summaries"

dir_ablation = r"E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\KNN_ET_date-ablation\retake_25.11.2025\dfs_results_summary"

dir_out = r'C:\Users\xnf878\PycharmProjects\Proj_multipurpose_1\submission-ready-figures\Figure_6\figures_out\time_ablation_frequency_thinning_right_panels'

# ------------------------------------------------------------
# FIGURE SETUP (2 vertical panels)
# ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 11), sharex=False)

# ============================================================
# PANEL 1 — DATE ABLATION
# ============================================================
records = []

for fname in os.listdir(dir_ablation):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(dir_ablation, fname))

    # 🔧 FIX 1: drop index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 🔧 FIX 2: skip broken files
    if "n_train_dates" not in df.columns:
        print(f"⚠️ Skipping file (missing column): {fname}")
        continue

    if "accuracy" not in df.columns:
        print(f"⚠️ Skipping file (missing accuracy): {fname}")
        continue

    records.extend(df[["n_train_dates", "accuracy"]].to_dict("records"))

df_all = pd.DataFrame(records)

summary = df_all.groupby("n_train_dates")["accuracy"].agg(["mean", "std"]).sort_index()

x = summary.index.values
y = summary["mean"].values
y = (y / y.max()) * 100

y_std = summary["std"].values
y_std = (y_std / summary["mean"].values.max()) * 100

# Plot
ax1.plot(x, y, color="black", linewidth=2, alpha=0.7, label="Mean accuracy")

ax1.fill_between(
    x, y - y_std, y + y_std,
    color="gray", alpha=0.3, label="± 1 SD"
)

ax1.set_title("D — Date Ablation | Accuracy vs Number of Training Dates",
              fontsize=15, fontweight='bold', loc='left')

ax1.tick_params(axis='both', labelsize=11)

ax1.set_ylabel("Accuracy (%) relative to 13 dates", fontsize=13)
ax1.set_xticks(x)
ax1.set_ylim(75, 101)

# Grid styling
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_minor_formatter(NullFormatter())
ax1.grid(which="minor", linestyle="--", alpha=0.2)

ax1.legend(fontsize=13, loc='lower right')

# ============================================================
# PANEL 2 — FREQUENCY THINNING
# ============================================================

files = glob.glob(os.path.join(dir_freq, "*.csv"))
dfs = [pd.read_csv(f) for f in files]

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
x2 = [frequency_weeks[s] for s in ordered_scenarios]

acc_matrix = []

for df in dfs:
    df_sorted = df.set_index("scenario").loc[ordered_scenarios]
    acc_matrix.append(df_sorted["accuracy"].values)

acc_matrix = np.array(acc_matrix)

acc_mean = acc_matrix.mean(axis=0)
acc_mean = acc_mean / acc_mean.max() * 100
acc_std  = acc_matrix.std(axis=0)

# Plot
ax2.plot(
    x2, acc_mean,
    linewidth=2,
    color="black", alpha=0.8,
    label="Mean accuracy"
)

ax2.fill_between(
    x2, acc_mean - acc_std, acc_mean + acc_std,
    color="gray", alpha=0.3,
    #label="±1 SD"
)

ax2.set_title("E — Frequency Thinning | Accuracy vs Revisit Period",
              fontsize=15, fontweight='bold', loc='left')

ax2.set_xlabel("Revisit period (weeks)", fontsize=13)
ax2.set_ylabel("Accuracy (%) relative to 1-week revisit", fontsize=13)

ax2.tick_params(axis='both', labelsize=11)

ax2.set_xticks(x2)
ax2.set_ylim(80, 101)

# Grid styling
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_minor_locator(MultipleLocator(1))
ax2.yaxis.set_minor_formatter(NullFormatter())
ax2.grid(which="minor", linestyle="--", alpha=0.2)

ax2.legend(fontsize=13, loc='lower right')

# ------------------------------------------------------------
# FINAL LAYOUT
# ------------------------------------------------------------
plt.tight_layout()

out_path = os.path.join(dir_out, "Figure_6_right_panels_combined_ablation_frequency_v0.png")
plt.tight_layout(pad=1.2)
#fig.subplots_adjust(hspace=0.35)
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("✅ Combined figure saved:", out_path)