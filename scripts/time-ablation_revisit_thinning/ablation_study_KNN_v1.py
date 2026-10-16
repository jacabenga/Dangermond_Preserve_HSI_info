
# ablation_study_KNN_v1.py

# date: 10.09.2025

# We do this: leave-one-date-out / incremental-date ablation study for KNN,
# with random sampling for the test set

# df, and other objects can be borrowed from "read_HDF5_minimal_KNN_classification_v3.py"

'''
How this works:
Incrementally train KNN on 1,2,...13 dates.
Test always on a random 20% sample from the full dataset.
Tracks F1-weighted and accuracy for each experiment.
You can later plot n_train_dates vs F1 to see how performance improves as you include more dates.
'''

import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score

import matplotlib.pyplot as plt

'''
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
'''




# # # QUICKLY RELOAD 198K.parquet file

dir_ml_tasks = r'D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\ML_tasks'
df = pd.read_parquet(os.path.join(dir_ml_tasks, f"df_X_y_{size_pixels}.parquet"))
print('df loaded from file')
print(df.head())


# # #


# Parameters
dates = df['date'].unique()
n_neighbors = 21  # best hyperparameter found so far
test_size = 0.2   # fraction of test data
random_seed = 42

# Storage for results
results = []

# -----------------------------
# Ablation study: incrementally REMOVE/ADD DATA in training (we remove both dates and instances, so both X_train grows AND increments in diversity of dates)
# -----------------------------

for n_dates in range(1, len(dates) + 1):

    # Sample n_dates for training (random combination of n_dates)
    train_dates = np.random.choice(dates, n_dates, replace=False)

    # Training data
    train_df = df[df['date'].isin(train_dates)]
    X_train = train_df.drop(columns=['veg_class' ,'date']).values
    y_train = train_df['veg_class'].values

    # Test data: random 20% from all 13 dates
    test_df = df.sample(frac=test_size, random_state=random_seed)
    X_test = test_df.drop(columns=['veg_class' ,'date']).values
    y_test = test_df['veg_class'].values

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    # Predict
    y_pred = knn.predict(X_test_scaled)

    # Metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = knn.score(X_test_scaled, y_test)

    results.append({
        'n_train_dates': n_dates,
        'train_dates': train_dates.tolist(),
        'f1_weighted': f1,
        'accuracy': acc,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    })

    print(f"Trained on {n_dates} date(s): F1={f1:.3f}, Accuracy={acc:.3f}")

'''
Now, we randomly pick n_dates from all 13 dates.
The training data (train_df) contains all samples from those selected dates.
So the training size increases as n_dates increases, 
because each date contributes its full number of samples.
'''


# -----------------------------
# Convert results to DataFrame
# -----------------------------
results_df = pd.DataFrame(results)
print(results_df)
dir_res_out = r'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\KNN_ET_date-ablation\retake_25.11.2025'
results_df.to_csv(os.path.join(dir_res_out,'df_results_KNN_data-ablation_v2.csv'),index= True)

# -----------------------------
# Plot n_dates vs F1
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(results_df['n_train_dates'], results_df['f1_weighted'], marker='o', linestyle='-')

plt.xticks(range(1, len(dates)+1))
plt.xlabel("Number of Training Dates")
plt.ylabel("Weighted F1 Score")
plt.title("KNN Ablation Study: #Training Dates vs F1 Score")
plt.grid(True)
os.makedirs(dir_res_out, exist_ok=True)
save_path = os.path.join(dir_res_out, "ablation_study_knn_v1.png")
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to {save_path}")
plt.close()  # Close the figure to free memory
#plt.show()


# -----------------------------
# Plot n_dates vs F1 &  Plot n_dates vs Accuracy
# -----------------------------


fig, ax1 = plt.subplots(figsize=(8,5))

# Primary y-axis: Weighted F1
color_f1 = 'tab:blue'
ax1.set_xlabel("Number of Training Dates")
ax1.set_ylabel("Weighted F1 Score", color=color_f1)
ax1.plot(results_df['n_train_dates'], results_df['f1_weighted'], marker='o', linestyle='-', color=color_f1, label='Weighted F1')
ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_xticks(range(1, len(dates)+1))
ax1.invert_xaxis()
ax1.set_title("KNN Ablation Study: #Training Dates vs Performance")
ax1.grid(True)

# Secondary y-axis: Accuracy
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color_acc = 'tab:red'
ax2.set_ylabel("Accuracy", color=color_acc)
ax2.plot(results_df['n_train_dates'], results_df['accuracy'], marker='s', linestyle='--', color=color_acc, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color_acc)

# Optionally, annotate Accuracy values on the line
for x, y in zip(results_df['n_train_dates'], results_df['accuracy']):
    ax2.text(x, y, f"{y:.2f}", color=color_acc, fontsize=8, ha='left', va='bottom')

# Legend
fig.tight_layout()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best') # upper left

# Save figure
os.makedirs(dir_res_out, exist_ok=True)
save_path = os.path.join(dir_res_out, "ablation_study_knn_with_accuracy_v1.png")
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to {save_path}")
plt.close()


# -----------------------------
# Date-ablation study: incrementally remove dates (so X_train is constant BUT it increments in diversity of dates)
# -----------------------------


# Parameters
n_neighbors = 21
test_size = 0.2
random_seed = 42
results = []
total_train_samples = 15_000  # total training samples, constant
dates = df['date'].unique()


np.random.seed(random_seed)


for n_dates in range(1, len(dates) + 1):
    # Randomly select n_dates
    train_dates = np.random.choice(dates, n_dates, replace=False)

    # Split total_train_samples evenly among chosen dates
    samples_per_date = total_train_samples // n_dates

    train_chunks = []
    for d in train_dates:
        date_df = df[df['date'] == d]
        # Sample min(samples_per_date, available samples) from this date
        sampled_df = date_df.sample(n=min(samples_per_date, len(date_df)), random_state=random_seed)
        train_chunks.append(sampled_df)

    train_df = pd.concat(train_chunks)
    X_train = train_df.drop(columns=['veg_class', 'date']).values
    y_train = train_df['veg_class'].values

    # Test data: random fraction of all dates (independent of training)
    test_df = df.sample(frac=test_size, random_state=random_seed)
    X_test = test_df.drop(columns=['veg_class', 'date']).values
    y_test = test_df['veg_class'].values

    # Scale features
    scaler = RobustScaler() # StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)

    # Predict
    y_pred = knn.predict(X_test_scaled)

    # Metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = knn.score(X_test_scaled, y_test)

    results.append({
        'n_train_dates': n_dates,
        'train_dates': train_dates.tolist(),
        'f1_weighted': f1,
        'accuracy': acc,
        'train_samples': len(train_df),
        'test_samples': len(test_df)
    })

    print(f"Trained on {n_dates} date(s) ({len(train_df)} samples): F1={f1:.3f}, Accuracy={acc:.3f}")

# Convert results to DataFrame
results_df_15K = pd.DataFrame(results)
#results_df_5K = pd.DataFrame(results)


# -----------------------------
# Plot n_dates vs F1 &  Plot n_dates vs Accuracy
# -----------------------------

#results_df = results_df_5k  # results_df_15k


fig, ax1 = plt.subplots(figsize=(8,5))

# Primary y-axis: Weighted F1
color_f1 = 'tab:blue'
ax1.set_xlabel("Number of Training Dates")
ax1.set_ylabel("Weighted F1 Score", color=color_f1)

ax1.plot(results_df_5K['n_train_dates'], results_df_5K['f1_weighted'], marker='o', linestyle='-', color=color_f1, label='Weighted F1 (5K pixels)')
ax1.plot(results_df_15K['n_train_dates'], results_df_15K['f1_weighted'], marker='o', linestyle='-', color=color_f1, label='Weighted F1 (15K pixels)')

ax1.tick_params(axis='y', labelcolor=color_f1)
ax1.set_xticks(range(1, len(dates)+1))
ax1.invert_xaxis()
ax1.set_title("KNN Date-Ablation Study: #Training Dates vs Performance (training size: 5K, 15K pixels)")
ax1.grid(True)

# Secondary y-axis: Accuracy
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color_acc = 'tab:red'
ax2.set_ylabel("Accuracy", color=color_acc)

ax2.plot(results_df_5K['n_train_dates'], results_df_5K['accuracy'], marker='s', linestyle='--', color=color_acc, label='Accuracy (5K pixels)')
ax2.plot(results_df_15K['n_train_dates'], results_df_15K['accuracy'], marker='s', linestyle='--', color=color_acc, label='Accuracy (15K pixels)')


ax2.tick_params(axis='y', labelcolor=color_acc)

# Optionally, annotate Accuracy values on the line  ---  5K
for x, y in zip(results_df_5K['n_train_dates'], results_df_5K['accuracy']):
    ax2.text(x, y, f"{y:.2f}", color=color_acc, fontsize=8, ha='left', va='bottom')

# Optionally, annotate Accuracy values on the line --- 15K
for x, y in zip(results_df_15K['n_train_dates'], results_df_15K['accuracy']):
    ax2.text(x, y, f"{y:.2f}", color=color_acc, fontsize=8, ha='left', va='bottom')

# Legend
fig.tight_layout()
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='best') # upper left

# Save figure
os.makedirs(dir_res_out, exist_ok=True)
save_path = os.path.join(dir_res_out, "date_ablation_study_knn_with_accuracy_5K_15K.png")
plt.savefig(save_path, dpi=300)
print(f"✅ Plot saved to {save_path}")
plt.close()



