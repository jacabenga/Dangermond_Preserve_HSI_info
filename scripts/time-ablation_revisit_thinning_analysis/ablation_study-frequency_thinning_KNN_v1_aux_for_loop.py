'''

ablation_study-frequency_thinning_KNN_v1_aux_for_loop.py


# date: 10.09.2025

# "frequency_thinning" here, we remove the dates systematically, so we simulate a longer revisit period.

# We do this: leave-one-date-out / incremental-date ablation study for KNN,
# with random sampling for the test set

# df, and other objects can be borrowed from "read_HDF5_minimal_KNN_classification_v3.py"
'''

'''
# # # # These are the scenarios

| Case | Approx. Revisit period | Desired #dates | Meaning                           |
| ---- | ---------------------- | -------------- | --------------------------------- |
| 1    | 1 week (baseline)      | 13             | keep all                          |
| 1b   | 2 weeks                | ~7             | keep every ≈2nd date              |
| 2    | 3 weeks                | 5              | keep every 2nd–3rd date           |
| 3    | 4 weeks                | 4              | keep every 3rd date               |
| 4    | 6 weeks                | 3              | keep every 4th–5th date           |
| 5    | 8 weeks                | 2              | keep every 6th date               |
| 6    | 10 weeks               | 2              | keep 2 dates only (further apart) |
| 7    | 12 weeks               | 2              | keep 2 dates only (max spacing)   |
'''

# IMPORT LIBRARIES

import os.path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

'''
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
'''

# # # define dirs

dir_res_out = r'E:\Dangermond_SHIFT_SSL_study\data_communication_v2\ML_results\KNN_ET_frequency_thinning\dfs_results_summaries'


# # # QUICKLY RELOAD 198K.parquet file

size_pixels = '198Kpixels'
dir_ml_tasks = r'D:\Jaime_C._Revenga_LandCRAFT_2025\Project_with_F.Schneider_S.Normand\data\ML_tasks'
df = pd.read_parquet(os.path.join(dir_ml_tasks, f"df_X_y_{size_pixels}.parquet"))
print('df loaded from file')
print(df.head())


# # # Function to sample dates, by stepping through the timeline

def frequency_thinning(dates, step, n_keep=None):
    """
    Select dates by stepping index positions:
    step = keep every `step` dates.
    If n_keep is provided, truncate or pad as needed.
    """
    selected = dates[::step]

    if n_keep is not None:
        if len(selected) > n_keep:
            selected = selected[:n_keep]
        elif len(selected) < n_keep:
            # pad by spreading evenly
            idx = np.linspace(0, len(dates)-1, n_keep, dtype=int)
            selected = dates[idx]

    return np.array(selected)

# # # # # # # # # # # # # # # # # # # # # # # #

# KNN classification Parameters
dates = df['date'].unique()
n_neighbors = 21  # best hyperparameter found so far
test_size = 0.2   # fraction of test data
random_seed = 42

# # # DICTIONARY with the frequency scenarios we want to consider

dates_sorted = np.sort(dates)    # ensure chronological order

thinning_scenarios = {
    "1-week":        frequency_thinning(dates_sorted, step=1),        # all 13
    "2-weeks":       frequency_thinning(dates_sorted, step=2),        # ~7 dates
    "3-weeks":       frequency_thinning(dates_sorted, step=3, n_keep=5),
    "4-weeks":       frequency_thinning(dates_sorted, step=3, n_keep=4),
    "6-weeks":       frequency_thinning(dates_sorted, step=4, n_keep=3),

    # The last three: choose 2 dates with maximum separation
    "8-weeks":       dates_sorted[[0, 6]],   # first + mid
    "10-weeks":      dates_sorted[[0, 8]],   # increased gap
    "12-weeks":      dates_sorted[[0, 12]],  # extreme gap (first + last)
}


# -----------------------------
# LOOP for Frequency thinning experiment: incrementally REMOVE/ADD DATA in training SYSTEMATICALLY, so we simulate longer revisit periods (e.g. of a satellite).
# -----------------------------

for run in range(0,31):

    print(run)
    print( )

    results_thinning = []

    for scenario, selected_dates in thinning_scenarios.items():
        print(f"\n=== Scenario: {scenario} — {len(selected_dates)} dates ===")
        print("Selected:", selected_dates)

        # Training data from selected dates
        train_df = df[df['date'].isin(selected_dates)]
        X_train = train_df.drop(columns=['veg_class','date']).values
        y_train = train_df['veg_class'].values

        # Test data: random 20% from ALL dates (same as your protocol)
        test_df = df.sample(frac=0.2, random_state=42)
        X_test = test_df.drop(columns=['veg_class','date']).values
        y_test = test_df['veg_class'].values

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=21, n_jobs=-1)
        knn.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = knn.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = knn.score(X_test_scaled, y_test)

        results_thinning.append({
            "scenario": scenario,
            "n_dates": len(selected_dates),
            "dates": selected_dates.tolist(),
            "train_samples": len(train_df),
            "f1_weighted": f1,
            "accuracy": acc
        })

        print(f"F1={f1:.3f}, Accuracy={acc:.3f}, Train samples={len(train_df)}")

    print('All is good now; all scenarios considered have been evaluated.')


    # -----------------------------
    # Convert to dataframe and save
    # -----------------------------

    results_thinning_df = pd.DataFrame(results_thinning)
    print(results_thinning_df)
    print(' ')

    results_thinning_df.to_csv(os.path.join(dir_res_out, f"KNN_frequency_thinning_results_v{run}.csv"), index=False)


    print('results have been saved to: ', dir_res_out)