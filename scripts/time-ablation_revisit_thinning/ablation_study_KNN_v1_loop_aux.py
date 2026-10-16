

# ablation_study_KNN_v1_loop_aux.py

# here we just loop over N time, so we get results for random selection of dates, in the time-ablation experiment.

# like this, we then plot mean across these random (N=30) executions, as mean \pm std.

for run in range(0,31):

    print(run)

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
    results_df.to_csv(os.path.join(dir_res_out, f'df_results_KNN_data-ablation_v{run}.csv'),index= True)



