import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from .metrics import compute_wasserstein_distance, compute_ks_score, compute_kl_divergence

def run_feature_selection_attack(df_processed, target_col, generators, ml_models, attack_ratios):
    """
    Run feature selection attack:
    - Select top features based on Random Forest importance
    - Train generator on reduced feature set
    - Evaluate synthetic data quality and ML model performance
    """
    # Compute feature importance using Random Forest (excluding target column)
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_names = X.columns.tolist()
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    sorted_features = [feat for feat, imp in feature_importance]

    print(f"Total features (after one-hot): {len(sorted_features)}")

    # Iterate over attack ratios
    for ratio in attack_ratios:
        num_features_to_keep = max(1, int(ratio * len(sorted_features)))
        selected_features = sorted_features[:num_features_to_keep]
        print(f"\n[Attack Ratio: {ratio}] -> Keeping {num_features_to_keep} features")

        # Construct attacked dataset: keep only selected features and target
        attacked_df = df_processed[selected_features + [target_col]].copy()

        # Split data: 80% for generator training, 20% as held-out real test set
        train_real, test_real = train_test_split(attacked_df, test_size=0.2, random_state=42,
                                                 stratify=attacked_df[target_col])

        # Iterate over generators
        for gen_name, gen_func in generators.items():
            print(f"\n  >> Using generator: {gen_name}")
            generator = gen_func()
            try:
                # Train generator (all columns treated as variables)
                generator.fit(train_real)
                # Generate synthetic data with same number of rows as training set
                synth_data = generator.sample(len(train_real))
            except Exception as e:
                print(f"     Error in generator {gen_name}: {e}")
                continue

            # Compute distribution similarity metrics
            avg_wd = compute_wasserstein_distance(train_real, synth_data, selected_features)
            ks_score = compute_ks_score(train_real, synth_data, selected_features)
            kl_score = compute_kl_divergence(train_real, synth_data, selected_features)

            print(f"    [Synthetic Data Quality Metrics]")
            print(f"    Average Wasserstein Distance: {avg_wd:.4f}")
            print(f"    KS Score: {ks_score:.4f}")
            print(f"    KL Divergence: {kl_score:.4f}")

            # ML model evaluation: train on synthetic, test on real held-out
            X_synth = synth_data[selected_features]
            y_synth = synth_data[target_col]
            X_test_real = test_real[selected_features]
            y_test_real = test_real[target_col]

            print("    [ML Model Evaluation] - Train on synthetic, test on real held-out")
            for ml_name, ml_model in ml_models.items():
                model = clone(ml_model)
                try:
                    model.fit(X_synth, y_synth)
                    y_pred = model.predict(X_test_real)
                    acc = accuracy_score(y_test_real, y_pred)
                    print(f"      {ml_name:20s} Accuracy: {acc:.4f}")
                except Exception as e:
                    print(f"      {ml_name:20s} encountered an error: {e}")