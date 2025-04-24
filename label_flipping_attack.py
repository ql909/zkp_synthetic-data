import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from .metrics import compute_wasserstein_distance, compute_ks_score, compute_kl_divergence

def run_label_flipping_attack(df_processed, target_col, generators, ml_models, attack_ratios):
    """
    Run label flipping attack:
    - Flip a portion of target labels in the training set
    - Train generator on attacked training set
    - Evaluate synthetic data quality and ML model performance
    """
    # Split dataset: 80% training, 20% testing (test set remains clean)
    train_clean, test_clean = train_test_split(df_processed, test_size=0.2, random_state=42,
                                               stratify=df_processed[target_col])

    # Iterate over attack ratios
    for attack_ratio in attack_ratios:
        # Create a copy of the training set for attack
        train_attacked = train_clean.copy()
        num_samples = train_attacked.shape[0]
        num_to_flip = int(attack_ratio * num_samples)
        # Randomly select samples and flip target labels (assuming labels are 0 and 1)
        flip_indices = np.random.choice(train_attacked.index, size=num_to_flip, replace=False)
        train_attacked.loc[flip_indices, target_col] = 1 - train_attacked.loc[flip_indices, target_col]

        print(f"\n[Attack Ratio (Label Flipping): {attack_ratio}] -> Flipped {num_to_flip} labels out of {num_samples}")

        # Iterate over generators
        for gen_name, gen_func in generators.items():
            print(f"\n  >> Using generator: {gen_name}")
            generator = gen_func()
            try:
                generator.fit(train_attacked)
                # Generate synthetic data with same number of rows as attacked training set
                synth_data = generator.sample(len(train_attacked))
            except Exception as e:
                print(f"     Error in generator {gen_name}: {e}")
                continue

            # Compute distribution similarity metrics (only for features, not target)
            feature_columns = [col for col in df_processed.columns if col != target_col]
            avg_wd = compute_wasserstein_distance(train_attacked, synth_data, feature_columns)
            ks_score = compute_ks_score(train_attacked, synth_data, feature_columns)
            kl_score = compute_kl_divergence(train_attacked, synth_data, feature_columns)

            print(f"    [Synthetic Data Quality Metrics]")
            print(f"    Average Wasserstein Distance: {avg_wd:.4f}")
            print(f"    KS Score: {ks_score:.4f}")
            print(f"    KL Divergence: {kl_score:.4f}")

            # ML model evaluation: train on synthetic, test on clean held-out test set
            X_synth = synth_data[feature_columns]
            y_synth = synth_data[target_col]
            X_test = test_clean[feature_columns]
            y_test = test_clean[target_col]

            print("    [ML Model Evaluation] - Train on synthetic, test on clean held-out")
            for ml_name, ml_model in ml_models.items():
                model = clone(ml_model)
                try:
                    model.fit(X_synth, y_synth)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    print(f"      {ml_name:20s} Accuracy: {acc:.4f}")
                except Exception as e:
                    print(f"      {ml_name:20s} encountered an error: {e}")