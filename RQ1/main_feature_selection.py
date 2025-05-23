from data_loader import load_datasets, DATASETS
from preprocessing import preprocess_data
from generators import get_generators
from models import get_ml_models
from feature_selection_attack import run_feature_selection_attack

# Define attack ratios
attack_ratios = [0.1, 0.3, 0.5]

def main():
    """
    Main function to run feature selection attack experiments.
    """
    # Load datasets
    datasets = load_datasets()
    generators = get_generators()
    ml_models = get_ml_models()

    # Process each dataset
    for ds_name, target_col in DATASETS.items():
        print("\n==========================")
        print(f"Dataset: {ds_name}")
        print("==========================")

        # Preprocess data
        df = datasets[ds_name]
        df_processed = preprocess_data(df, ds_name, target_col)
        print("Total features (after one-hot):", df_processed.shape[1] - 1)

        # Run feature selection attack
        run_feature_selection_attack(df_processed, target_col, generators, ml_models, attack_ratios)

if __name__ == "__main__":
    main()
