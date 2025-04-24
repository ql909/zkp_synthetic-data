
import pandas as pd

# 
adult_df = pd.read_csv('/content/adult.csv')
loan_df = pd.read_excel('/content/Bank_Personal_Loan_Modelling.xlsx', sheet_name=1)
law_df = pd.read_csv('/content/post_processing_law.csv')

# 
datasets = {
    "Adult": adult_df,
    "Loan": loan_df,
    "Post Processing Law": law_df
}

# 
for name, df in datasets.items():
    print("=" * 50)
    print(f"Dataset: {name}")
    print(f"Shape: {df.shape}")  

    # 
    print("\nColumns and Data Types:")
    print(df.dtypes)

    # 
    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # 
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))

    # 
    print("\nUnique Values (for categorical features):")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")

    #
    print("\nNumerical Feature Ranges:")
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"  {col}: min = {min_val}, max = {max_val}")

    print("=" * 50, "\n")
