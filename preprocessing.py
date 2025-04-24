import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, ds_name, target_col):
    """
    Preprocess the dataset:
    - Drop rows with missing target values
    - Drop 'ID' and 'ZIP Code' for loan dataset if present
    - Encode string target variable
    - Apply one-hot encoding to non-numeric features
    Returns processed DataFrame with features and target.
    """
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col]).copy()

    # For loan dataset, drop 'ID' and 'ZIP Code' columns if they exist
    if ds_name == 'loan':
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])
        if 'ZIP Code' in df.columns:
            df = df.drop(columns=['ZIP Code'])

    # Encode target variable if it is a string
    if df[target_col].dtype == 'object':
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])

    # Separate features and target, apply one-hot encoding
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X)
    y = df[target_col]
    df_processed = pd.concat([X, y], axis=1)

    return df_processed