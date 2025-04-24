import pandas as pd

# Define dataset paths (adjust according to actual paths)
DATA_PATHS = {
    'adult': '/content/adult.csv',
    'law': '/content/post_processing_law.csv',
    'loan': '/content/Bank_Personal_Loan_Modelling.xlsx'
}

# Define datasets and their target columns
DATASETS = {
    'adult': ('income>50K'),
    'law': ('first_pf'),
    'loan': ('Personal Loan')
}

def load_datasets():
    """
    Load datasets from specified paths.
    Returns a dictionary with dataset names mapped to their DataFrames.
    """
    datasets = {}
    for ds_name in DATASETS:
        if ds_name == 'loan':
            datasets[ds_name] = pd.read_excel(DATA_PATHS[ds_name], sheet_name=1)
        else:
            datasets[ds_name] = pd.read_csv(DATA_PATHS[ds_name])
    return datasets