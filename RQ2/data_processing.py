import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ctgan import CTGAN
import logging

logger = logging.getLogger(__name__)

def load_dataset(file_path, dataset_type='csv'):
    """Load dataset based on file type."""
    try:
        if dataset_type == 'csv':
            data = pd.read_csv(file_path)
        elif dataset_type == 'excel':
            data = pd.read_excel(file_path, sheet_name=1)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        data = data.select_dtypes(include=[np.number]).dropna()
        logger.info(f"Loaded dataset from {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_and_train_ctgan(data, epochs=500):
    """Preprocess data and train CTGAN model."""
    try:
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        ctgan = CTGAN(epochs=epochs)
        ctgan.fit(data_scaled)
        logger.info("CTGAN model trained successfully")
        return scaler, ctgan, data_scaled
    except Exception as e:
        logger.error(f"Error in preprocessing or training CTGAN: {str(e)}")
        raise

def sample_data(data, size, scaler, ctgan, random_state=42):
    """Sample data and generate synthetic data."""
    try:
        sampled_data = data.sample(n=min(size, len(data)), random_state=random_state)
        real_scaled = scaler.transform(sampled_data)
        synthetic_scaled = np.asarray(ctgan.sample(len(real_scaled)))
        logger.info(f"Sampled {size} rows, generated synthetic data with shape {synthetic_scaled.shape}")
        return real_scaled, synthetic_scaled
    except Exception as e:
        logger.error(f"Error sampling data: {str(e)}")
        raise
