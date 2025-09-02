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

def preprocess_and_train_nflow(data, epochs=50):
    """Preprocess data and train NFLOW model."""
    try:
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        nflow = NFLOWSynthesizer(epochs=epochs, batch_size=256)
        categorical_cols = data.select_dtypes(include=['category']).columns.tolist()
        nflow.train(data, categorical_cols)
        logger.info("NFLOW model trained successfully")
        return scaler, nflow, data_scaled
    except Exception as e:
        logger.error(f"Error in preprocessing or training NFLOW: {str(e)}")
        raise

def preprocess_and_train_tvae(data, epochs=50):
    """Preprocess data and train TVAE model."""
    try:
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        tvae = TVAESynthesizer(epochs=epochs, batch_size=256)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        tvae.train(data, categorical_cols)
        logger.info("TVAE model trained successfully")
        return scaler, tvae, data_scaled
    except Exception as e:
        logger.error(f"Error in preprocessing or training TVAE: {str(e)}")
        raise

def sample_data(data, size, scaler, tvae, random_state=42):
    """Sample data and generate synthetic data."""
    try:
        sampled_data = data.sample(n=min(size, len(data)), random_state=random_state)
        real_scaled = scaler.transform(sampled_data)
        synthetic_data = tvae.generate(num_samples=len(real_scaled))
        synthetic_scaled = scaler.transform(synthetic_data)
        logger.info(f"Sampled {size} rows, generated synthetic data with shape {synthetic_scaled.shape}")
        return real_scaled, synthetic_scaled
    except Exception as e:
        logger.error(f"Error sampling data: {str(e)}")
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

def sample_data(data, size, scaler, nflow, random_state=42):
    """Sample data and generate synthetic data."""
    try:
        sampled_data = data.sample(n=min(size, len(data)), random_state=random_state)
        real_scaled = scaler.transform(sampled_data)
        synthetic_data = nflow.generate(num_samples=len(real_scaled))
        synthetic_scaled = scaler.transform(synthetic_data)
        logger.info(f"Sampled {size} rows, generated synthetic data with shape {synthetic_scaled.shape}")
        logger.info(f"Real data mean: {np.mean(real_scaled, axis=0)}")
        logger.info(f"Synthetic data mean: {np.mean(synthetic_scaled, axis=0)}")
        return real_scaled, synthetic_scaled
    except Exception as e:
        logger.error(f"Error sampling data: {str(e)}")
        raise
