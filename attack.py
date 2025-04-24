import numpy as np
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

def compute_statistical_summary(data):
    """Compute statistical summary for dataset verification."""
    return {
        "mean": np.mean(data, axis=0).tolist(),
        "std": np.std(data, axis=0).tolist(),
        "quantiles": np.quantile(data, [0.25, 0.5, 0.75], axis=0).tolist()
    }

def generate_dataset_hash(data, stats, nonce=""):
    """Generate hash including statistical summary."""
    stats_str = json.dumps(stats, sort_keys=True)
    return hashlib.sha256((data.tobytes() + stats_str.encode() + nonce.encode())).hexdigest()

def simulate_data_source_attack(real_data, ctgan, correct_hash, correct_stats):
    """Simulate data source attack with incorrect dataset."""
    try:
        wrong_data = np.random.uniform(low=-10, high=10, size=real_data.shape)
        synthetic = wrong_data + np.random.normal(0, 0.1, wrong_data.shape)
        stats = compute_statistical_summary(wrong_data)
        logger.info("Simulated data source attack with incorrect dataset")
        return np.asarray(synthetic), correct_hash, stats  # Use correct_hash to trigger stats mismatch
    except Exception as e:
        logger.error(f"Error in data source attack: {str(e)}")
        raise

def simulate_quality_fraud_attack(real_data, ctgan):
    """Simulate quality fraud attack."""
    try:
        synthetic = ctgan.sample(len(real_data))
        synthetic = np.asarray(synthetic) + np.random.normal(0, 10, synthetic.shape)
        stats = compute_statistical_summary(real_data)
        return synthetic, generate_dataset_hash(real_data, stats)
    except Exception as e:
        logger.error(f"Error in quality fraud attack: {str(e)}")
        raise

def simulate_privacy_leak_attack(real_data, ctgan, leak_ratio=0.5):
    """Simulate privacy leak attack."""
    try:
        synthetic = ctgan.sample(len(real_data))
        synthetic = np.asarray(synthetic)
        mask = np.random.rand(len(real_data)) < leak_ratio
        synthetic[mask] = real_data[mask]
        stats = compute_statistical_summary(real_data)
        return synthetic, generate_dataset_hash(real_data, stats)
    except Exception as e:
        logger.error(f"Error in privacy leak attack: {str(e)}")
        raise

def simulate_distribution_tamper_attack(real_data, ctgan):
    """Simulate distribution tamper attack."""
    try:
        synthetic = ctgan.sample(len(real_data))
        synthetic = np.asarray(synthetic) * 2
        stats = compute_statistical_summary(real_data)
        return synthetic, generate_dataset_hash(real_data, stats)
    except Exception as e:
        logger.error(f"Error in distribution tamper attack: {str(e)}")
        raise