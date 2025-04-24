from scipy.stats import wasserstein_distance, ks_2samp, entropy
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_wd(real, synthetic):
    """Compute Wasserstein Distance."""
    return wasserstein_distance(real, synthetic)

def compute_ks(real, synthetic):
    """Compute Kolmogorov-Smirnov statistic."""
    return ks_2samp(real, synthetic).statistic

def compute_kld(real, synthetic):
    """Compute Kullback-Leibler Divergence."""
    real_hist = np.histogram(real, bins=50, density=True)[0]
    synthetic_hist = np.histogram(synthetic, bins=50, density=True)[0]
    return entropy(np.clip(real_hist, 1e-10, None), np.clip(synthetic_hist, 1e-10, None))

def compute_dcr(real, synthetic):
    """Compute Distance to Closest Record."""
    return np.mean(np.min(euclidean_distances(real, synthetic), axis=1))

def compute_nndr(real, synthetic):
    """Compute Nearest Neighbor Distance Ratio."""
    real_dist = euclidean_distances(real)
    synth_dist = euclidean_distances(synthetic)
    return np.mean(np.mean(real_dist, axis=0) / (np.mean(synth_dist, axis=0) + 1e-10))

def compute_ims(real, synthetic):
    """Compute Identity Matching Score."""
    return sum(1 for r in real if any(np.allclose(r, s) for s in synthetic[:400])) / len(real)

def compute_anonymity_score(real, synthetic):
    """Calculate anonymity score."""
    uniqueness = 1 - compute_ims(real, synthetic)
    reid_risk = np.mean(np.abs(real - synthetic)) / np.std(real) if np.std(real) != 0 else 0
    linkage_risk = compute_ks(real.flatten(), synthetic.flatten())
    return 0.4 * uniqueness + 0.3 * reid_risk + 0.3 * linkage_risk

def compute_metrics(real_data, synthetic_data):
    """Compute all metrics for verification."""
    try:
        real_flat, synth_flat = real_data.flatten(), synthetic_data.flatten()
        metrics = {}
        for metric, func in [('wd', compute_wd), ('ks', compute_ks), ('kld', compute_kld),
                            ('dcr', compute_dcr), ('nndr', compute_nndr), ('ims', compute_ims),
                            ('anonymity', compute_anonymity_score)]:
            metrics[metric] = func(real_flat if metric in ['wd', 'ks', 'kld'] else real_data,
                                  synth_flat if metric in ['wd', 'ks', 'kld'] else synthetic_data)
        logger.info(f"Computed metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise