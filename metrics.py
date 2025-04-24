import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, entropy

def compute_wasserstein_distance(real_df, synth_df, columns):
    """
    Compute the average Wasserstein distance for specified columns.
    """
    wd_list = []
    for col in columns:
        real_vals = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values
        if len(real_vals) > 0 and len(synth_vals) > 0:
            wd = wasserstein_distance(real_vals, synth_vals)
            wd_list.append(wd)
    return np.mean(wd_list) if wd_list else np.nan

def compute_ks_score(real_df, synth_df, columns):
    """
    Compute the average KS statistic for each column using ks_2samp.
    """
    ks_scores = []
    for col in columns:
        real_vals = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
        stat, _ = ks_2samp(real_vals, synth_vals)
        ks_scores.append(stat)
    return np.mean(ks_scores) if ks_scores else np.nan

def compute_kl_divergence(real_df, synth_df, columns, bins=50, epsilon=1e-10):
    """
    Compute the average KL divergence for each column:
    1. Compute histograms using np.histogram with the same bin edges
    2. Normalize histograms to probability distributions (add epsilon to avoid zeros)
    3. Compute KL divergence using entropy
    """
    kl_scores = []
    for col in columns:
        real_vals = real_df[col].dropna().values
        synth_vals = synth_df[col].dropna().values
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
        data_combined = np.concatenate([real_vals, synth_vals])
        bin_edges = np.histogram_bin_edges(data_combined, bins=bins)
        real_counts, _ = np.histogram(real_vals, bins=bin_edges)
        synth_counts, _ = np.histogram(synth_vals, bins=bin_edges)
        real_probs = real_counts / (real_counts.sum() + epsilon)
        synth_probs = synth_counts / (synth_counts.sum() + epsilon)
        real_probs = real_probs + epsilon
        synth_probs = synth_probs + epsilon
        real_probs = real_probs / real_probs.sum()
        synth_probs = synth_probs / synth_probs.sum()
        kl = entropy(real_probs, synth_probs)
        kl_scores.append(kl)
    return np.mean(kl_scores) if kl_scores else np.nan

class DistributionEvaluator:
    """Evaluate distribution similarity between real and synthetic data."""
    @staticmethod
    def compute_metrics(real_df, synth_df, features, bins=10):
        """
        Compute distribution similarity metrics for specified features.
        Args:
            real_df: Real DataFrame.
            synth_df: Synthetic DataFrame.
            features: List of feature columns to evaluate.
            bins: Number of histogram bins for KL divergence.
        Returns:
            Dictionary with average Wasserstein, KS, and KL metrics.
        """
        metrics = {
            'Wasserstein': [],
            'KS': [],
            'KL': []
        }
        for feat in features:
            real = real_df[feat].dropna().values
            synth = synth_df[feat].dropna().values
            if len(real) == 0 or len(synth) == 0:
                continue
            # Wasserstein distance
            metrics['Wasserstein'].append(wasserstein_distance(real, synth))
            # KS statistic
            metrics['KS'].append(ks_2samp(real, synth)[0])
            # KL divergence
            hist_real = np.histogram(real, bins=bins, density=True)[0] + 1e-10
            hist_synth = np.histogram(synth, bins=bins, density=True)[0] + 1e-10
            metrics['KL'].append(entropy(hist_real, hist_synth))
        return {k: np.mean(v) for k, v in metrics.items()}