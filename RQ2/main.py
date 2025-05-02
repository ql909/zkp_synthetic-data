import logging
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread
from data_processing import load_dataset, preprocess_and_train_ctgan, sample_data
from attacks import simulate_data_source_attack, simulate_quality_fraud_attack, simulate_privacy_leak_attack, simulate_distribution_tamper_attack, compute_statistical_summary, generate_dataset_hash
from zkp_verification import client, server, compute_dynamic_thresholds
from metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------
# Main Experiment for ctgan
# ----------------------
def perform_statistical_analysis(metrics_normal, metrics_attack, metric_name):
    """Perform statistical analysis between normal and attack metrics."""
    normal_values = [m[metric_name] for m in metrics_normal if metric_name in m]
    attack_values = [m[metric_name] for m in metrics_attack if metric_name in m]
    if len(normal_values) > 1 and len(attack_values) > 1:
        t_stat, p_value = ttest_ind(normal_values, attack_values, equal_var=False)
        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_normal": np.mean(normal_values),
            "mean_attack": np.mean(attack_values),
            "std_normal": np.std(normal_values),
            "std_attack": np.std(attack_values)
        }
    return {"t_stat": None, "p_value": None, "mean_normal": None, "mean_attack": None}

def main():
    datasets = {
        "Adult": ("/content/adult.csv", "csv"),
        "Loan": ("/content/Bank_Personal_Loan_Modelling.xlsx", "excel"),
        "Law": ("/content/post_processing_law.csv", "csv")
    }

    ctgan_models = {}
    scalers = {}
    for name, (file_path, file_type) in datasets.items():
        data = load_dataset(file_path, file_type)
        scaler, ctgan, _ = preprocess_and_train_ctgan(data)
        scalers[name] = scaler
        ctgan_models[name] = ctgan

    data_sizes = [500, 1000, 5000]
    base_thresholds = {'wd': 1.0, 'ks': 0.5, 'kld': 10.0, 'ims': 0.03, 'anonymity': 0.7}
    attack_types = [None, "data_source", "quality_fraud", "privacy_leak", "distribution_tamper"]
    results, all_metrics = [], defaultdict(list)

    for name, (file_path, file_type) in datasets.items():
        data = load_dataset(file_path, file_type)
        ctgan = ctgan_models[name]
        scaler = scalers[name]
        if ctgan is None:
            logger.error(f"CTGAN model for {name} is None")
            continue
        logger.info(f"\n{'='*50}\nProcessing dataset: {name}\n{'='*50}")

        # Compute dataset-specific base thresholds
        sampled_data = data.sample(n=min(500, len(data)), random_state=42)
        sampled_scaled = scaler.transform(sampled_data)
        synthetic_data = ctgan.sample(500).values
        wd_base = np.mean([wasserstein_distance(sampled_scaled.flatten(), synthetic_data.flatten()) for _ in range(5)])
        base_thresholds['wd'] = wd_base * 1.5
        logger.info(f"Dataset {name} base WD threshold: {base_thresholds['wd']:.4f}")

        for size in data_sizes:
            for attack in attack_types:
                real_scaled, _ = sample_data(data, size, scaler, ctgan)
                expected_stats = compute_statistical_summary(real_scaled)
                correct_hash = generate_dataset_hash(real_scaled, expected_stats)

                if attack == "data_source":
                    synthetic_scaled, dataset_hash, stats = simulate_data_source_attack(real_scaled, ctgan, correct_hash, expected_stats)
                elif attack == "quality_fraud":
                    synthetic_scaled, dataset_hash = simulate_quality_fraud_attack(real_scaled, ctgan)
                    stats = expected_stats
                elif attack == "privacy_leak":
                    synthetic_scaled, dataset_hash = simulate_privacy_leak_attack(real_scaled, ctgan, leak_ratio=0.3)
                    stats = expected_stats
                elif attack == "distribution_tamper":
                    synthetic_scaled, dataset_hash = simulate_distribution_tamper_attack(real_scaled, ctgan)
                    stats = expected_stats
                else:
                    synthetic_scaled = np.asarray(ctgan.sample(len(real_scaled)))
                    dataset_hash = correct_hash
                    stats = expected_stats
                thresholds = compute_dynamic_thresholds(size, base_thresholds)
                logger.info(f"Thresholds for size {size}: {thresholds}")

                q1, q2 = Queue(maxsize=10), Queue(maxsize=10)
                result_queue = Queue()
                threads = [
                    Thread(target=server, args=(q2, q1, correct_hash, expected_stats), daemon=True),
                    Thread(target=client, args=(q1, q2, real_scaled, synthetic_scaled, dataset_hash, dataset_hash, thresholds, stats, result_queue, attack), daemon=True),
                ]
                for t in threads: t.start()
                for t in threads: t.join(timeout=600)

                result = result_queue.get(timeout=120) if not result_queue.empty() else {
                    "result": "Timeout",
                    "violations": ["Timeout"],
                    "hash_status": "Fail",
                    "metrics": {}
                }
                zkp_status = result.get("result", "N/A")
                quality_status = "Pass" if not result.get("violations") else "Fail"
                hash_status = result.get("hash_status", "Fail")
                # Safely format metrics, handling 'N/A' strings
                wd_value = result.get('metrics', {}).get('wd', 'N/A')
                anonymity_value = result.get('metrics', {}).get('anonymity', 'N/A')
                wd_formatted = f"{wd_value:.4f}" if isinstance(wd_value, (int, float)) else wd_value
                anonymity_formatted = f"{anonymity_value:.4f}" if isinstance(anonymity_value, (int, float)) else anonymity_value

                results.append({
                    "Dataset": name,
                    "Size": size,
                    "Attack": attack or "None",
                    "ZKP Time(ms)": result.get("zkp_time", "N/A"),
                    "Total Time(ms)": result.get("total_time", "N/A"),
                    "ZKP Status": zkp_status,
                    "Quality Status": quality_status,
                    "Hash Status": hash_status,
                    "Violations": "\n".join(result.get("violations", [])) or "None",
                    "WD": wd_formatted,
                    "Anonymity": anonymity_formatted
                })
                all_metrics[(name, attack)].append(result.get("metrics", {}))

    # Statistical Analysis
    stat_results = []
    for name in datasets:
        for metric in ['wd', 'anonymity']:
            normal_metrics = all_metrics[(name, None)]
            for attack in attack_types[1:]:
                attack_metrics = all_metrics[(name, attack)]
                stat = perform_statistical_analysis(normal_metrics, attack_metrics, metric)
                stat_results.append({
                    "Dataset": name,
                    "Metric": metric,
                    "Attack": attack,
                    "T-Stat": stat["t_stat"],
                    "P-Value": stat["p_value"],
                    "Mean Normal": stat["mean_normal"],
                    "Mean Attack": stat["mean_attack"]
                })

    print("\nFinal Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))
    print("\nStatistical Analysis Results:")
    print(tabulate(stat_results, headers="keys", tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    main()

# ----------------------
# Main Experiment for tvae
# ----------------------
def perform_statistical_analysis(metrics_normal, metrics_attack, metric_name):
    """Perform statistical analysis between normal and attack metrics."""
    normal_values = [m[metric_name] for m in metrics_normal if metric_name in m]
    attack_values = [m[metric_name] for m in metrics_attack if metric_name in m]
    if len(normal_values) > 1 and len(attack_values) > 1:
        t_stat, p_value = ttest_ind(normal_values, attack_values, equal_var=False)
        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_normal": np.mean(normal_values),
            "mean_attack": np.mean(attack_values),
            "std_normal": np.std(normal_values),
            "std_attack": np.std(attack_values)
        }
    return {"t_stat": None, "p_value": None, "mean_normal": None, "mean_attack": None}

def main():
    datasets = {
        "Adult": ("/content/adult.csv", "csv"),
        "Loan": ("/content/Bank_Personal_Loan_Modelling.xlsx", "excel"),
        "Law": ("/content/post_processing_law.csv", "csv")
    }

    tvae_models = {}
    scalers = {}
    for name, (file_path, file_type) in datasets.items():
        data = load_dataset(file_path, file_type)
        scaler, tvae, _ = preprocess_and_train_tvae(data)
        scalers[name] = scaler
        tvae_models[name] = tvae

    data_sizes = [500, 1000, 5000]
    base_thresholds = {'wd': 1.0, 'ks': 0.5, 'kld': 10.0, 'ims': 0.03, 'anonymity': 0.7}
    attack_types = [None, "data_source", "quality_fraud", "privacy_leak", "distribution_tamper"]
    results, all_metrics = [], defaultdict(list)

    for name, (file_path, file_type) in datasets.items():
        data = load_dataset(file_path, file_type)
        tvae = tvae_models[name]
        scaler = scalers[name]
        if tvae is None:
            logger.error(f"TVAE model for {name} is None")
            continue
        logger.info(f"\n{'='*50}\nProcessing dataset: {name}\n{'='*50}")

        # Compute dataset-specific base thresholds
        sampled_data = data.sample(n=min(500, len(data)), random_state=42)
        sampled_scaled = scaler.transform(sampled_data)
        synthetic_data = tvae.generate(num_samples=500)
        synthetic_scaled = scaler.transform(synthetic_data)
        wd_base = np.mean([wasserstein_distance(sampled_scaled.flatten(), synthetic_scaled.flatten()) for _ in range(5)])
        base_thresholds['wd'] = wd_base * 1.5
        logger.info(f"Dataset {name} base WD threshold: {base_thresholds['wd']:.4f}")

        for size in data_sizes:
            for attack in attack_types:
                real_scaled, _ = sample_data(data, size, scaler, tvae)
                expected_stats = compute_statistical_summary(real_scaled)
                correct_hash = generate_dataset_hash(real_scaled, expected_stats)

                if attack == "data_source":
                    synthetic_scaled, dataset_hash, stats = simulate_data_source_attack(real_scaled, tvae, correct_hash, expected_stats, scaler)
                elif attack == "quality_fraud":
                    synthetic_scaled, dataset_hash = simulate_quality_fraud_attack(real_scaled, tvae, scaler)
                    stats = expected_stats
                elif attack == "privacy_leak":
                    synthetic_scaled, dataset_hash = simulate_privacy_leak_attack(real_scaled, tvae, scaler, leak_ratio=0.3)
                    stats = expected_stats
                elif attack == "distribution_tamper":
                    synthetic_scaled, dataset_hash = simulate_distribution_tamper_attack(real_scaled, tvae, scaler)
                    stats = expected_stats
                else:
                    synthetic_scaled = sample_data(data, size, scaler, tvae)[1]
                    dataset_hash = correct_hash
                    stats = expected_stats
                thresholds = compute_dynamic_thresholds(size, base_thresholds)
                logger.info(f"Thresholds for size {size}: {thresholds}")

                q1, q2 = Queue(maxsize=10), Queue(maxsize=10)
                result_queue = Queue()
                threads = [
                    Thread(target=server, args=(q2, q1, correct_hash, expected_stats), daemon=True),
                    Thread(target=client, args=(q1, q2, real_scaled, synthetic_scaled, dataset_hash, dataset_hash, thresholds, stats, result_queue, attack), daemon=True),
                ]
                for t in threads: t.start()
                for t in threads: t.join(timeout=600)

                result = result_queue.get(timeout=120) if not result_queue.empty() else {
                    "result": "Timeout",
                    "violations": ["Timeout"],
                    "hash_status": "Fail",
                    "metrics": {}
                }
                zkp_status = result.get("result", "N/A")
                quality_status = "Pass" if not result.get("violations") else "Fail"
                hash_status = result.get("hash_status", "Fail")
                wd_value = result.get('metrics', {}).get('wd', 'N/A')
                anonymity_value = result.get('metrics', {}).get('anonymity', 'N/A')
                wd_formatted = f"{wd_value:.4f}" if isinstance(wd_value, (int, float)) else wd_value
                anonymity_formatted = f"{anonymity_value:.4f}" if isinstance(anonymity_value, (int, float)) else anonymity_value

                results.append({
                    "Dataset": name,
                    "Size": size,
                    "Attack": attack or "None",
                    "ZKP Time(ms)": result.get("zkp_time", "N/A"),
                    "Total Time(ms)": result.get("total_time", "N/A"),
                    "ZKP Status": zkp_status,
                    "Quality Status": quality_status,
                    "Hash Status": hash_status,
                    "Violations": "\n".join(result.get("violations", [])) or "None",
                    "WD": wd_formatted,
                    "Anonymity": anonymity_formatted
                })
                all_metrics[(name, attack)].append(result.get("metrics", {}))

    # Statistical Analysis
    stat_results = []
    for name in datasets:
        for metric in ['wd', 'anonymity']:
            normal_metrics = all_metrics[(name, None)]
            for attack in attack_types[1:]:
                attack_metrics = all_metrics[(name, attack)]
                stat = perform_statistical_analysis(normal_metrics, attack_metrics, metric)
                stat_results.append({
                    "Dataset": name,
                    "Metric": metric,
                    "Attack": attack,
                    "T-Stat": stat["t_stat"],
                    "P-Value": stat["p_value"],
                    "Mean Normal": stat["mean_normal"],
                    "Mean Attack": stat["mean_attack"]
                })

    print("\nFinal Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))
    print("\nStatistical Analysis Results:")
    print(tabulate(stat_results, headers="keys", tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    main()


# ----------------------
# Main Experiment for nflow
# ----------------------
def perform_statistical_analysis(metrics_normal, metrics_attack, metric_name):
    """Perform statistical analysis between normal and attack metrics."""
    normal_values = [m[metric_name] for m in metrics_normal if metric_name in m]
    attack_values = [m[metric_name] for m in metrics_attack if metric_name in m]
    if len(normal_values) > 1 and len(attack_values) > 1:
        t_stat, p_value = ttest_ind(normal_values, attack_values, equal_var=False)
        return {
            "t_stat": t_stat,
            "p_value": p_value,
            "mean_normal": np.mean(normal_values),
            "mean_attack": np.mean(attack_values),
            "std_normal": np.std(normal_values),
            "std_attack": np.std(attack_values)
        }
    return {"t_stat": None, "p_value": None, "mean_normal": None, "mean_attack": None}

def main():
    datasets = {
        "Adult": ("/content/adult.csv", "csv"),
        "Loan": ("/content/Bank_Personal_Loan_Modelling.xlsx", "excel"),
        "Law": ("/content/post_processing_law.csv", "csv")
    }

    nflow_models = {}
    scalers = {}
    for name, (file_path, file_type) in datasets.items():
        data = load_dataset(file_path, file_type)
        scaler, nflow, _ = preprocess_and_train_nflow(data)
        scalers[name] = scaler
        nflow_models[name] = nflow

    data_sizes = [500, 1000, 5000]
    base_thresholds = {'wd': 1.0, 'ks': 0.5, 'kld': 10.0, 'ims': 0.03, 'anonymity': 0.7}
    attack_types = [None, "data_source", "quality_fraud", "privacy_leak", "distribution_tamper"]
    results, all_metrics = [], defaultdict(list)

    for name, (file_path, file_type) in datasets.items():
        data = load_dataset(file_path, file_type)
        nflow = nflow_models[name]
        scaler = scalers[name]
        if nflow is None:
            logger.error(f"NFLOW model for {name} is None")
            continue
        logger.info(f"\n{'='*50}\nProcessing dataset: {name}\n{'='*50}")

        # Compute dataset-specific base thresholds
        sampled_data = data.sample(n=min(500, len(data)), random_state=42)
        sampled_scaled = scaler.transform(sampled_data)
        synthetic_data = nflow.generate(num_samples=500)
        synthetic_scaled = scaler.transform(synthetic_data)
        wd_base = np.mean([wasserstein_distance(sampled_scaled.flatten(), synthetic_scaled.flatten()) for _ in range(5)])
        base_thresholds['wd'] = wd_base * 1.5
        logger.info(f"Dataset {name} base WD threshold: {base_thresholds['wd']:.4f}")

        for size in data_sizes:
            for attack in attack_types:
                real_scaled, _ = sample_data(data, size, scaler, nflow)
                expected_stats = compute_statistical_summary(real_scaled)
                correct_hash = generate_dataset_hash(real_scaled, expected_stats)

                if attack == "data_source":
                    synthetic_scaled, dataset_hash, stats = simulate_data_source_attack(real_scaled, nflow, correct_hash, expected_stats, scaler)
                elif attack == "quality_fraud":
                    synthetic_scaled, dataset_hash = simulate_quality_fraud_attack(real_scaled, nflow, scaler)
                    stats = expected_stats
                elif attack == "privacy_leak":
                    synthetic_scaled, dataset_hash = simulate_privacy_leak_attack(real_scaled, nflow, scaler, leak_ratio=0.3)
                    stats = expected_stats
                elif attack == "distribution_tamper":
                    synthetic_scaled, dataset_hash = simulate_distribution_tamper_attack(real_scaled, nflow, scaler)
                    stats = expected_stats
                else:
                    synthetic_scaled = sample_data(data, size, scaler, nflow)[1]
                    dataset_hash = correct_hash
                    stats = expected_stats
                thresholds = compute_dynamic_thresholds(size, base_thresholds)
                logger.info(f"Thresholds for size {size}: {thresholds}")

                q1, q2 = Queue(maxsize=10), Queue(maxsize=10)
                result_queue = Queue()
                threads = [
                    Thread(target=server, args=(q2, q1, correct_hash, expected_stats), daemon=True),
                    Thread(target=client, args=(q1, q2, real_scaled, synthetic_scaled, dataset_hash, dataset_hash, thresholds, stats, result_queue, attack), daemon=True),
                ]
                for t in threads: t.start()
                for t in threads: t.join(timeout=600)

                result = result_queue.get(timeout=120) if not result_queue.empty() else {
                    "result": "Timeout",
                    "violations": ["Timeout"],
                    "hash_status": "Fail",
                    "metrics": {}
                }
                zkp_status = result.get("result", "N/A")
                quality_status = "Pass" if not result.get("violations") else "Fail"
                hash_status = result.get("hash_status", "Fail")
                wd_value = result.get('metrics', {}).get('wd', 'N/A')
                anonymity_value = result.get('metrics', {}).get('anonymity', 'N/A')
                wd_formatted = f"{wd_value:.4f}" if isinstance(wd_value, (int, float)) else wd_value
                anonymity_formatted = f"{anonymity_value:.4f}" if isinstance(anonymity_value, (int, float)) else anonymity_formatted

                results.append({
                    "Dataset": name,
                    "Size": size,
                    "Attack": attack or "None",
                    "ZKP Time(ms)": result.get("zkp_time", "N/A"),
                    "Total Time(ms)": result.get("total_time", "N/A"),
                    "ZKP Status": zkp_status,
                    "Quality Status": quality_status,
                    "Hash Status": hash_status,
                    "Violations": "\n".join(result.get("violations", [])) or "None",
                    "WD": wd_formatted,
                    "Anonymity": anonymity_formatted
                })
                all_metrics[(name, attack)].append(result.get("metrics", {}))

    # Statistical Analysis
    stat_results = []
    for name in datasets:
        for metric in ['wd', 'anonymity']:
            normal_metrics = all_metrics[(name, None)]
            for attack in attack_types[1:]:
                attack_metrics = all_metrics[(name, attack)]
                stat = perform_statistical_analysis(normal_metrics, attack_metrics, metric)
                stat_results.append({
                    "Dataset": name,
                    "Metric": metric,
                    "Attack": attack,
                    "T-Stat": stat["t_stat"],
                    "P-Value": stat["p_value"],
                    "Mean Normal": stat["mean_normal"],
                    "Mean Attack": stat["mean_attack"]
                })

    print("\nFinal Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))
    print("\nStatistical Analysis Results:")
    print(tabulate(stat_results, headers="keys", tablefmt="grid", stralign="left"))

if __name__ == "__main__":
    main()
