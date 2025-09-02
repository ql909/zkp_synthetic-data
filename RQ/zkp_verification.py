from noknow.core import ZK, ZKSignature, ZKData
from queue import Queue
import numpy as np
import hashlib
import json
import time
import logging
import traceback

logger = logging.getLogger(__name__)

def zkp_prover(real_data, synthetic_data, thresholds, attack_type=None):
    """Non-interactive ZKP prover for synthetic data verification."""
    start_time = time.time()
    comm_cost, zkp_time = 0, 0
    result_queue = Queue()
    
    try:
        logger.info(f"Prover: Starting with data size {len(real_data)}...")
        real_data = np.asarray(real_data)
        synthetic_data = np.asarray(synthetic_data)
        if real_data.shape != synthetic_data.shape:
            raise ValueError(f"Data shape mismatch: Real {real_data.shape}, Synthetic {synthetic_data.shape}")

        # Initialize ZKP instance
        client_zk = ZK.new(curve_name="secp256k1", hash_alg="sha3_256")
        logger.info("Prover: ZK initialized")

        # Compute dataset hashes
        dataset_hash = hashlib.sha256(real_data.tobytes()).hexdigest()
        synth_hash = hashlib.sha256(synthetic_data.tobytes()).hexdigest()
        logger.info(f"Prover: Dataset hash {dataset_hash[:16]}..., Synth hash {synth_hash[:16]}...")

        # Pre-register dataset hash on blockchain
        blockchain_transaction = {"dataset_hash": dataset_hash, "timestamp": time.time()}
        if not add_to_blockchain(blockchain_transaction):
            raise ValueError("Failed to pre-register dataset hash on blockchain")

        # Compute statistical summaries
        stats = compute_statistical_summaries(real_data)
        stats_str = json.dumps(stats, sort_keys=True)

        # Compute quality metrics
        metrics = compute_metrics(real_data, synthetic_data)
        violations = [
            f"{m.upper()} violation: {metrics[m]:.4f} {'>' if m in ['wd', 'ks', 'kld', 'ims', 'anonymity'] else '<'} {thresholds[m]:.4f}"
            for m in metrics
            if (m in ['wd', 'ks', 'kld', 'ims', 'anonymity'] and metrics[m] > thresholds[m]) or
               (m not in ['wd', 'ks', 'kld', 'ims', 'anonymity'] and metrics[m] < thresholds[m])
        ]
        logger.info(f"Prover: Threshold check: {violations or 'No violations'}")

        # Step 1: Prover Initialization
        zkp_start = time.time()
        domain_separator = "ZKP_Synthetic_Data_Verification"
        nonce = str(time.time())
        metrics_str = ":".join(f"{v:.8f}" for v in metrics.values())
        metric_hash = hashlib.sha256((metrics_str + stats_str).encode()).hexdigest()

        # Step 2: Commitment and Proof (Fiat-Shamir transform)
        commitment = client_zk.commit()
        secret = hashlib.sha256((dataset_hash + synth_hash + nonce + domain_separator).encode()).hexdigest()
        challenge = hashlib.sha256((commitment.dump() + client_zk.params.public + dataset_hash + synth_hash + metric_hash + nonce + domain_separator).encode()).hexdigest()
        proof = client_zk.sign(secret + metric_hash, challenge)
        proof_dump = proof.dump()
        comm_cost += len(proof_dump) + len(dataset_hash) + len(synth_hash) + len(metrics_str) + len(metric_hash) + len(nonce) + len(stats_str)

        # Package data for verifier
        prover_output = {
            "dataset_hash": dataset_hash,
            "synth_hash": synth_hash,
            "metric_hash": metric_hash,
            "proof": proof_dump,
            "commitment": commitment.dump(),
            "nonce": nonce,
            "stats": stats_str,
            "metrics": metrics,
            "violations": violations
        }
        zkp_time = (time.time() - zkp_start) * 1000

        # Log and store result
        transaction = {
            "dataset_hash": dataset_hash,
            "synth_hash": synth_hash,
            "metrics": metrics,
            "attack_type": attack_type
        }
        add_to_blockchain(transaction)
        result_queue.put({
            "zkp_time": zkp_time,
            "total_time": (time.time() - start_time) * 1000,
            "cost": comm_cost,
            "metrics": metrics,
            "violations": violations,
            "attack_type": attack_type,
            "prover_output": prover_output
        })

    except Exception as e:
        logger.error(f"Prover error: {str(e)}\n{traceback.format_exc()}")
        result_queue.put({
            "zkp_time": "N/A",
            "total_time": (time.time() - start_time) * 1000,
            "cost": comm_cost,
            "metrics": {},
            "violations": [str(e)],
            "attack_type": attack_type,
            "prover_output": None
        })

    return result_queue.get()

def zkp_verifier(prover_output, expected_dataset_hash, expected_stats, thresholds):
    """Non-interactive ZKP verifier for synthetic data verification."""
    start_time = time.time()
    comm_cost = 0
    try:
        logger.info(f"Verifier: Processing dataset hash {expected_dataset_hash[:16]}...")
        server_zk = ZK.new(curve_name="secp256k1", hash_alg="sha3_256")
        
        # Extract prover output
        dataset_hash = prover_output["dataset_hash"]
        synth_hash = prover_output["synth_hash"]
        metric_hash = prover_output["metric_hash"]
        proof = ZKData.load(prover_output["proof"])
        commitment = ZKData.load(prover_output["commitment"])
        nonce = prover_output["nonce"]
        stats_str = prover_output["stats"]
        metrics = prover_output["metrics"]
        received_stats = json.loads(stats_str)
        
        # Verify statistical summary
        for key in ['mean', 'std', 'quantiles']:
            if not np.allclose(received_stats[key], expected_stats[key], atol=1e-2):
                error_msg = f"Statistical {key} mismatch detected"
                logger.error(error_msg)
                return {
                    "result": error_msg,
                    "total_time": (time.time() - start_time) * 1000,
                    "cost": comm_cost,
                    "hash_status": "Fail"
                }

        # Verify dataset hash against blockchain-registered hash
        if dataset_hash != expected_dataset_hash:
            error_msg = f"Hash mismatch: Expected {expected_dataset_hash[:16]}... vs Received {dataset_hash[:16]}..."
            logger.error(error_msg)
            return {
                "result": error_msg,
                "total_time": (time.time() - start_time) * 1000,
                "cost": comm_cost,
                "hash_status": "Fail"
            }

        # Verify metric hash
        metrics_str = ":".join(f"{v:.8f}" for v in metrics.values())
        if hashlib.sha256((metrics_str + stats_str).encode()).hexdigest() != metric_hash:
            error_msg = "Metric or stats hash mismatch detected"
            logger.error(error_msg)
            return {
                "result": error_msg,
                "total_time": (time.time() - start_time) * 1000,
                "cost": comm_cost,
                "hash_status": "Fail"
            }

        # Recompute challenge (Fiat-Shamir transform)
        domain_separator = "ZKP_Synthetic_Data_Verification"
        challenge = hashlib.sha256((commitment.dump() + server_zk.params.public + dataset_hash + synth_hash + metric_hash + nonce + domain_separator).encode()).hexdigest()

        # Verify proof
        verification_result = server_zk.verify(proof, server_zk.create_signature(dataset_hash + synth_hash + nonce + domain_separator + metric_hash), data=challenge)
        result = "SUCCESS" if verification_result else "ZKP verification failed"
        comm_cost += len(prover_output["proof"]) + len(dataset_hash) + len(synth_hash) + len(metrics_str) + len(metric_hash) + len(nonce) + len(stats_str)

        logger.info(f"Verifier: Verification result: {result}")
        return {
            "result": result,
            "total_time": (time.time() - start_time) * 1000,
            "cost": comm_cost,
            "hash_status": "Pass" if result == "SUCCESS" else "Fail"
        }

    except Exception as e:
        logger.error(f"Verifier error: {str(e)}\n{traceback.format_exc()}")
        return {
            "result": f"Verifier error: {str(e)}",
            "total_time": (time.time() - start_time) * 1000,
            "cost": comm_cost,
            "hash_status": "Fail"
        }
