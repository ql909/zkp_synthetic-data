from noknow.core import ZK, ZKSignature, ZKData
from queue import Queue
import numpy as np
import hashlib
import json
import time
import logging
import traceback

logger = logging.getLogger(__name__)


# ----------------------
# Client with Adversarial Testing
# ----------------------
def client(iq: Queue, oq: Queue, real_data, synthetic_data, dataset_hash, synth_hash, thresholds, stats, result_queue, attack_type=None):
    start_time = time.time()
    comm_cost, zkp_time = 0, 0
    metric_times, metrics = {}, {}
    nonce = str(time.time())

    try:
        logger.info(f"Client: Starting with data size {len(real_data)}, hash: {dataset_hash[:16]}...")
        real_data = np.asarray(real_data)
        synthetic_data = np.asarray(synthetic_data)
        if real_data.shape != synthetic_data.shape:
            raise ValueError(f"Data shape mismatch: Real {real_data.shape}, Synthetic {synthetic_data.shape}")

        client_zk = ZK.new(curve_name="secp256k1", hash_alg="sha3_256")
        logger.info("Client: ZK initialized")
        metrics = compute_metrics(real_data, synthetic_data)

        violations = [f"{m.upper()} violation: {metrics[m]:.4f} {'>' if m in ['wd', 'ks', 'kld', 'ims', 'anonymity'] else '<'} {thresholds[m]:.4f}"
                     for m in metrics if (m in ['wd', 'ks', 'kld', 'ims', 'anonymity'] and metrics[m] > thresholds[m]) or
                     (m in ['dcr', 'nndr'] and metrics[m] < thresholds[m])]
        logger.info(f"Client: Threshold check: {violations or 'No violations'}")

        zkp_start = time.time()
        secret = dataset_hash + synth_hash + nonce
        metrics_str = ":".join(f"{v:.8f}" for v in metrics.values())
        stats_str = json.dumps(stats, sort_keys=True)
        metric_hash = hashlib.sha256((metrics_str + stats_str).encode()).hexdigest()
        signature = client_zk.create_signature(secret + metric_hash)
        sig_dump = signature.dump()
        logger.info("Client: Sending initial data to server")
        oq.put((dataset_hash, synth_hash, metrics_str, metric_hash, sig_dump, nonce, stats_str), timeout=30)
        comm_cost += len(sig_dump) + len(dataset_hash) + len(synth_hash) + len(metrics_str) + len(metric_hash) + len(nonce) + len(stats_str)

        logger.info("Client: Waiting for token from server")
        token = iq.get(timeout=180)
        comm_cost += len(str(token))

        proof = client_zk.sign(secret + metric_hash, token).dump()
        logger.info("Client: Sending proof to server")
        oq.put(proof, timeout=30)
        comm_cost += len(proof)

        result = iq.get(timeout=600)
        comm_cost += len(str(result))
        zkp_time = (time.time() - zkp_start) * 1000
        logger.info(f"Client: Completed with result: {result}")

        transaction = {
            "dataset_hash": dataset_hash,
            "synth_hash": synth_hash,
            "metrics": metrics,
            "result": result,
            "attack_type": attack_type
        }
        add_to_blockchain(transaction)

        result_queue.put({
            "zkp_time": zkp_time,
            "total_time": (time.time() - start_time) * 1000,
            "metric_times": metric_times,
            "cost": comm_cost,
            "result": result,
            "metrics": metrics,
            "violations": violations,
            "attack_type": attack_type,
            "hash_status": "Pass" if result == "SUCCESS" or "mismatch" not in result.lower() else "Fail"
        })

    except Exception as e:
        logger.error(f"Client error: {str(e)}\n{traceback.format_exc()}")
        result_queue.put({
            "zkp_time": "N/A",
            "total_time": (time.time() - start_time) * 1000,
            "metric_times": metric_times,
            "cost": comm_cost,
            "result": str(e),
            "metrics": {},  # Empty metrics on error
            "violations": [str(e)],
            "attack_type": attack_type,
            "hash_status": "Fail"
        })

# ----------------------
# Server with Enhanced Verification
# ----------------------
def server(iq: Queue, oq: Queue, dataset_hash, expected_stats):
    try:
        logger.info(f"Server: Processing hash {dataset_hash[:16]}...")
        server_zk = ZK.new(curve_name="secp256k1", hash_alg="sha3_256")

        logger.info("Server: Waiting for client data")
        received_hash, synth_hash, metrics_str, metric_hash, sig, nonce, stats_str = iq.get(timeout=180)
        client_sig = ZKSignature.load(sig)
        received_stats = json.loads(stats_str)
        logger.info(f"Server: Received hash {received_hash[:16]}..., synth_hash {synth_hash[:16]}...")

        # Verify statistical summary
        for key in ['mean', 'std', 'quantiles']:
            if not np.allclose(received_stats[key], expected_stats[key], atol=1e-2):
                error_msg = f"Statistical {key} mismatch detected"
                logger.error(error_msg)
                oq.put(error_msg, timeout=30)
                return

        if received_hash != dataset_hash:
            error_msg = f"Hash mismatch: Expected {dataset_hash[:16]}... vs Received {received_hash[:16]}..."
            logger.error(error_msg)
            oq.put(error_msg, timeout=30)
            return

        if hashlib.sha256((metrics_str + stats_str).encode()).hexdigest() != metric_hash:
            error_msg = "Metric or stats hash mismatch detected"
            logger.error(error_msg)
            oq.put(error_msg, timeout=30)
            return

        token = str(server_zk.token())
        logger.info(f"Server: Sending token: {token[:16]}...")
        oq.put(token, timeout=30)

        proof_data = iq.get(timeout=180)
        proof = ZKData.load(proof_data)
        logger.info("Server: Proof received")

        client_zk = ZK(client_sig.params)
        verification_result = client_zk.verify(proof, client_sig, data=token + dataset_hash + synth_hash + metric_hash + nonce)
        result = "SUCCESS" if verification_result else "ZKP verification failed"
        logger.info(f"Server: Verification result: {result}")
        oq.put(result, timeout=30)

    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        oq.put(f"Server error: {str(e)}", timeout=30)


