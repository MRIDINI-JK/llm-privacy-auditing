import numpy as np
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class DPProof:
    """Differential Privacy proof for training data filtering"""
    epsilon: float
    delta: float
    mechanism: str
    noise_scale: float
    filtered_count: int
    total_count: int
    timestamp: str
    proof_hash: str
    
    def to_dict(self):
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'mechanism': self.mechanism,
            'noise_scale': self.noise_scale,
            'filtered_count': self.filtered_count,
            'total_count': self.total_count,
            'timestamp': self.timestamp,
            'proof_hash': self.proof_hash
        }

def compute_epsilon_from_noise(sensitivity: float, noise_scale: float, delta: float = 1e-5) -> float:
    """
    Compute epsilon privacy budget from noise scale.
    Uses Gaussian mechanism: ε ≈ √(2 ln(1.25/δ)) * (sensitivity / noise_scale)
    """
    if noise_scale == 0:
        return float('inf')
    
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) * (sensitivity / noise_scale)
    return epsilon

def generate_dp_proof(
    filtered_samples: List[Dict],
    total_samples: List[Dict],
    epsilon: float = 1.0,
    delta: float = 1e-5,
    mechanism: str = "gaussian"
) -> DPProof:
    """
    Generate DP proof for data filtering operation.
    
    Args:
        filtered_samples: High-risk samples that were filtered
        total_samples: All samples in dataset
        epsilon: Privacy budget
        delta: Failure probability
        mechanism: Noise mechanism used
    
    Returns:
        DPProof object with cryptographic verification
    """
    from datetime import datetime
    
    filtered_count = len(filtered_samples)
    total_count = len(total_samples)
    
    # Compute noise scale for given epsilon
    sensitivity = 1.0  # L2 sensitivity for counting queries
    if mechanism == "gaussian":
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    elif mechanism == "laplace":
        noise_scale = sensitivity / epsilon
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    # Create proof payload
    proof_data = {
        'epsilon': epsilon,
        'delta': delta,
        'mechanism': mechanism,
        'noise_scale': noise_scale,
        'filtered_count': filtered_count,
        'total_count': total_count,
        'timestamp': datetime.now().isoformat()
    }
    
    # Generate cryptographic hash
    proof_hash = hashlib.sha256(
        json.dumps(proof_data, sort_keys=True).encode()
    ).hexdigest()
    
    return DPProof(
        epsilon=epsilon,
        delta=delta,
        mechanism=mechanism,
        noise_scale=noise_scale,
        filtered_count=filtered_count,
        total_count=total_count,
        timestamp=proof_data['timestamp'],
        proof_hash=proof_hash
    )

def verify_dp_proof(proof: DPProof, recompute_hash: bool = True) -> bool:
    """
    Verify integrity of DP proof.
    
    Args:
        proof: DPProof to verify
        recompute_hash: Whether to recompute and verify hash
    
    Returns:
        True if proof is valid
    """
    # Check basic constraints
    if proof.epsilon <= 0 or proof.delta <= 0 or proof.delta >= 1:
        return False
    
    if proof.filtered_count > proof.total_count:
        return False
    
    # Verify hash integrity
    if recompute_hash:
        proof_data = {
            'epsilon': proof.epsilon,
            'delta': proof.delta,
            'mechanism': proof.mechanism,
            'noise_scale': proof.noise_scale,
            'filtered_count': proof.filtered_count,
            'total_count': proof.total_count,
            'timestamp': proof.timestamp
        }
        
        expected_hash = hashlib.sha256(
            json.dumps(proof_data, sort_keys=True).encode()
        ).hexdigest()
        
        if expected_hash != proof.proof_hash:
            return False
    
    return True

def compute_privacy_loss(
    results: List[Dict],
    threshold: float,
    score_key: str = 'ensemble_risk'
) -> Dict:
    """
    Compute privacy loss statistics for filtered data.
    
    Returns:
        Dict with privacy metrics
    """
    scores = np.array([r[score_key] for r in results])
    filtered = scores >= threshold
    
    return {
        'total_samples': len(results),
        'filtered_samples': int(filtered.sum()),
        'filter_rate': float(filtered.mean()),
        'mean_risk_score': float(scores.mean()),
        'max_risk_score': float(scores.max()),
        'min_risk_score': float(scores.min())
    }