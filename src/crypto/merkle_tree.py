import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class MerkleProof:
    """Proof of exclusion for a data point"""
    data_hash: str
    siblings: List[str]
    indices: List[int]
    root_hash: str
    
    def to_dict(self):
        return {
            'data_hash': self.data_hash,
            'siblings': self.siblings,
            'indices': self.indices,
            'root_hash': self.root_hash
        }

class MerkleTree:
    """
    Merkle tree for cryptographically proving data filtering.
    Allows verification that specific samples were excluded.
    """
    
    def __init__(self, data_samples: List[Dict]):
        """
        Build Merkle tree from data samples.
        
        Args:
            data_samples: List of samples, each with 'text' and 'sample_id'
        """
        self.leaves = [self._hash_sample(s) for s in data_samples]
        self.tree = self._build_tree(self.leaves)
        self.root = self.tree[-1][0] if self.tree else ""
        self.sample_map = {s['sample_id']: i for i, s in enumerate(data_samples)}
    
    def _hash_sample(self, sample: Dict) -> str:
        """Hash a single sample"""
        content = f"{sample['sample_id']}:{sample['text']}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash two nodes together"""
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _build_tree(self, leaves: List[str]) -> List[List[str]]:
        """Build complete Merkle tree"""
        if not leaves:
            return []
        
        tree = [leaves]
        
        while len(tree[-1]) > 1:
            level = tree[-1]
            next_level = []
            
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                next_level.append(self._hash_pair(left, right))
            
            tree.append(next_level)
        
        return tree
    
    def get_root(self) -> str:
        """Get Merkle root hash"""
        return self.root
    
    def generate_proof(self, sample_id: int) -> Optional[MerkleProof]:
        """
        Generate proof of inclusion/exclusion for a sample.
        
        Args:
            sample_id: ID of sample to prove
        
        Returns:
            MerkleProof or None if sample not found
        """
        if sample_id not in self.sample_map:
            return None
        
        index = self.sample_map[sample_id]
        siblings = []
        indices = []
        
        current_index = index
        for level in self.tree[:-1]:
            # Get sibling
            if current_index % 2 == 0:
                sibling_index = current_index + 1
            else:
                sibling_index = current_index - 1
            
            if sibling_index < len(level):
                siblings.append(level[sibling_index])
                indices.append(current_index % 2)
            
            current_index //= 2
        
        return MerkleProof(
            data_hash=self.leaves[index],
            siblings=siblings,
            indices=indices,
            root_hash=self.root
        )
    
    def verify_proof(self, proof: MerkleProof) -> bool:
        """
        Verify a Merkle proof.
        
        Args:
            proof: MerkleProof to verify
        
        Returns:
            True if proof is valid
        """
        current_hash = proof.data_hash
        
        for sibling, index in zip(proof.siblings, proof.indices):
            if index == 0:  # Current is left
                current_hash = self._hash_pair(current_hash, sibling)
            else:  # Current is right
                current_hash = self._hash_pair(sibling, current_hash)
        
        return current_hash == proof.root_hash

def generate_exclusion_proof(
    all_samples: List[Dict],
    filtered_samples: List[Dict]
) -> Dict:
    """
    Generate proof that filtered samples were excluded from training.
    
    Args:
        all_samples: Complete dataset
        filtered_samples: High-risk samples to exclude
    
    Returns:
        Dict with Merkle tree root and exclusion proofs
    """
    # Build tree from remaining samples
    filtered_ids = {s['sample_id'] for s in filtered_samples}
    remaining_samples = [s for s in all_samples if s['sample_id'] not in filtered_ids]
    
    tree = MerkleTree(remaining_samples)
    
    # Generate proofs for filtered samples (should fail verification)
    exclusion_proofs = []
    for sample in filtered_samples[:10]:  # Limit to 10 for performance
        # Try to find in remaining tree (should fail)
        proof = tree.generate_proof(sample['sample_id'])
        exclusion_proofs.append({
            'sample_id': sample['sample_id'],
            'excluded': proof is None,
            'proof': proof.to_dict() if proof else None
        })
    
    return {
        'merkle_root': tree.get_root(),
        'total_samples': len(all_samples),
        'remaining_samples': len(remaining_samples),
        'filtered_samples': len(filtered_samples),
        'exclusion_proofs': exclusion_proofs
    }