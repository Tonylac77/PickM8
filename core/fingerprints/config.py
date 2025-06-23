"""
Configuration utilities for fingerprint calculations.
"""

from typing import Dict, Any


def create_default_fingerprint_config() -> Dict[str, Any]:
    """
    Create default configuration for fingerprint computation.
    
    Returns:
        Default fingerprint configuration
    """
    return {
        "compute_morgan": True,
        "morgan_radius": 2,
        "morgan_bits": 2048,
        "compute_rdkit": True,
        "rdkit_bits": 2048,
        "compute_mapchiral": True,
        "mapchiral_max_radius": 2,
        "mapchiral_n_permutations": 2048,
        "mapchiral_mapping": False,
        "compute_interactions": True
    }