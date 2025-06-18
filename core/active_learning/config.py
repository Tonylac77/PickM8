"""
Configuration utilities for active learning.
"""

from typing import Dict, Any


def create_default_ml_config() -> Dict[str, Any]:
    """
    Create default configuration for ML training.
    
    Returns:
        Default ML configuration
    """
    return {
        "model_type": "random_forest",
        "use_calibration": True,
        "include_pose_metrics": True,
        "selection_strategy": "uncertainty",
        "n_molecules_per_round": 10,
        "min_samples_for_training": 5,
        "model_params": {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42
        }
    }