"""
Configuration utilities for pose analysis.
"""

from typing import Dict, Any


def create_default_posecheck_config() -> Dict[str, Any]:
    """
    Create default configuration for PoseCheck analysis.
    
    Returns:
        Default PoseCheck configuration
    """
    return {
        "calculate_clashes": True,
        "calculate_strain": True,
        "clash_threshold": 2.5,
        "use_simple_fallback": True  # Use simplified analysis if PoseCheck fails
    }