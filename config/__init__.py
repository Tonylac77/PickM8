"""
PickM8 Configuration Module.

This module provides a clean API for loading and accessing configuration data.
It handles loading config.yaml from the project root and provides default values
for missing or incomplete configuration.

Usage:
    from config import config, get_ml_config, get_processing_config
    
    # Access full configuration
    ml_models = config['ml_models']
    
    # Access specific sections
    ml_config = get_ml_config()
    processing_config = get_processing_config()
"""

import logging
from typing import Dict, Any, Optional

from .loader import load_config, get_config_section, validate_config
from .defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Load configuration on module import
_config = load_config()

# Validate configuration
if not validate_config(_config):
    logger.warning("Configuration validation failed, some features may not work properly")

# Make configuration available as module-level variable
config = _config


def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Complete configuration dictionary with defaults applied.
    """
    return config


def get_ml_config() -> Dict[str, Any]:
    """
    Get machine learning configuration section.
    
    Returns:
        ML models configuration dictionary.
    """
    return get_config_section(config, 'ml_models')


def get_encoding_config() -> Dict[str, Any]:
    """
    Get encoding configuration section.
    
    Returns:
        Encoding configuration dictionary.
    """
    return get_config_section(config, 'encoding')


def get_interactions_config() -> Dict[str, Any]:
    """
    Get interactions configuration section.
    
    Returns:
        Interactions configuration dictionary.
    """
    return get_config_section(config, 'interactions')


def get_pose_quality_config() -> Dict[str, Any]:
    """
    Get pose quality configuration section.
    
    Returns:
        Pose quality configuration dictionary.
    """
    return get_config_section(config, 'pose_quality')


def get_processing_config() -> Dict[str, Any]:
    """
    Get processing configuration section.
    
    Returns:
        Processing configuration dictionary.
    """
    return get_config_section(config, 'processing')


def get_fingerprint_config() -> Dict[str, Any]:
    """
    Get fingerprint configuration section.
    
    Returns:
        Fingerprint configuration dictionary.
    """
    processing_config = get_processing_config()
    return processing_config.get('fingerprint_config', {})


def get_grade_config() -> Dict[str, Any]:
    """
    Get GRADE descriptor configuration section.
    
    Returns:
        GRADE configuration dictionary.
    """
    processing_config = get_processing_config()
    return processing_config.get('grade_config', {})


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get configuration for a specific ML model type.
    
    Args:
        model_type: Name of the ML model (e.g., 'RandomForest', 'GaussianProcess')
        
    Returns:
        Model-specific configuration dictionary.
    """
    ml_config = get_ml_config()
    return ml_config.get(model_type, {})


def get_default_model_type() -> str:
    """
    Get the default ML model type.
    
    Returns:
        Default model type string.
    """
    ml_config = get_ml_config()
    return ml_config.get('default_type', 'RandomForest')


def get_feature_engineering_config() -> Dict[str, Any]:
    """
    Get feature engineering configuration section.
    
    Returns:
        Feature engineering configuration dictionary.
    """
    ml_config = get_ml_config()
    return ml_config.get('feature_engineering', {})


def reload_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Reload configuration from file.
    
    Args:
        config_path: Optional path to config file.
        
    Returns:
        Reloaded configuration dictionary.
    """
    global config, _config
    
    from pathlib import Path
    path = Path(config_path) if config_path else None
    
    _config = load_config(path)
    config = _config
    
    if not validate_config(config):
        logger.warning("Configuration validation failed after reload")
    
    return config


# Expose commonly used functions and variables
__all__ = [
    'config',
    'get_config',
    'get_ml_config',
    'get_encoding_config',
    'get_interactions_config',
    'get_pose_quality_config',
    'get_processing_config',
    'get_fingerprint_config',
    'get_grade_config',
    'get_model_config',
    'get_default_model_type',
    'get_feature_engineering_config',
    'reload_config',
]