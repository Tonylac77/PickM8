"""
Configuration loader for PickM8.

This module handles loading and parsing the config.yaml file with proper
error handling and fallback to default values.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .defaults import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def get_config_path() -> Path:
    """Get the path to the config.yaml file in the project root."""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    return project_root / "config.yaml"


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        Dictionary containing configuration data, or empty dict if file not found.
    """
    if config_path is None:
        config_path = get_config_path()
    
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data is None:
                logger.warning(f"Config file {config_path} is empty, using defaults")
                return {}
            return config_data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file {config_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return {}


def merge_config(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user configuration with default configuration.
    
    Args:
        default_config: Default configuration dictionary
        user_config: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and merge configuration from YAML file with defaults.
    
    Args:
        config_path: Optional path to config file. If None, uses default location.
        
    Returns:
        Complete configuration dictionary with defaults applied.
    """
    user_config = load_yaml_config(config_path)
    return merge_config(DEFAULT_CONFIG, user_config)


def get_config_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Get a specific configuration section.
    
    Args:
        config: Full configuration dictionary
        section: Section name to retrieve
        
    Returns:
        Configuration section dictionary, or empty dict if not found.
    """
    return config.get(section, {})


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid, False otherwise.
    """
    required_sections = ['ml_models', 'encoding', 'interactions', 'pose_quality', 'processing']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate ML models section
    ml_models = config.get('ml_models', {})
    if 'default_type' not in ml_models:
        logger.error("Missing 'default_type' in ml_models configuration")
        return False
    
    # Validate encoding section
    encoding = config.get('encoding', {})
    if 'type' not in encoding:
        logger.error("Missing 'type' in encoding configuration")
        return False
    
    # Validate interactions section
    interactions = config.get('interactions', {})
    if 'interaction_type' not in interactions:
        logger.error("Missing 'interaction_type' in interactions configuration")
        return False
    
    return True