"""Configuration loading and management utilities for PickM8."""
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration loading fails."""
    pass

def load_config() -> Dict[str, Any]:
    """
    Load the entire config.yaml file.
    
    Returns:
        Complete configuration dictionary
        
    Raises:
        ConfigurationError: If config.yaml is missing or malformed
    """
    config_path = Path("config.yaml")
    
    if not config_path.exists():
        raise ConfigurationError("config.yaml file not found in the project root")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if config is None:
            raise ConfigurationError("config.yaml is empty or malformed")
            
        return config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Error parsing config.yaml: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading config.yaml: {e}")

def load_ml_config() -> Dict[str, Any]:
    """
    Load ML model configuration from config.yaml.
    
    Returns:
        ML models configuration dictionary
        
    Raises:
        ConfigurationError: If ml_models section is missing or malformed
    """
    try:
        config = load_config()
        
        if 'ml_models' not in config:
            raise ConfigurationError("'ml_models' section missing from config.yaml")
            
        ml_config = config['ml_models']
        
        # Validate required fields
        required_fields = ['default_type', 'RandomForest', 'GaussianProcess', 'LogisticAT']
        missing_fields = [field for field in required_fields if field not in ml_config]
        if missing_fields:
            raise ConfigurationError(f"Missing required ML config fields: {missing_fields}")
            
        return ml_config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Error loading ML configuration: {e}")

def load_fingerprint_config() -> Dict[str, Any]:
    """
    Load fingerprint configuration from config.yaml.
    
    Returns:
        Fingerprint configuration dictionary
        
    Raises:
        ConfigurationError: If fingerprint config section is missing
    """
    try:
        config = load_config()
        
        if 'processing' not in config:
            raise ConfigurationError("'processing' section missing from config.yaml")
            
        if 'fingerprint_config' not in config['processing']:
            raise ConfigurationError("'processing.fingerprint_config' section missing from config.yaml")
            
        fp_config = config['processing']['fingerprint_config']
        
        # Validate that we have the expected fingerprint compute flags
        expected_compute_flags = [
            'compute_mapchiral', 'compute_e3fp', 'compute_ecfp', 'compute_electroshape',
            'compute_functional_groups', 'compute_maccs', 'compute_pattern', 'compute_pharmacophore'
        ]
        
        missing_flags = [flag for flag in expected_compute_flags if flag not in fp_config]
        if missing_flags:
            raise ConfigurationError(f"Missing fingerprint compute flags in config.yaml: {missing_flags}")
            
        return fp_config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Error loading fingerprint configuration: {e}")

def load_interaction_config() -> Dict[str, Any]:
    """
    Load interaction configuration from config.yaml.
    
    Returns:
        Interaction configuration dictionary
        
    Raises:
        ConfigurationError: If interactions section is missing
    """
    try:
        config = load_config()
        
        if 'interactions' not in config:
            raise ConfigurationError("'interactions' section missing from config.yaml")
            
        interaction_config = config['interactions']
        
        # Validate required fields
        required_fields = ['interaction_type', 'ligand_name', 'max_workers']
        missing_fields = [field for field in required_fields if field not in interaction_config]
        if missing_fields:
            raise ConfigurationError(f"Missing required interaction config fields: {missing_fields}")
            
        return interaction_config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Error loading interaction configuration: {e}")

def load_pose_quality_config() -> Dict[str, Any]:
    """
    Load pose quality configuration from config.yaml.
    
    Returns:
        Pose quality configuration dictionary
        
    Raises:
        ConfigurationError: If pose_quality section is missing
    """
    try:
        config = load_config()
        
        if 'pose_quality' not in config:
            raise ConfigurationError("'pose_quality' section missing from config.yaml")
            
        pose_config = config['pose_quality']
        
        # Validate required fields
        required_fields = ['enabled', 'calculate_clashes', 'calculate_strain', 'max_workers']
        missing_fields = [field for field in required_fields if field not in pose_config]
        if missing_fields:
            raise ConfigurationError(f"Missing required pose quality config fields: {missing_fields}")
            
        return pose_config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Error loading pose quality configuration: {e}")

def load_grade_config() -> Dict[str, Any]:
    """
    Load GRADE configuration from config.yaml.
    
    Returns:
        GRADE configuration dictionary
        
    Raises:
        ConfigurationError: If GRADE config section is missing
    """
    try:
        config = load_config()
        
        if 'processing' not in config:
            raise ConfigurationError("'processing' section missing from config.yaml")
            
        if 'grade_config' not in config['processing']:
            raise ConfigurationError("'processing.grade_config' section missing from config.yaml")
            
        grade_config = config['processing']['grade_config']
        
        # Validate required fields
        required_fields = ['enabled', 'extended', 'normalize_charges', 'max_workers']
        missing_fields = [field for field in required_fields if field not in grade_config]
        if missing_fields:
            raise ConfigurationError(f"Missing required GRADE config fields: {missing_fields}")
            
        return grade_config
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Error loading GRADE configuration: {e}")

def build_session_config(user_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build complete session configuration by merging user selections with config.yaml defaults.
    
    Args:
        user_options: Dictionary containing user selections from the UI:
            - fingerprint_types: List[str] - selected fingerprint types
            - interaction_type: str - selected interaction type
            - compute_pose_quality: bool - whether to compute pose quality
            - compute_grade: bool - whether to compute GRADE descriptors
            
    Returns:
        Complete configuration dictionary for session creation
        
    Raises:
        ConfigurationError: If any configuration section is missing or malformed
    """
    try:
        # Load all base configurations from YAML
        fingerprint_config = load_fingerprint_config()
        interaction_config = load_interaction_config()
        pose_config = load_pose_quality_config()
        grade_config = load_grade_config()
        ml_config = load_ml_config()
        
        # Map user fingerprint selections to compute flags
        user_fingerprint_types = user_options.get('fingerprint_types', [])
        
        # Start with all fingerprints disabled
        for compute_flag in ['compute_mapchiral', 'compute_e3fp', 'compute_ecfp', 'compute_electroshape',
                            'compute_functional_groups', 'compute_maccs', 'compute_pattern', 'compute_pharmacophore']:
            fingerprint_config[compute_flag] = False
        
        # Enable only the selected fingerprints
        fingerprint_mapping = {
            'mapchiral': 'compute_mapchiral',
            'e3fp': 'compute_e3fp',
            'ecfp': 'compute_ecfp',
            'electroshape': 'compute_electroshape',
            'functional_groups': 'compute_functional_groups',
            'maccs': 'compute_maccs',
            'pattern': 'compute_pattern',
            'pharmacophore': 'compute_pharmacophore'
        }
        
        for fp_type in user_fingerprint_types:
            if fp_type in fingerprint_mapping:
                fingerprint_config[fingerprint_mapping[fp_type]] = True
            else:
                logger.warning(f"Unknown fingerprint type: {fp_type}")
        
        # Set interaction type from user selection
        interaction_config['interaction_type'] = user_options.get('interaction_type', 'plip')
        
        # Set pose quality enabled flag
        pose_enabled = user_options.get('compute_pose_quality', True)
        pose_config['enabled'] = pose_enabled
        
        # Set GRADE enabled flag  
        grade_enabled = user_options.get('compute_grade', False)
        grade_config['enabled'] = grade_enabled
        
        # Build final configuration
        session_config = {
            'fingerprint_config': fingerprint_config,
            'interaction_config': interaction_config,
            'pose_config': pose_config,
            'grade_config': grade_config,
            'compute_fingerprints': len(user_fingerprint_types) > 0,
            'compute_interactions': True,
            'compute_pose_quality': pose_enabled,
            'compute_grade_descriptors': grade_enabled,
            'model_config': {
                'model_type': user_options.get('model_type', ml_config.get('default_type', 'RandomForest')),
                'model_params': user_options.get('model_params', {}),
                'use_calibration': user_options.get('use_calibration', False)
            }
        }
        
        logger.info(f"Built session config with fingerprints: {user_fingerprint_types}, "
                   f"interaction: {interaction_config['interaction_type']}, "
                   f"pose_quality: {pose_enabled}, grade: {grade_enabled}")
        
        return session_config
        
    except ConfigurationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Error building session configuration: {e}")