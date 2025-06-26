"""ML training and predictions functions."""
import pandas as pd
import numpy as np
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

from data.encodings import (
    encode_grades_for_training as encode_grades_with_type,
    decode_predictions,
    get_ml_strategy,
    SEQUENTIAL, NOMINAL, ORDINAL
)

logger = logging.getLogger(__name__)

def load_model_config() -> Dict[str, Any]:
    """
    Load ML model configuration from config.yaml with fallback defaults.
    
    Returns:
        Dictionary containing model configurations
    """
    config_path = Path("config.yaml")
    
    # Fallback defaults if config not found
    default_config = {
        'default_type': 'RandomForest',
        'calibration_enabled': True,
        'calibration_method': 'isotonic',
        'calibration_cv': 3,
        'RandomForest': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        },
        'GradientBoosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'random_state': 42
        },
        'SVM': {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        },
        'GaussianProcess': {
            'kernel': 'RBF',
            'n_restarts_optimizer': 0,
            'random_state': 42
        },
        'MLP': {
            'hidden_layer_sizes': [100],
            'learning_rate': 'constant',
            'alpha': 0.0001,
            'max_iter': 200,
            'activation': 'relu',
            'solver': 'adam',
            'random_state': 42
        }
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'ml_models' in config:
                    return config['ml_models']
                else:
                    logger.warning("ml_models section not found in config.yaml, using defaults")
                    return default_config
        else:
            logger.warning("config.yaml not found, using default model configuration")
            return default_config
    except Exception as e:
        logger.error(f"Error loading config.yaml: {e}, using defaults")
        return default_config

def load_encoding_config() -> Dict[str, Any]:
    """
    Load encoding configuration from config.yaml with fallback defaults.
    
    Returns:
        Dictionary containing encoding configuration
    """
    config_path = Path("config.yaml")
    
    # Fallback defaults if config not found (ensures backward compatibility)
    default_config = {
        'type': SEQUENTIAL,
        'default_grades': ['A', 'B', 'C', 'D', 'F']
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'encoding' in config:
                    encoding_config = config['encoding']
                    
                    # Validate encoding type
                    encoding_type = encoding_config.get('type', SEQUENTIAL)
                    if encoding_type not in [SEQUENTIAL, NOMINAL, ORDINAL]:
                        logger.warning(f"Invalid encoding type '{encoding_type}', using default '{SEQUENTIAL}'")
                        encoding_config['type'] = SEQUENTIAL
                    
                    # Ensure default_grades is a list
                    if 'default_grades' not in encoding_config or not isinstance(encoding_config['default_grades'], list):
                        encoding_config['default_grades'] = default_config['default_grades']
                    
                    return encoding_config
                else:
                    logger.info("encoding section not found in config.yaml, using defaults for backward compatibility")
                    return default_config
        else:
            logger.info("config.yaml not found, using default encoding configuration")
            return default_config
    except Exception as e:
        logger.error(f"Error loading encoding config from config.yaml: {e}, using defaults")
        return default_config

def create_model(model_type: str, model_params: Optional[Dict[str, Any]] = None, 
                use_calibration: bool = True) -> object:
    """
    Factory function to create sklearn models with proper configuration.
    
    Args:
        model_type: Type of model ('RandomForest', 'GradientBoosting', 'SVM', 'GaussianProcess', 'MLP')
        model_params: Model-specific parameters (uses config defaults if None)
        use_calibration: Whether to wrap model with calibration
        
    Returns:
        Configured sklearn model
    """
    if model_params is None:
        config = load_model_config()
        model_params = config.get(model_type, {})
    
    # Create base model
    if model_type == 'RandomForest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(**model_params)
    elif model_type == 'SVM':
        # Ensure probability is enabled for uncertainty estimation
        model_params = model_params.copy()
        model_params['probability'] = True
        model = SVC(**model_params)
    elif model_type == 'GaussianProcess':
        # Handle kernel specification
        params = model_params.copy()
        kernel_type = params.pop('kernel', 'RBF')
        if kernel_type == 'RBF':
            params['kernel'] = RBF()
        elif kernel_type == 'Matern':
            params['kernel'] = Matern()
        model = GaussianProcessClassifier(**params)
    elif model_type == 'MLP':
        model = MLPClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Add calibration wrapper if requested and beneficial
    if use_calibration and model_type in ['RandomForest', 'SVM', 'MLP']:
        config = load_model_config()
        calibration_method = config.get('calibration_method', 'isotonic')
        calibration_cv = config.get('calibration_cv', 3)
        model = CalibratedClassifierCV(model, method=calibration_method, cv=calibration_cv)
    
    return model

def get_uncertainty_estimate(model: object, X: np.ndarray) -> np.ndarray:
    """
    Get uncertainty estimates tailored to each model type.
    
    Args:
        model: Trained sklearn model
        X: Feature matrix
        
    Returns:
        Array of uncertainty estimates (higher = more uncertain)
    """
    try:
        # For Gaussian Process, use native uncertainty
        if hasattr(model, 'predict_proba') and isinstance(model, GaussianProcessClassifier):
            try:
                # GP can provide true uncertainty via standard deviation
                _, std = model.predict(X, return_std=True)
                return std
            except:
                # Fallback to probability-based uncertainty
                probabilities = model.predict_proba(X)
                return 1.0 - np.max(probabilities, axis=1)
        
        # For other models with predict_proba
        elif hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
            # Uncertainty = 1 - max probability (higher = more uncertain)
            return 1.0 - np.max(probabilities, axis=1)
        
        else:
            # Fallback for models without probability estimates
            logger.warning("Model does not support uncertainty estimation, returning default values")
            return np.ones(len(X)) * 0.5
            
    except Exception as e:
        logger.error(f"Error computing uncertainty estimates: {e}")
        return np.ones(len(X)) * 0.5

def prepare_features_from_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    """
    Extract feature matrix from molecules DataFrame for ML training/prediction.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Tuple of (feature_matrix, molecule_ids)
    """
    if len(df) == 0:
        return np.array([]), []
        
    # Get molecules with computed fingerprints
    valid_mask = (df['morgan_fp'].notna()) & (df['interaction_fp'].notna())
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return np.array([]), []
    
    features = []
    for _, row in valid_df.iterrows():
        feature_vector = []
        
        # Morgan fingerprint
        if row['morgan_fp'] is not None:
            feature_vector.extend([int(x) for x in row['morgan_fp']])
        
        # RDKit fingerprint  
        if row['rdkit_fp'] is not None:
            feature_vector.extend([int(x) for x in row['rdkit_fp']])
            
        # Interaction fingerprint (convert JSON to numeric)
        if row['interaction_fp'] is not None:
            try:
                ifp_data = json.loads(row['interaction_fp'])
                if isinstance(ifp_data, list):
                    feature_vector.extend(ifp_data)
                elif isinstance(ifp_data, dict):
                    feature_vector.extend(list(ifp_data.values()))
            except (json.JSONDecodeError, TypeError):
                pass
        
        features.append(feature_vector)
    
    # Ensure all feature vectors have the same length
    if features:
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]
    
    return np.array(features), valid_df['id'].tolist()

def encode_grades_for_training(df: pd.DataFrame, encoding_type: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades for ML training using configurable encoding type.
    
    Args:
        df: Molecules DataFrame with grades
        encoding_type: Type of encoding to use (sequential, nominal, ordinal). Uses config default if None.
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
    """
    if encoding_type is None:
        encoding_config = load_encoding_config()
        encoding_type = encoding_config.get('type', SEQUENTIAL)
    
    # Use the new encoding function from encodings module
    return encode_grades_with_type(df, encoding_type)

def train_model(df: pd.DataFrame, model_config: Optional[Dict[str, Any]] = None) -> Tuple[object, Dict[str, Any]]:
    """
    Train ML model on graded molecules with configurable model type and parameters.
    
    Args:
        df: Molecules DataFrame with grades
        model_config: Configuration dict with model_type, model_params, use_calibration, encoding_type
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    # Get graded molecules only
    graded_df = df[df['grade'].notna()].copy()
    
    if len(graded_df) < 3:
        raise ValueError("Need at least 3 graded molecules to train a model")
    
    # Prepare features
    X, _ = prepare_features_from_dataframe(graded_df)
    
    if len(X) == 0:
        raise ValueError("No valid features found for training")
    
    # Get model configuration (use defaults if not provided for backward compatibility)
    if model_config is None:
        logger.info("No model config provided, using defaults")
        model_type = 'RandomForest'
        model_params = None
        use_calibration = True
        encoding_type = None  # Will use config default
    else:
        model_type = model_config.get('model_type', 'RandomForest')
        model_params = model_config.get('model_params')
        use_calibration = model_config.get('use_calibration', True)
        encoding_type = model_config.get('encoding_type')
    
    # Encode grades with specified encoding type
    y, label_mapping = encode_grades_for_training(graded_df, encoding_type)
    
    # Get the actual encoding type used (in case it was None and defaulted)
    if encoding_type is None:
        encoding_config = load_encoding_config()
        encoding_type = encoding_config.get('type', SEQUENTIAL)
    
    # Create model using factory function
    try:
        model = create_model(model_type, model_params, use_calibration=False)  # We'll handle calibration below
    except Exception as e:
        logger.error(f"Error creating {model_type} model: {e}, falling back to RandomForest")
        model = create_model('RandomForest', None, use_calibration=False)
        model_type = 'RandomForest'
    
    # Add calibration for uncertainty estimates if requested and beneficial
    unique_classes = np.unique(y)
    if use_calibration and len(unique_classes) > 1 and len(y) >= 15:
        # Check if we have enough samples per class for cross-validation
        class_counts = np.bincount(y)
        min_class_count = np.min(class_counts)
        if min_class_count >= 3:
            config = load_model_config()
            calibration_method = config.get('calibration_method', 'isotonic')
            calibration_cv = config.get('calibration_cv', 3)
            
            # Only calibrate models that benefit from it (not GaussianProcess)
            if model_type != 'GaussianProcess':
                model = CalibratedClassifierCV(model, method=calibration_method, cv=calibration_cv)
                logger.info(f"Added {calibration_method} calibration to {model_type} model")
    
    # Train the model
    model.fit(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'n_samples': len(X),
        'n_features': X.shape[1] if len(X) > 0 else 0,
        'label_mapping': label_mapping,
        'model_type': model_type,
        'use_calibration': use_calibration,
        'encoding_type': encoding_type,
        'ml_strategy': get_ml_strategy(encoding_type)
    }
    
    logger.info(f"Trained {model_type} model with accuracy: {accuracy:.3f}")
    
    return model, metrics

def update_predictions(df: pd.DataFrame, model: object, metrics: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Update DataFrame with ML predictions using enhanced uncertainty estimation.
    
    Args:
        df: Molecules DataFrame
        model: Trained ML model
        metrics: Training metrics containing encoding information
        
    Returns:
        Updated DataFrame with predictions
    """
    df = df.copy()
    
    # Prepare features for all molecules
    X, mol_ids = prepare_features_from_dataframe(df)
    
    if len(X) == 0:
        return df
    
    # Get encoding information from metrics (backward compatibility)
    if metrics is not None:
        encoding_type = metrics.get('encoding_type', SEQUENTIAL)
        label_mapping = metrics.get('label_mapping', {})
    else:
        # Fallback for backward compatibility
        encoding_type = SEQUENTIAL
        label_mapping = {}
        logger.warning("No metrics provided, using sequential encoding for predictions")
    
    # Make predictions
    raw_predictions = model.predict(X)
    
    # Decode predictions to grade strings if we have label mapping
    if label_mapping:
        try:
            grade_predictions = decode_predictions(raw_predictions, label_mapping, encoding_type)
        except Exception as e:
            logger.error(f"Error decoding predictions: {e}, using raw predictions")
            grade_predictions = [str(pred) for pred in raw_predictions]
    else:
        grade_predictions = [str(pred) for pred in raw_predictions]
    
    # Get uncertainty estimates using the enhanced function
    uncertainties = get_uncertainty_estimate(model, X)
    
    # Update DataFrame
    timestamp = pd.Timestamp.now()
    for i, mol_id in enumerate(mol_ids):
        mask = df['id'] == mol_id
        if mask.any():
            df.loc[mask, 'prediction'] = grade_predictions[i]
            df.loc[mask, 'prediction_uncertainty'] = uncertainties[i]
            df.loc[mask, 'prediction_timestamp'] = timestamp
    
    logger.info(f"Updated predictions for {len(mol_ids)} molecules")
    
    return df


def ensure_backward_compatibility(df: pd.DataFrame, model_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure backward compatibility for existing sessions and models.
    
    Args:
        df: Molecules DataFrame
        model_config: Optional model configuration
        
    Returns:
        Updated model configuration with encoding defaults
    """
    if model_config is None:
        model_config = {}
    
    # Add encoding type if not specified (backward compatibility)
    if 'encoding_type' not in model_config:
        model_config['encoding_type'] = SEQUENTIAL
        logger.info("Added default sequential encoding for backward compatibility")
    
    # Validate encoding type
    encoding_type = model_config.get('encoding_type', SEQUENTIAL)
    if encoding_type not in [SEQUENTIAL, NOMINAL, ORDINAL]:
        logger.warning(f"Invalid encoding type '{encoding_type}', falling back to sequential")
        model_config['encoding_type'] = SEQUENTIAL
    
    return model_config