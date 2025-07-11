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

from machine_learning.models.ml_base import MLModelBase
from machine_learning.models.sklearn_models import SklearnModelWrapper
from machine_learning.models.autoparty_models import AutoPartyEnsemble
from machine_learning.models.ordinal_models import OrdinalRegressionWrapper

from machine_learning.encoding.encodings import (
    encode_grades_for_training as encode_grades_with_type,
    decode_predictions,
    get_ml_strategy,
    SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION
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
        'calibration_enabled': False,
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
        'GaussianProcess': {
            'kernel': 'RBF',
            'n_restarts_optimizer': 0,
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
        'default_grades': ['A', 'B', 'C', 'D']
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if 'encoding' in config:
                    encoding_config = config['encoding']
                    
                    # Validate encoding type
                    encoding_type = encoding_config.get('type', SEQUENTIAL)
                    if encoding_type not in [SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION]:
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

def determine_model_category(encoding_type: str) -> str:
    """
    Determine the appropriate model category based on encoding type.
    
    Args:
        encoding_type: The encoding strategy being used
        
    Returns:
        String indicating model category ('classification', 'regression', 'ordinal')
    """
    category_mapping = {
        SEQUENTIAL: 'classification',
        ONE_HOT: 'classification', 
        ORDINAL: 'ordinal',
        ORDINAL_REGRESSION: 'regression'
    }
    
    return category_mapping.get(encoding_type, 'classification')

def create_model(model_type: str, model_params: Optional[Dict[str, Any]] = None, 
                use_calibration: bool = False, encoding_type: Optional[str] = None) -> MLModelBase:
    """
    Factory function to create ML models with proper configuration based on encoding strategy.
    
    Args:
        model_type: Type of model ('RandomForest', 'GaussianProcess', 'LogisticAT')
        model_params: Model-specific parameters (uses config defaults if None)
        use_calibration: Whether to wrap model with calibration (classification only)
        encoding_type: Encoding strategy to determine model category
        
    Returns:
        MLModelBase instance
    """
    if model_params is None:
        config = load_model_config()
        model_params = config.get(model_type, {})
    
    # Determine model category based on encoding type
    if encoding_type is not None:
        model_category = determine_model_category(encoding_type)
    else:
        model_category = 'classification'  # Default
    
    # Create model configuration
    model_config = {
        'model_type': model_type,
        'model_params': model_params,
        'use_calibration': use_calibration,
        'model_category': model_category,
        'encoding_type': encoding_type
    }
    
    # Supported models (deprecated models removed)
    sklearn_models = ['RandomForest', 'GaussianProcess']
    ordinal_models = ['LogisticAT']
    
    if model_type in sklearn_models:
        # Calibration disabled - no longer adding calibration config
        return SklearnModelWrapper(model_config)
        
    elif model_type in ordinal_models:
        # Ordinal regression models
        return OrdinalRegressionWrapper(model_config)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: {sklearn_models + ordinal_models}")


def prepare_features_from_dataframe(df: pd.DataFrame, 
                                    use_interaction_fp: bool = True,
                                    use_e3fp_fp: bool = False,
                                    use_ecfp_fp: bool = True,
                                    use_electroshape_fp: bool = False,
                                    use_functional_groups_fp: bool = True,
                                    use_maccs_fp: bool = True,
                                    use_pattern_fp: bool = False,
                                    use_pharmacophore_fp: bool = False) -> Tuple[np.ndarray, List[int]]:
    """
    Extract feature matrix from molecules DataFrame for ML training/prediction.
    
    Args:
        df: Molecules DataFrame
        use_interaction_fp: Include interaction fingerprints
        use_e3fp_fp: Include E3FP fingerprints
        use_ecfp_fp: Include ECFP fingerprints
        use_electroshape_fp: Include ElectroShape fingerprints
        use_functional_groups_fp: Include FunctionalGroups fingerprints
        use_maccs_fp: Include MACCS fingerprints
        use_pattern_fp: Include Pattern fingerprints
        use_pharmacophore_fp: Include Pharmacophore fingerprints
        
    Returns:
        Tuple of (feature_matrix, molecule_ids)
    """
    if len(df) == 0:
        return np.array([]), []
    
    # Ensure at least one fingerprint type is selected
    fingerprint_selections = [
        use_interaction_fp, use_e3fp_fp, use_ecfp_fp, use_electroshape_fp,
        use_functional_groups_fp, use_maccs_fp, use_pattern_fp,
        use_pharmacophore_fp
    ]
    
    if not any(fingerprint_selections):
        raise ValueError("At least one fingerprint type must be selected")
        
    # Build valid mask based on selected fingerprint types
    valid_mask = pd.Series([True] * len(df), index=df.index)
    
    if use_interaction_fp:
        valid_mask = valid_mask & df['interaction_fp'].notna()
    if use_e3fp_fp:
        valid_mask = valid_mask & df['e3fp_fp'].notna()
    if use_ecfp_fp:
        valid_mask = valid_mask & df['ecfp_fp'].notna()
    if use_electroshape_fp:
        valid_mask = valid_mask & df['electroshape_fp'].notna()
    if use_functional_groups_fp:
        valid_mask = valid_mask & df['functional_groups_fp'].notna()
    if use_maccs_fp:
        valid_mask = valid_mask & df['maccs_fp'].notna()
    if use_pattern_fp:
        valid_mask = valid_mask & df['pattern_fp'].notna()
    if use_pharmacophore_fp:
        valid_mask = valid_mask & df['pharmacophore_fp'].notna()
    
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return np.array([]), []
    
    features = []
    for _, row in valid_df.iterrows():
        feature_vector = []
        
        # Add interaction fingerprints
        if use_interaction_fp and row['interaction_fp'] is not None:
            try:
                ifp_data = json.loads(row['interaction_fp'])
                if isinstance(ifp_data, list):
                    feature_vector.extend(ifp_data)
                elif isinstance(ifp_data, dict):
                    feature_vector.extend(list(ifp_data.values()))
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Add scikit-fingerprints features
        if use_e3fp_fp and row['e3fp_fp'] is not None:
            feature_vector.extend([float(x) for x in row['e3fp_fp']])
        
        if use_ecfp_fp and row['ecfp_fp'] is not None:
            feature_vector.extend([float(x) for x in row['ecfp_fp']])
        
        if use_electroshape_fp and row['electroshape_fp'] is not None:
            feature_vector.extend([float(x) for x in row['electroshape_fp']])
        
        if use_functional_groups_fp and row['functional_groups_fp'] is not None:
            feature_vector.extend([float(x) for x in row['functional_groups_fp']])
        
        if use_maccs_fp and row['maccs_fp'] is not None:
            feature_vector.extend([float(x) for x in row['maccs_fp']])
        
        if use_pattern_fp and row['pattern_fp'] is not None:
            feature_vector.extend([float(x) for x in row['pattern_fp']])
        
        if use_pharmacophore_fp and row['pharmacophore_fp'] is not None:
            feature_vector.extend([float(x) for x in row['pharmacophore_fp']])
        
        features.append(feature_vector)
    
    # Ensure all feature vectors have the same length
    if features:
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]
    
    selected_fps = []
    if use_interaction_fp: selected_fps.append("Interaction")
    if use_e3fp_fp: selected_fps.append("E3FP")
    if use_ecfp_fp: selected_fps.append("ECFP")
    if use_electroshape_fp: selected_fps.append("ElectroShape")
    if use_functional_groups_fp: selected_fps.append("FunctionalGroups")
    if use_maccs_fp: selected_fps.append("MACCS")
    if use_pattern_fp: selected_fps.append("Pattern")
    if use_pharmacophore_fp: selected_fps.append("Pharmacophore")
    
    logger.info(f"Prepared features for {len(features)} molecules using fingerprints: {', '.join(selected_fps)}")
    
    return np.array(features), valid_df['id'].tolist()

def encode_grades_for_training(df: pd.DataFrame, encoding_type: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades for ML training using configurable encoding type.
    
    Args:
        df: Molecules DataFrame with grades
        encoding_type: Type of encoding to use (sequential, one_hot, ordinal, ordinal_regression). Uses config default if None.
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
    """
    if encoding_type is None:
        encoding_config = load_encoding_config()
        encoding_type = encoding_config.get('type', SEQUENTIAL)
    
    # Use the new encoding function from encodings module
    return encode_grades_with_type(df, encoding_type)

def train_model(df: pd.DataFrame, 
                model_config: Optional[Dict[str, Any]] = None,
                use_interaction_fp: bool = True,
                use_e3fp_fp: bool = False,
                use_ecfp_fp: bool = True,
                use_electroshape_fp: bool = False,
                use_functional_groups_fp: bool = True,
                use_maccs_fp: bool = True,
                use_pattern_fp: bool = False,
                use_pharmacophore_fp: bool = False,
                feature_engineering_config: Optional[Dict[str, Any]] = None) -> Tuple[MLModelBase, Dict[str, Any]]:
    """
    Train ML model on graded molecules with configurable model type and parameters.
    
    Args:
        df: Molecules DataFrame with grades
        model_config: Configuration dict with model_type, model_params, use_calibration, encoding_type
        use_interaction_fp: Include interaction fingerprints
        use_e3fp_fp: Include E3FP fingerprints
        use_ecfp_fp: Include ECFP fingerprints
        use_electroshape_fp: Include ElectroShape fingerprints
        use_functional_groups_fp: Include FunctionalGroups fingerprints
        use_maccs_fp: Include MACCS fingerprints
        use_pattern_fp: Include Pattern fingerprints
        use_pharmacophore_fp: Include Pharmacophore fingerprints
        feature_engineering_config: Feature engineering configuration dict
        
    Returns:
        Tuple of (trained_model, metrics)
    """
    # Get graded molecules only
    graded_df = df[df['grade'].notna()].copy()
    
    if len(graded_df) < 3:
        raise ValueError("Need at least 3 graded molecules to train a model")
    
    # Prepare features with fingerprint selection and optional feature engineering
    if feature_engineering_config is not None:
        # Import feature engineering module
        from machine_learning.training.feature_engineering import engineer_features_for_training
        
        # Create feature engineering configuration
        fe_config = {
            'use_interaction_fp': use_interaction_fp,
            'use_e3fp_fp': use_e3fp_fp,
            'use_ecfp_fp': use_ecfp_fp,
            'use_electroshape_fp': use_electroshape_fp,
            'use_functional_groups_fp': use_functional_groups_fp,
            'use_maccs_fp': use_maccs_fp,
            'use_pattern_fp': use_pattern_fp,
            'use_pharmacophore_fp': use_pharmacophore_fp
        }
        
        # Apply feature engineering
        X, feature_metadata = engineer_features_for_training(
            graded_df,
            variance_threshold=feature_engineering_config.get('variance_threshold', 0.0),
            use_importance_selection=feature_engineering_config.get('use_importance_selection', False),
            use_dimensionality_reduction=feature_engineering_config.get('use_dimensionality_reduction', False),
            normalize=feature_engineering_config.get('normalize', False),
            config=fe_config
        )
    else:
        # Use original feature preparation
        X, _ = prepare_features_from_dataframe(
            graded_df, use_interaction_fp, use_e3fp_fp, use_ecfp_fp, 
            use_electroshape_fp, use_functional_groups_fp, use_maccs_fp, 
            use_pattern_fp, use_pharmacophore_fp
        )
        feature_metadata = None
    
    if len(X) == 0:
        raise ValueError("No valid features found for training")
    
# Get model configuration
    if model_config is None:
        logger.info("No model config provided, using defaults")
        model_type = 'RandomForest'
        model_params = None
        use_calibration = False
        encoding_type = None  # Will use config default
    else:
        model_type = model_config.get('model_type', 'RandomForest')
        model_params = model_config.get('model_params')
        use_calibration = model_config.get('use_calibration', False)
        encoding_type = model_config.get('encoding_type')
    
    # Force single-threaded execution for sklearn models
    if model_params is None:
        model_params = {}
    if model_type in ['RandomForest', 'GradientBoosting']:
        model_params['n_jobs'] = 1
    
    # Encode grades with specified encoding type
    y, label_mapping = encode_grades_for_training(graded_df, encoding_type)
    
    # Get the actual encoding type used (in case it was None and defaulted)
    if encoding_type is None:
        encoding_config = load_encoding_config()
        encoding_type = encoding_config.get('type', SEQUENTIAL)
    
    # Create model using factory function with encoding type
    try:
        model = create_model(model_type, model_params, use_calibration, encoding_type)
    except Exception as e:
        logger.error(f"Error creating {model_type} model: {e}, falling back to RandomForest")
        model = create_model('RandomForest', None, use_calibration, encoding_type)
        model_type = 'RandomForest'
    
    # Add output_type to model config for AutoParty
    if hasattr(model, 'model_config'):
        model.model_config['output_type'] = 'ordinal' if encoding_type == ORDINAL else 'classes'
        model.model_config['encoding_type'] = encoding_type
    
    # Train the model
    model.fit(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    
    # Calculate metrics based on model category
    if model.is_regressor:
        # For regression models, use MSE and R²
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        accuracy = r2  # Use R² as the main metric for regression
        
        logger.info(f"Regression metrics - MSE: {mse:.3f}, R²: {r2:.3f}")
    else:
        # For classification models, use accuracy
        # For ordinal encoding, convert back to class indices
        if encoding_type == ORDINAL and len(y_pred.shape) > 1:
            y_pred = np.sum(y_pred > 0.5, axis=1) - 1
            y_pred = np.clip(y_pred, 0, len(label_mapping) - 1)
        # Compare predictions with the format used for training
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'n_samples': len(X),
        'n_features': X.shape[1] if len(X) > 0 else 0,
        'label_mapping': label_mapping,
        'model_type': model_type,
        'use_calibration': use_calibration,
        'encoding_type': encoding_type,
        'ml_strategy': get_ml_strategy(encoding_type),
        'feature_metadata': feature_metadata
    }
    
    logger.info(f"Trained {model_type} model with accuracy: {accuracy:.3f}")
    
    return model, metrics

def update_predictions(df: pd.DataFrame, 
                       model: MLModelBase, 
                       metrics: Optional[Dict[str, Any]] = None,
                       use_interaction_fp: bool = True,
                       use_e3fp_fp: bool = False,
                       use_ecfp_fp: bool = True,
                       use_electroshape_fp: bool = False,
                       use_functional_groups_fp: bool = True,
                       use_maccs_fp: bool = True,
                       use_pattern_fp: bool = False,
                       use_pharmacophore_fp: bool = False,
                       feature_engineering_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Update DataFrame with ML predictions.
    
    Args:
        df: Molecules DataFrame
        model: Trained MLModelBase model
        metrics: Training metrics containing encoding information
        use_interaction_fp: Include interaction fingerprints
        use_e3fp_fp: Include E3FP fingerprints
        use_ecfp_fp: Include ECFP fingerprints
        use_electroshape_fp: Include ElectroShape fingerprints
        use_functional_groups_fp: Include FunctionalGroups fingerprints
        use_maccs_fp: Include MACCS fingerprints
        use_pattern_fp: Include Pattern fingerprints
        use_pharmacophore_fp: Include Pharmacophore fingerprints
        feature_engineering_config: Feature engineering configuration dict
        
    Returns:
        Updated DataFrame with predictions
    """
    df = df.copy()
    
    # Prepare features for all molecules with fingerprint selection
    if feature_engineering_config is not None and metrics is not None and 'feature_metadata' in metrics:
        # Use feature engineering with consistent transformations
        from machine_learning.training.feature_engineering import create_hybrid_fingerprint_features, apply_feature_engineering_for_prediction
        
        # Create raw features
        fe_config = {
            'use_interaction_fp': use_interaction_fp,
            'use_e3fp_fp': use_e3fp_fp,
            'use_ecfp_fp': use_ecfp_fp,
            'use_electroshape_fp': use_electroshape_fp,
            'use_functional_groups_fp': use_functional_groups_fp,
            'use_maccs_fp': use_maccs_fp,
            'use_pattern_fp': use_pattern_fp,
            'use_pharmacophore_fp': use_pharmacophore_fp
        }
        
        X_raw, _ = create_hybrid_fingerprint_features(
            df, 
            use_ecfp=fe_config['use_ecfp_fp'],
            use_maccs=fe_config['use_maccs_fp'],
            use_interaction=fe_config['use_interaction_fp']
        )
        
        if len(X_raw) == 0:
            return df
            
        # Apply same feature engineering transformations used during training
        X = apply_feature_engineering_for_prediction(X_raw, metrics['feature_metadata'])
        mol_ids = df['id'].tolist()
    else:
        # Use original feature preparation
        X, mol_ids = prepare_features_from_dataframe(
            df, use_interaction_fp, use_e3fp_fp, use_ecfp_fp,
            use_electroshape_fp, use_functional_groups_fp, use_maccs_fp,
            use_pattern_fp, use_pharmacophore_fp
        )
    
    if len(X) == 0:
        return df
    
    # Get encoding information from metrics
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
    
    # Handle different prediction formats based on encoding type and model category
    if encoding_type == ORDINAL_REGRESSION:
        # For ordinal regression with continuous predictions, use decoding function
        try:
            grade_predictions = decode_predictions(raw_predictions, label_mapping, encoding_type)
        except Exception as e:
            logger.error(f"Error decoding ordinal regression predictions: {e}")
            # Fallback to binning
            grade_predictions = []
            for pred in raw_predictions:
                if pred < 25:
                    grade_predictions.append('D')
                elif pred < 50:
                    grade_predictions.append('C')
                elif pred < 75:
                    grade_predictions.append('B')
                else:
                    grade_predictions.append('A')
    elif encoding_type == ORDINAL and len(raw_predictions.shape) > 1:
        # Convert ordinal predictions to class indices
        pred_indices = np.sum(raw_predictions > 0.5, axis=1) - 1
        pred_indices = np.clip(pred_indices, 0, len(label_mapping) - 1)
    elif encoding_type == ONE_HOT and len(raw_predictions.shape) > 1:
        # Get class with highest probability
        pred_indices = np.argmax(raw_predictions, axis=1)
    else:
        # Sequential or already converted
        pred_indices = np.round(raw_predictions).astype(int)
        max_val = max(label_mapping.values()) if label_mapping else 3
        pred_indices = np.clip(pred_indices, 0, max_val)
    
    # Decode predictions to grade strings (if not already done for ORDINAL_REGRESSION)
    if encoding_type != ORDINAL_REGRESSION:
        if label_mapping:
            try:
                grade_predictions = decode_predictions(pred_indices, label_mapping, encoding_type)
            except Exception as e:
                logger.error(f"Error decoding predictions: {e}, using raw predictions")
                grade_predictions = [str(pred) for pred in pred_indices]
        else:
            grade_predictions = [str(pred) for pred in pred_indices]
    
    # Update DataFrame
    timestamp = pd.Timestamp.now()
    for i, mol_id in enumerate(mol_ids):
        mask = df['id'] == mol_id
        if mask.any():
            df.loc[mask, 'prediction'] = grade_predictions[i]
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
    if encoding_type not in [SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION]:
        logger.warning(f"Invalid encoding type '{encoding_type}', falling back to sequential")
        model_config['encoding_type'] = SEQUENTIAL
    
    return model_config