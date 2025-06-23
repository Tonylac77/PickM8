"""
Machine learning models for active learning.
"""

import numpy as np
import pickle
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Machine learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


def create_ml_model(model_type: str = "random_forest", **kwargs) -> object:
    """
    Create ML model for active learning.
    
    Args:
        model_type: Type of model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Scikit-learn model instance
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', None),
            min_samples_split=kwargs.get('min_samples_split', 2),
            min_samples_leaf=kwargs.get('min_samples_leaf', 1),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1)
        )
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            learning_rate=kwargs.get('learning_rate', 0.1),
            max_depth=kwargs.get('max_depth', 3),
            random_state=kwargs.get('random_state', 42)
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            C=kwargs.get('C', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            n_jobs=kwargs.get('n_jobs', -1)
        )
    elif model_type == "svm":
        return SVC(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            probability=True,  # Enable probability prediction for uncertainty
            random_state=kwargs.get('random_state', 42)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model_with_calibration(X: np.ndarray, y: np.ndarray, model_type: str = "random_forest",
                               use_calibration: bool = True, **kwargs) -> Tuple[object, Dict[str, float]]:
    """
    Train ML model with optional probability calibration.
    
    Args:
        X: Feature matrix
        y: Target labels (encoded)
        model_type: Type of model to train
        use_calibration: Whether to use probability calibration
        **kwargs: Model parameters
        
    Returns:
        Tuple of (trained_model, performance_metrics)
    """
    logger.info(f"Starting model training with {len(X)} samples, {X.shape[1] if len(X) > 0 else 0} features")
    logger.info(f"Target labels: {y}")
    logger.info(f"Model type: {model_type}, Use calibration: {use_calibration}")
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Empty training data")
    
    # Check data consistency
    if len(X) != len(y):
        raise ValueError(f"Feature matrix and labels have different lengths: {len(X)} vs {len(y)}")
    
    # Create base model
    logger.info(f"Creating {model_type} model...")
    base_model = create_ml_model(model_type, **kwargs)
    logger.info(f"Created model: {base_model}")
    
    # Train base model first
    logger.info("Training base model...")
    try:
        base_model.fit(X, y)
        logger.info("Base model training completed successfully")
    except Exception as e:
        logger.error(f"Error training base model: {e}")
        raise
    
    # Use calibration for better uncertainty estimates (only if enough samples and classes)
    unique_classes = np.unique(y)
    min_samples_per_class = min([np.sum(y == cls) for cls in unique_classes])
    
    logger.info(f"Unique classes: {unique_classes}")
    logger.info(f"Min samples per class: {min_samples_per_class}")
    logger.info(f"Total samples: {len(X)}")
    
    if (use_calibration and len(unique_classes) > 1 and 
        len(X) >= 20 and min_samples_per_class >= 3):
        try:
            logger.info("Attempting probability calibration...")
            # Use 3-fold CV only if we have enough samples per class
            cv_folds = min(3, min_samples_per_class)
            logger.info(f"Using {cv_folds} CV folds for calibration")
            model = CalibratedClassifierCV(base_model, method='isotonic', cv=cv_folds)
            model.fit(X, y)
            logger.info("Calibration completed successfully")
        except Exception as e:
            logger.warning(f"Calibration failed, using base model: {e}")
            model = base_model
    else:
        logger.info("Skipping calibration (insufficient data or single class)")
        model = base_model
    
    # Calculate simple training accuracy
    logger.info("Calculating performance metrics...")
    metrics = {}
    if len(np.unique(y)) > 1:  # Only if we have multiple classes
        try:
            y_pred = model.predict(X)
            metrics['train_accuracy'] = float(accuracy_score(y, y_pred))
            logger.info(f"Training accuracy: {metrics['train_accuracy']:.3f}")
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {e}")
            metrics['error'] = str(e)
            metrics['train_accuracy'] = 0.0
    else:
        metrics['train_accuracy'] = 1.0  # Single class case
        logger.info("Single class case, setting accuracy to 1.0")
    
    logger.info(f"Model training completed. Final metrics: {metrics}")
    
    return model, metrics


def predict_with_uncertainty(model: object, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with uncertainty estimates.
    
    Args:
        model: Trained ML model
        X: Feature matrix
        
    Returns:
        Tuple of (predictions, probabilities, uncertainties)
    """
    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Get predictions and probabilities
    predictions = model.predict(X)
    
    # Get prediction probabilities for uncertainty estimation
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        
        # Calculate uncertainty as 1 - max_probability (higher uncertainty = less confident)
        uncertainties = 1.0 - np.max(probabilities, axis=1)
    else:
        # Fallback for models without probability prediction
        probabilities = np.zeros((len(X), 1))
        uncertainties = np.ones(len(X)) * 0.5  # Medium uncertainty
    
    return predictions, probabilities, uncertainties


def save_model_and_encoders(model: object, label_encoders: Dict[str, Any], 
                          session_dir: str, model_name: str = "model") -> None:
    """
    Save trained model and label encoders.
    
    Args:
        model: Trained ML model
        label_encoders: Label encoding dictionaries
        session_dir: Session directory path
        model_name: Name for model file
    """
    try:
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = session_path / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save encoders
        encoders_path = session_path / f"{model_name}_encoders.json"
        with open(encoders_path, 'w') as f:
            json.dump(label_encoders, f, indent=2, default=str)
        
        logger.info(f"Saved model to {model_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model_and_encoders(session_dir: str, model_name: str = "model") -> Tuple[Optional[object], Optional[Dict[str, Any]]]:
    """
    Load trained model and label encoders.
    
    Args:
        session_dir: Session directory path
        model_name: Name of model file
        
    Returns:
        Tuple of (model, label_encoders) or (None, None) if not found
    """
    try:
        session_path = Path(session_dir)
        
        model_path = session_path / f"{model_name}.pkl"
        encoders_path = session_path / f"{model_name}_encoders.json"
        
        if not model_path.exists() or not encoders_path.exists():
            return None, None
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load encoders
        with open(encoders_path, 'r') as f:
            encoders = json.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        
        return model, encoders
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None