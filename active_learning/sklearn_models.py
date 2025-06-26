"""
Sklearn model wrapper to implement MLModelBase interface.
"""
import numpy as np
import pickle
import logging
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from .ml_base import MLModelBase

logger = logging.getLogger(__name__)


class SklearnModelWrapper(MLModelBase):
    """Wrapper for sklearn models to implement MLModelBase interface."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize sklearn model wrapper.
        
        Args:
            model_config: Configuration with model_type, model_params, use_calibration
        """
        super().__init__(model_config)
        self._backend = 'sklearn'
        
        # Create the sklearn model
        self.model = self._create_sklearn_model(
            model_config['model_type'],
            model_config.get('model_params', {}),
            model_config.get('use_calibration', True)
        )
        
    def _create_sklearn_model(self, model_type: str, model_params: Dict[str, Any], 
                              use_calibration: bool) -> object:
        """Create sklearn model based on type and parameters."""
        # Create base model
        if model_type == 'RandomForest':
            base_model = RandomForestClassifier(**model_params)
        elif model_type == 'GradientBoosting':
            base_model = GradientBoostingClassifier(**model_params)
        elif model_type == 'SVM':
            # Ensure probability is enabled
            model_params = model_params.copy()
            model_params['probability'] = True
            base_model = SVC(**model_params)
        elif model_type == 'GaussianProcess':
            # Handle kernel specification
            params = model_params.copy()
            kernel_type = params.pop('kernel', 'RBF')
            if kernel_type == 'RBF':
                params['kernel'] = RBF()
            elif kernel_type == 'Matern':
                params['kernel'] = Matern()
            base_model = GaussianProcessClassifier(**params)
        elif model_type == 'MLP':
            base_model = MLPClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported sklearn model type: {model_type}")
        
        # Add calibration if requested and beneficial
        if use_calibration and model_type in ['RandomForest', 'SVM', 'MLP']:
            calibration_method = self.model_config.get('calibration_method', 'isotonic')
            calibration_cv = self.model_config.get('calibration_cv', 3)
            return CalibratedClassifierCV(base_model, method=calibration_method, cv=calibration_cv)
        
        return base_model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the sklearn model."""
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"Trained {self.model_type} model with sklearn backend")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fallback for models without predict_proba
            predictions = self.predict(X)
            # Convert to one-hot style probabilities
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, int(pred)] = 1.0
            return proba
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Get uncertainty estimates."""
        if not self.is_trained:
            raise ValueError("Model must be trained before estimating uncertainty")
        
        # Special handling for Gaussian Process
        if isinstance(self.model, GaussianProcessClassifier):
            try:
                _, std = self.model.predict(X, return_std=True)
                return std
            except:
                pass
        
        # General approach: use probability entropy
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            # Uncertainty = 1 - max probability
            return 1.0 - np.max(probabilities, axis=1)
        else:
            # Fallback: return moderate uncertainty
            return np.ones(len(X)) * 0.5
    
    def save(self, filepath: str) -> None:
        """Save model using pickle."""
        with open(f"{filepath}_sklearn.pkl", 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.model_config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load model from pickle."""
        with open(f"{filepath}_sklearn.pkl", 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_config = data['config']
            self.is_trained = data['is_trained']