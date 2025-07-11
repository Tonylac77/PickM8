"""
Sklearn model wrapper to implement MLModelBase interface.
"""
import numpy as np
import pickle
import logging
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic

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
        if 'model' in self.model_config:
            base_model = self.model_config['model']
        else:
            # Determine if we need classification or regression model
            is_regression = self.model_category == 'regression'
            
            # Create base model
            if model_type == 'RandomForest':
                if is_regression:
                    base_model = RandomForestRegressor(**model_params)
                else:
                    base_model = RandomForestClassifier(**model_params)
            elif model_type == 'GaussianProcess':
                params = model_params.copy()
                kernel_type = params.pop('kernel', 'RBF')
                if kernel_type == 'RBF':
                    params['kernel'] = RBF()
                elif kernel_type == 'Matern':
                    params['kernel'] = Matern()
                elif kernel_type == 'RationalQuadratic':
                    params['kernel'] = RationalQuadratic()
                
                if is_regression:
                    base_model = GaussianProcessRegressor(**params)
                else:
                    base_model = GaussianProcessClassifier(**params)
            else:
                raise ValueError(f"Unsupported sklearn model type: {model_type}")
        
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
        
        # For regression models, return predictions as single column
        if self.is_regressor:
            predictions = self.predict(X)
            return predictions.reshape(-1, 1)
        
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