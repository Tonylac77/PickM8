"""
Ordinal regression model wrapper using mord library.
"""
import numpy as np
import pickle
import logging
from typing import Dict, Any, Optional

from mord import LogisticAT
from .ml_base import MLModelBase

logger = logging.getLogger(__name__)


class OrdinalRegressionWrapper(MLModelBase):
    """Wrapper for mord ordinal regression models to implement MLModelBase interface."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize ordinal regression model wrapper.
        
        Args:
            model_config: Configuration with model_type, model_params
        """
        super().__init__(model_config)
        self._backend = 'mord'
        self.model_category = 'ordinal'
        
        # Create the mord model
        self.model = self._create_ordinal_model(
            model_config['model_type'],
            model_config.get('model_params', {})
        )
        
    def _create_ordinal_model(self, model_type: str, model_params: Dict[str, Any]) -> object:
        """Create mord model based on type and parameters."""
        if model_type == 'LogisticAT':
            # LogisticAT is the All-Threshold variant of ordinal logistic regression
            return LogisticAT(**model_params)
        else:
            raise ValueError(f"Unsupported ordinal model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ordinal regression model."""
        # Ensure y is in the correct format for ordinal regression
        # mord expects integer labels starting from 0
        if len(y.shape) > 1:
            # If y is ordinal encoded (cumulative), convert to class indices
            y = np.sum(y > 0.5, axis=1) - 1
            y = np.clip(y, 0, y.max())
        
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"Trained {self.model_type} ordinal regression model with mord backend")
    
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
            # Fallback: convert predictions to one-hot style probabilities
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, int(pred)] = 1.0
            return proba
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Get uncertainty estimates for ordinal regression."""
        if not self.is_trained:
            raise ValueError("Model must be trained before estimating uncertainty")
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
            # For ordinal regression, uncertainty can be calculated as entropy
            # or as the spread of the probability distribution
            epsilon = 1e-10
            probs = probabilities + epsilon
            probs = probs / np.sum(probs, axis=1, keepdims=True)  # Normalize
            entropy = -np.sum(probs * np.log(probs), axis=1)
            return entropy
        else:
            # Fallback: return moderate uncertainty
            return np.ones(len(X)) * 0.5
    
    def save(self, filepath: str) -> None:
        """Save model using pickle."""
        with open(f"{filepath}_ordinal.pkl", 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.model_config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load model from pickle."""
        with open(f"{filepath}_ordinal.pkl", 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_config = data['config']
            self.is_trained = data['is_trained']