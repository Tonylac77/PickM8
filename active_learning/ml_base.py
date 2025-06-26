"""
Abstract base class for ML models to support both sklearn and PyTorch backends.
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MLModelBase(ABC):
    """Abstract base class for all ML models in PickM8."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize base model.
        
        Args:
            model_config: Configuration dictionary containing model parameters
        """
        self.model_config = model_config
        self.model_type = model_config.get('model_type', 'Unknown')
        self.is_trained = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels (encoded)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (format depends on encoding type)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        pass
    
    @abstractmethod
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get uncertainty estimates for predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save model to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load model from file."""
        pass
    
    @property
    def backend(self) -> str:
        """Return the backend type (sklearn or pytorch)."""
        return self._backend