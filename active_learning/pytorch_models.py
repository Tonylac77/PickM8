"""
PyTorch model wrapper to implement MLModelBase interface.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Any, Optional, List
import copy

from .ml_base import MLModelBase

logger = logging.getLogger(__name__)


class FingerprintDataset(Dataset):
    """PyTorch dataset for fingerprint data."""
    
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class PyTorchModelWrapper(MLModelBase):
    """Base wrapper for PyTorch models to implement MLModelBase interface."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize PyTorch model wrapper.
        
        Args:
            model_config: Configuration dictionary
        """
        super().__init__(model_config)
        self._backend = 'pytorch'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        self.batch_size = model_config.get('batch_size', 128)
        self.learning_rate = model_config.get('learning_rate', 1e-4)
        self.weight_decay = model_config.get('weight_decay', 1e-2)
        self.n_epochs = model_config.get('n_epochs', 100)
        
        # Model will be created by subclasses
        self.model = None
        self.optimizer = None
        
    def _create_dataloader(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                          shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader."""
        dataset = FingerprintDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the PyTorch model."""
        if self.model is None:
            raise ValueError("Model must be initialized before training")
        
        self.model.to(self.device)
        self.model.train()
        
        # Create optimizer if not exists
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Create dataloader
        dataloader = self._create_dataloader(X, y, shuffle=True)
        
        # Training loop
        for epoch in range(self.n_epochs):
            epoch_losses = []
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self._compute_loss(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if epoch % 10 == 0:
                avg_loss = np.mean(epoch_losses)
                logger.debug(f"Epoch {epoch}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"Trained {self.model_type} model with PyTorch backend")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        predictions = []
        
        dataloader = self._create_dataloader(X, shuffle=False)
        
        with torch.no_grad():
            for batch_x in dataloader:
                if isinstance(batch_x, tuple):
                    batch_x = batch_x[0]
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        # For most PyTorch models, predict already returns probabilities
        return self.predict(X)
    
    def save(self, filepath: str) -> None:
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'config': self.model_config,
            'is_trained': self.is_trained
        }, f"{filepath}_pytorch.pt")
    
    def load(self, filepath: str) -> None:
        """Load model state."""
        checkpoint = torch.load(f"{filepath}_pytorch.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _compute_loss")
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Get uncertainty estimates - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_uncertainty")