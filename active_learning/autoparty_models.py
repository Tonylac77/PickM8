"""
AutoParty ensemble model implementation for PickM8.
Based on the AutoParty paper architecture.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.utils import resample

from .pytorch_models import PyTorchModelWrapper

logger = logging.getLogger(__name__)


class AutoPartyNetwork(nn.Module):
    """Single neural network for AutoParty ensemble member."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 n_hidden_layers: int = 2, dropout: float = 0.2):
        """
        Initialize AutoParty network.
        
        Args:
            input_size: Size of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output classes
            n_hidden_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass through network."""
        return self.network(x)


class AutoPartyEnsemble(PyTorchModelWrapper):
    """
    AutoParty ensemble model implementation.
    Uses multiple neural networks with bootstrap sampling for uncertainty estimation.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize AutoParty ensemble.
        
        Args:
            model_config: Configuration dictionary with AutoParty-specific parameters
        """
        super().__init__(model_config)
        
        # AutoParty specific parameters
        self.committee_size = model_config.get('committee_size', 3)
        self.hidden_size = model_config.get('n_neurons', 1024)
        self.n_hidden_layers = model_config.get('hidden_layers', 2)
        self.dropout = model_config.get('dropout', 0.2)
        self.data_split = model_config.get('data_split', 'bootstrap')
        
        # Model components
        self.ensemble_members = []
        self.optimizers = []
        self.input_size = None
        self.output_size = None
        
        # Training data for bootstrap sampling
        self.training_indices = []
        
    def _initialize_ensemble(self, input_size: int, output_size: int):
        """Initialize ensemble members."""
        self.input_size = input_size
        self.output_size = output_size
        
        # Create ensemble members
        for i in range(self.committee_size):
            model = AutoPartyNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                output_size=output_size,
                n_hidden_layers=self.n_hidden_layers,
                dropout=self.dropout
            )
            model.to(self.device)
            
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            self.ensemble_members.append(model)
            self.optimizers.append(optimizer)
    
    def _create_bootstrap_indices(self, n_samples: int) -> List[np.ndarray]:
        """Create bootstrap indices for each ensemble member."""
        indices = []
        
        if self.data_split == 'bootstrap':
            # Sample with replacement
            for _ in range(self.committee_size):
                idx = np.random.choice(n_samples, size=n_samples, replace=True)
                indices.append(idx)
        else:
            # Split data without replacement
            shuffled_idx = np.random.permutation(n_samples)
            step = n_samples // self.committee_size
            for i in range(self.committee_size):
                start = i * step
                end = (i + 1) * step if i < self.committee_size - 1 else n_samples
                indices.append(shuffled_idx[start:end])
        
        return indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ensemble."""
        n_samples, n_features = X.shape
        
        # Initialize ensemble if needed
        if not self.ensemble_members:
            # Determine output size from y
            if len(y.shape) == 1:
                output_size = len(np.unique(y))
            else:
                output_size = y.shape[1]
            
            self._initialize_ensemble(n_features, output_size)
        
        # Create bootstrap indices
        self.training_indices = self._create_bootstrap_indices(n_samples)
        
        # Train each ensemble member
        for i, (model, optimizer, indices) in enumerate(
            zip(self.ensemble_members, self.optimizers, self.training_indices)
        ):
            logger.debug(f"Training ensemble member {i+1}/{self.committee_size}")
            
            # Get bootstrap sample
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Create dataloader
            dataloader = self._create_dataloader(X_boot, y_boot, shuffle=True)
            
            model.train()
            
            # Training loop
            for epoch in range(self.n_epochs):
                epoch_losses = []
                
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_x)
                    
                    # Use sigmoid for ordinal encoding (cumulative)
                    if self.model_config.get('output_type') == 'ordinal':
                        outputs = torch.sigmoid(outputs)
                        loss = F.binary_cross_entropy(outputs, batch_y)
                    else:
                        # Use softmax for one-hot encoding
                        outputs = F.log_softmax(outputs, dim=1)
                        loss = F.nll_loss(outputs, batch_y.argmax(dim=1))
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                if epoch % 20 == 0:
                    avg_loss = np.mean(epoch_losses)
                    logger.debug(f"Member {i+1}, Epoch {epoch}/{self.n_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info(f"Trained AutoParty ensemble with {self.committee_size} members")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble average."""
        all_predictions = self._get_all_predictions(X)
        
        # Average predictions across ensemble
        mean_predictions = np.mean(all_predictions, axis=0)
        
        # Apply appropriate activation
        if self.model_config.get('output_type') == 'ordinal':
            # For ordinal, predictions are already sigmoid probabilities
            return mean_predictions
        else:
            # For one-hot, convert to probabilities
            return self._softmax(mean_predictions)
    
    def get_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get uncertainty estimates using ensemble variance.
        
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        all_predictions = self._get_all_predictions(X)
        
        # Calculate variance across ensemble
        if self.model_config.get('output_type') == 'ordinal':
            # For ordinal, use average variance across all outputs
            variance = np.var(all_predictions, axis=0)
            uncertainty = np.mean(variance, axis=1)
        else:
            # For one-hot, use entropy of averaged predictions
            mean_predictions = np.mean(all_predictions, axis=0)
            probs = self._softmax(mean_predictions)
            # Calculate entropy
            epsilon = 1e-10
            entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
            # Normalize to [0, 1]
            max_entropy = np.log(probs.shape[1])
            uncertainty = entropy / max_entropy
        
        return uncertainty
    
    def _get_all_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all ensemble members."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        all_predictions = []
        
        for model in self.ensemble_members:
            model.eval()
            predictions = []
            
            dataloader = self._create_dataloader(X, shuffle=False)
            
            with torch.no_grad():
                for batch_x in dataloader:
                    if isinstance(batch_x, tuple):
                        batch_x = batch_x[0]
                    batch_x = batch_x.to(self.device)
                    
                    outputs = model(batch_x)
                    
                    if self.model_config.get('output_type') == 'ordinal':
                        outputs = torch.sigmoid(outputs)
                    
                    predictions.append(outputs.cpu().numpy())
            
            member_predictions = np.concatenate(predictions, axis=0)
            all_predictions.append(member_predictions)
        
        return np.stack(all_predictions, axis=0)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def save(self, filepath: str) -> None:
        """Save ensemble model."""
        state = {
            'ensemble_states': [model.state_dict() for model in self.ensemble_members],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
            'config': self.model_config,
            'is_trained': self.is_trained,
            'input_size': self.input_size,
            'output_size': self.output_size
        }
        torch.save(state, f"{filepath}_autoparty.pt")
    
    def load(self, filepath: str) -> None:
        """Load ensemble model."""
        checkpoint = torch.load(f"{filepath}_autoparty.pt", map_location=self.device)
        
        # Reinitialize ensemble with correct sizes
        self._initialize_ensemble(
            checkpoint['input_size'], 
            checkpoint['output_size']
        )
        
        # Load states
        for model, state_dict in zip(self.ensemble_members, checkpoint['ensemble_states']):
            model.load_state_dict(state_dict)
        
        for opt, state_dict in zip(self.optimizers, checkpoint['optimizer_states']):
            opt.load_state_dict(state_dict)
        
        self.model_config = checkpoint['config']
        self.is_trained = checkpoint['is_trained']
        self.input_size = checkpoint['input_size']
        self.output_size = checkpoint['output_size']