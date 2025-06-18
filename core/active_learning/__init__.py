"""
Active learning modules for PickM8.
"""

from .features import prepare_features_from_dataframe, get_molecule_features
from .models import (
    create_ml_model, train_model_with_calibration, predict_with_uncertainty,
    save_model_and_encoders, load_model_and_encoders
)
from .selection import select_molecules_for_labeling, select_diverse_samples
from .encoding import encode_grades_for_training
from .utils import get_training_statistics, update_model_predictions
from .config import create_default_ml_config

__all__ = [
    'prepare_features_from_dataframe',
    'get_molecule_features',
    'create_ml_model',
    'train_model_with_calibration',
    'predict_with_uncertainty',
    'save_model_and_encoders',
    'load_model_and_encoders',
    'select_molecules_for_labeling',
    'select_diverse_samples',
    'encode_grades_for_training',
    'get_training_statistics',
    'update_model_predictions',
    'create_default_ml_config'
]