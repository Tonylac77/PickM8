"""
Machine Learning module for PickM8.

This module provides a complete machine learning pipeline for molecular screening,
including model training, encoding strategies, and feature engineering.

The module is organized into:
- models: Base classes and specific model implementations
- encoding: Grade encoding/decoding strategies 
- training: Feature engineering and training utilities
- ml_models: Main training and prediction functions
"""

# Import main training functions
from .ml_models import (
    load_model_config,
    load_encoding_config,
    create_model,
    train_model,
    update_predictions,
    prepare_features_from_dataframe,
    encode_grades_for_training,
    determine_model_category,
    ensure_backward_compatibility
)

# Import model classes
from .models import (
    MLModelBase,
    SklearnModelWrapper,
    AutoPartyEnsemble,
    OrdinalRegressionWrapper
)

# Import encoding utilities
from .encoding import (
    SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION, VALID_ENCODING_TYPES,
    encode_grades_for_training,
    decode_predictions,
    get_ml_strategy,
    get_encoding_function,
    get_decoding_function,
    get_active_learning_ranking
)

# Import training utilities
from .training import (
    engineer_features_for_training,
    apply_feature_engineering_for_prediction,
    create_hybrid_fingerprint_features,
    remove_low_variance_features,
    calculate_feature_importance,
    normalize_features
)

__all__ = [
    # Main training functions
    'load_model_config',
    'load_encoding_config', 
    'create_model',
    'train_model',
    'update_predictions',
    'prepare_features_from_dataframe',
    'encode_grades_for_training',
    'determine_model_category',
    'ensure_backward_compatibility',
    
    # Model classes
    'MLModelBase',
    'SklearnModelWrapper',
    'AutoPartyEnsemble',
    'OrdinalRegressionWrapper',
    
    # Encoding constants
    'SEQUENTIAL', 'ONE_HOT', 'ORDINAL', 'ORDINAL_REGRESSION', 'VALID_ENCODING_TYPES',
    
    # Encoding functions
    'decode_predictions',
    'get_ml_strategy',
    'get_encoding_function',
    'get_decoding_function',
    'get_active_learning_ranking',
    
    # Training utilities
    'engineer_features_for_training',
    'apply_feature_engineering_for_prediction',
    'create_hybrid_fingerprint_features',
    'remove_low_variance_features',
    'calculate_feature_importance',
    'normalize_features'
]