"""
Encoding strategies module for PickM8.

This module contains different encoding strategies for grade labels.
"""

from .encodings import (
    # Encoding type constants
    SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION, VALID_ENCODING_TYPES,
    
    # Main encoding/decoding functions
    encode_grades_for_training,
    decode_predictions,
    
    # Individual encoding functions
    encode_sequential,
    encode_one_hot, 
    encode_ordinal,
    encode_ordinal_regression,
    
    # Individual decoding functions
    decode_sequential,
    decode_one_hot,
    decode_ordinal,
    decode_ordinal_regression,
    
    # Strategy helpers
    get_encoding_function,
    get_decoding_function,
    get_ml_strategy,
    
    # Active learning utilities
    get_active_learning_ranking
)

__all__ = [
    # Constants
    'SEQUENTIAL', 'ONE_HOT', 'ORDINAL', 'ORDINAL_REGRESSION', 'VALID_ENCODING_TYPES',
    
    # Main functions
    'encode_grades_for_training',
    'decode_predictions',
    
    # Individual encoding functions
    'encode_sequential',
    'encode_one_hot', 
    'encode_ordinal',
    'encode_ordinal_regression',
    
    # Individual decoding functions
    'decode_sequential',
    'decode_one_hot',
    'decode_ordinal',
    'decode_ordinal_regression',
    
    # Helper functions
    'get_encoding_function',
    'get_decoding_function', 
    'get_ml_strategy',
    
    # Active learning
    'get_active_learning_ranking'
]