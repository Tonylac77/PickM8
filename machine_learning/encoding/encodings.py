"""
Grade encoding/decoding functions for different encoding strategies.

This module provides three encoding strategies:
1. Sequential: A=0, B=1, C=2, etc. (current approach)
2. Nominal: One-hot encoding A=[1,0,0,0,0], B=[0,1,0,0,0], etc.
3. Ordinal: Cumulative encoding A=[1,0,0,0,0], B=[1,1,0,0,0], etc.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Union
import logging
from sklearn.preprocessing import KBinsDiscretizer

logger = logging.getLogger(__name__)

# Encoding type constants
SEQUENTIAL = "sequential"
ONE_HOT = "one_hot" 
ORDINAL = "ordinal"
ORDINAL_REGRESSION = "ordinal_regression"

VALID_ENCODING_TYPES = [SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION]


def encode_sequential(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades using sequential numbering with reversed mapping (A=3, B=2, C=1, D=0).
    
    Args:
        grades: List of grade strings (e.g., ['A', 'B', 'C'])
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: 1D array of integers [3, 2, 1, 0] for [A, B, C, D]
        - label_mapping: Dict mapping grade strings to integers
    """
    unique_grades = sorted(list(set(grades)))
    # Reverse the mapping so A gets the highest value
    label_mapping = {grade: len(unique_grades) - 1 - idx for idx, grade in enumerate(unique_grades)}
    encoded_labels = np.array([label_mapping[grade] for grade in grades])
    
    logger.debug(f"Sequential encoding (reversed): {len(unique_grades)} unique grades -> {encoded_labels.shape}")
    return encoded_labels, label_mapping


def encode_one_hot(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades using one-hot encoding.
    
    Args:
        grades: List of grade strings (e.g., ['A', 'B', 'C'])
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: 2D array of one-hot vectors
        - label_mapping: Dict mapping grade strings to integers (for reference)
    """
    unique_grades = sorted(list(set(grades)))
    label_mapping = {grade: idx for idx, grade in enumerate(unique_grades)}
    num_classes = len(unique_grades)
    
    # Create one-hot encoded matrix
    encoded_labels = np.zeros((len(grades), num_classes))
    for i, grade in enumerate(grades):
        class_idx = label_mapping[grade]
        encoded_labels[i, class_idx] = 1
    
    logger.debug(f"One-hot encoding: {len(unique_grades)} unique grades -> {encoded_labels.shape}")
    return encoded_labels, label_mapping


def encode_ordinal_regression(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades using continuous values between 0 and 100 for regression.
    
    Args:
        grades: List of grade strings (e.g., ['A', 'B', 'C'])
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: 1D array of continuous values [0-100]
        - label_mapping: Dict mapping grade strings to continuous ranges
    """
    unique_grades = sorted(list(set(grades)))
    
    # Define grade to continuous value mapping
    # D: 0-25, C: 25-50, B: 50-75, A: 75-100
    grade_to_range = {
        'D': (0, 25),
        'C': (25, 50), 
        'B': (50, 75),
        'A': (75, 100)
    }
    
    # Create mapping with midpoint values for training
    label_mapping = {}
    for grade in unique_grades:
        if grade in grade_to_range:
            low, high = grade_to_range[grade]
            label_mapping[grade] = (low + high) / 2.0
        else:
            # Fallback for unexpected grades
            idx = ord(grade) - ord('A')
            label_mapping[grade] = 100 - (idx * 25)
    
    encoded_labels = np.array([label_mapping[grade] for grade in grades])
    
    logger.debug(f"Ordinal regression encoding: {len(unique_grades)} unique grades -> {encoded_labels.shape}")
    return encoded_labels, label_mapping


def encode_ordinal(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades using cumulative ordinal encoding.
    
    Args:
        grades: List of grade strings (e.g., ['A', 'B', 'C'])
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: 2D array of cumulative vectors
        - label_mapping: Dict mapping grade strings to integers (for reference)
    """
    unique_grades = sorted(list(set(grades)))
    label_mapping = {grade: idx for idx, grade in enumerate(unique_grades)}
    num_classes = len(unique_grades)
    
    # Create ordinal encoded matrix
    encoded_labels = np.zeros((len(grades), num_classes))
    for i, grade in enumerate(grades):
        class_idx = label_mapping[grade]
        # Fill cumulative pattern: A=[1,0,0], B=[1,1,0], C=[1,1,1]
        encoded_labels[i, :class_idx + 1] = 1
    
    logger.debug(f"Ordinal encoding: {len(unique_grades)} unique grades -> {encoded_labels.shape}")
    return encoded_labels, label_mapping


def decode_sequential(predictions: np.ndarray, label_mapping: Dict[str, int]) -> List[str]:
    """
    Decode sequential predictions back to grade strings (handles reversed mapping).
    
    Args:
        predictions: 1D array of numeric predictions
        label_mapping: Original grade to integer mapping
        
    Returns:
        List of predicted grade strings
    """
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Round predictions to nearest integer and clip to valid range
    max_val = max(label_mapping.values()) if label_mapping else 3
    pred_ints = np.clip(np.round(predictions).astype(int), 0, max_val)
    decoded_grades = [reverse_mapping.get(pred_int, 'D') for pred_int in pred_ints]
    
    logger.debug(f"Sequential decoding: {len(predictions)} predictions -> {len(decoded_grades)} grades")
    return decoded_grades


def decode_one_hot(predictions: np.ndarray, label_mapping: Dict[str, int]) -> List[str]:
    """
    Decode one-hot predictions back to grade strings.
    
    Args:
        predictions: 2D array of probability vectors
        label_mapping: Original grade to integer mapping
        
    Returns:
        List of predicted grade strings (most likely class)
    """
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Get class with highest probability
    pred_classes = np.argmax(predictions, axis=1)
    decoded_grades = [reverse_mapping[pred_class] for pred_class in pred_classes]
    
    logger.debug(f"One-hot decoding: {predictions.shape} predictions -> {len(decoded_grades)} grades")
    return decoded_grades


def decode_ordinal_regression(predictions: np.ndarray, label_mapping: Dict[str, int]) -> List[str]:
    """
    Decode continuous predictions back to grade strings using binning.
    
    Args:
        predictions: 1D array of continuous predictions [0-100]
        label_mapping: Original grade to continuous value mapping
        
    Returns:
        List of predicted grade strings
    """
    # Define bins for continuous values
    # 0-25→D, 25-50→C, 50-75→B, 75-100→A
    bins = [0, 25, 50, 75, 100]
    labels = ['D', 'C', 'B', 'A']
    
    # Clip predictions to valid range
    predictions = np.clip(predictions, 0, 100)
    
    # Use KBinsDiscretizer approach but manually implement binning
    decoded_grades = []
    for pred in predictions:
        if pred < 25:
            decoded_grades.append('D')
        elif pred < 50:
            decoded_grades.append('C')
        elif pred < 75:
            decoded_grades.append('B')
        else:
            decoded_grades.append('A')
    
    logger.debug(f"Ordinal regression decoding: {len(predictions)} predictions -> {len(decoded_grades)} grades")
    return decoded_grades


def decode_ordinal(predictions: np.ndarray, label_mapping: Dict[str, int]) -> List[str]:
    """
    Decode ordinal predictions back to grade strings.
    
    Args:
        predictions: 2D array of cumulative probability vectors
        label_mapping: Original grade to integer mapping
        
    Returns:
        List of predicted grade strings
    """
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Convert cumulative probabilities to class predictions
    # Use threshold of 0.5 to determine cutoff point
    pred_classes = np.sum(predictions > 0.5, axis=1) - 1
    pred_classes = np.clip(pred_classes, 0, len(reverse_mapping) - 1)
    
    decoded_grades = [reverse_mapping[pred_class] for pred_class in pred_classes]
    
    logger.debug(f"Ordinal decoding: {predictions.shape} predictions -> {len(decoded_grades)} grades")
    return decoded_grades


def get_encoding_function(encoding_type: str) -> Callable:
    """Get the appropriate encoding function for the specified type."""
    encoding_functions = {
        SEQUENTIAL: encode_sequential,
        ONE_HOT: encode_one_hot,
        ORDINAL: encode_ordinal,
        ORDINAL_REGRESSION: encode_ordinal_regression
    }
    
    if encoding_type not in encoding_functions:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Valid types: {VALID_ENCODING_TYPES}")
    
    return encoding_functions[encoding_type]


def get_decoding_function(encoding_type: str) -> Callable:
    """Get the appropriate decoding function for the specified type."""
    decoding_functions = {
        SEQUENTIAL: decode_sequential,
        ONE_HOT: decode_one_hot,
        ORDINAL: decode_ordinal,
        ORDINAL_REGRESSION: decode_ordinal_regression
    }
    
    if encoding_type not in decoding_functions:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Valid types: {VALID_ENCODING_TYPES}")
    
    return decoding_functions[encoding_type]


def get_ml_strategy(encoding_type: str) -> str:
    """
    Get the recommended ML strategy for each encoding type.
    
    Returns:
        String indicating the ML approach to use
    """
    ml_strategies = {
        SEQUENTIAL: "classification",  # Can treat as regression or classification
        ONE_HOT: "multiclass",         # Multi-class classification
        ORDINAL: "ordinal",            # Ordinal regression (or multi-output classification)
        ORDINAL_REGRESSION: "regression"  # Continuous regression
    }
    
    if encoding_type not in ml_strategies:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Valid types: {VALID_ENCODING_TYPES}")
    
    return ml_strategies[encoding_type]




def get_active_learning_ranking(predictions: np.ndarray, encoding_type: str, 
                               strategy: str = "best_predictions") -> np.ndarray:
    """
    Get molecule ranking for active learning based on encoding type and strategy.
    
    Args:
        predictions: Model predictions
        encoding_type: Type of encoding used
        strategy: Active learning strategy ("best_predictions")
        
    Returns:
        Array of indices sorted by ranking (best candidates first)
    """
    if strategy == "best_predictions":
        if encoding_type == SEQUENTIAL:
            # Sort by prediction descending (best grades first: A=3, B=2, etc.)
            ranking = np.argsort(-predictions)
            
        elif encoding_type == ONE_HOT:
            # Sort by max probability descending (most confident predictions first)
            max_probs = np.max(predictions, axis=1)
            ranking = np.argsort(-max_probs)
            
        elif encoding_type == ORDINAL:
            # Convert ordinal to sequential-like score for ranking
            ordinal_scores = np.sum(predictions > 0.5, axis=1) - 1
            ranking = np.argsort(-ordinal_scores)
            
        elif encoding_type == ORDINAL_REGRESSION:
            # Sort by continuous value descending (higher values = better grades)
            ranking = np.argsort(-predictions)
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
            
    else:
        raise ValueError(f"Unknown active learning strategy: {strategy}. Valid: 'best_predictions'")
    
    logger.debug(f"Generated {strategy} ranking for {encoding_type}: {len(ranking)} molecules")
    return ranking


def encode_grades_for_training(df: pd.DataFrame, encoding_type: str = SEQUENTIAL) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Main function to encode grades for ML training using specified encoding type.
    
    Args:
        df: DataFrame with 'grade' column
        encoding_type: Type of encoding to use
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
    """
    graded_df = df[df['grade'].notna()].copy()
    
    if len(graded_df) == 0:
        logger.warning("No graded molecules found for encoding")
        return np.array([]), {}
    
    grades = graded_df['grade'].tolist()
    encoding_function = get_encoding_function(encoding_type)
    return encoding_function(grades)


def decode_predictions(predictions: np.ndarray, label_mapping: Dict[str, int], 
                      encoding_type: str = SEQUENTIAL) -> List[str]:
    """
    Main function to decode predictions back to grade strings.
    
    Args:
        predictions: Model predictions  
        label_mapping: Original grade to integer mapping
        encoding_type: Type of encoding used
        
    Returns:
        List of predicted grade strings
    """
    if len(predictions) == 0:
        return []
    
    decoding_function = get_decoding_function(encoding_type)
    return decoding_function(predictions, label_mapping)