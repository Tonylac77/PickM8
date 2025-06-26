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

logger = logging.getLogger(__name__)

# Encoding type constants
SEQUENTIAL = "sequential"
NOMINAL = "nominal" 
ORDINAL = "ordinal"

VALID_ENCODING_TYPES = [SEQUENTIAL, NOMINAL, ORDINAL]


def encode_sequential(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades using sequential numbering (current approach).
    
    Args:
        grades: List of grade strings (e.g., ['A', 'B', 'C'])
        
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: 1D array of integers [0, 1, 2, ...]
        - label_mapping: Dict mapping grade strings to integers
    """
    unique_grades = sorted(list(set(grades)))
    label_mapping = {grade: idx for idx, grade in enumerate(unique_grades)}
    encoded_labels = np.array([label_mapping[grade] for grade in grades])
    
    logger.debug(f"Sequential encoding: {len(unique_grades)} unique grades -> {encoded_labels.shape}")
    return encoded_labels, label_mapping


def encode_nominal(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode grades using one-hot encoding (nominal).
    
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
    
    logger.debug(f"Nominal encoding: {len(unique_grades)} unique grades -> {encoded_labels.shape}")
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
    Decode sequential predictions back to grade strings.
    
    Args:
        predictions: 1D array of numeric predictions
        label_mapping: Original grade to integer mapping
        
    Returns:
        List of predicted grade strings
    """
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Round predictions to nearest integer and clip to valid range
    pred_ints = np.clip(np.round(predictions).astype(int), 0, len(reverse_mapping) - 1)
    decoded_grades = [reverse_mapping[pred_int] for pred_int in pred_ints]
    
    logger.debug(f"Sequential decoding: {len(predictions)} predictions -> {len(decoded_grades)} grades")
    return decoded_grades


def decode_nominal(predictions: np.ndarray, label_mapping: Dict[str, int]) -> List[str]:
    """
    Decode nominal (one-hot) predictions back to grade strings.
    
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
    
    logger.debug(f"Nominal decoding: {predictions.shape} predictions -> {len(decoded_grades)} grades")
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
        NOMINAL: encode_nominal,
        ORDINAL: encode_ordinal
    }
    
    if encoding_type not in encoding_functions:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Valid types: {VALID_ENCODING_TYPES}")
    
    return encoding_functions[encoding_type]


def get_decoding_function(encoding_type: str) -> Callable:
    """Get the appropriate decoding function for the specified type."""
    decoding_functions = {
        SEQUENTIAL: decode_sequential,
        NOMINAL: decode_nominal,
        ORDINAL: decode_ordinal
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
        SEQUENTIAL: "regression",  # Can treat as regression or classification
        NOMINAL: "multiclass",     # Multi-class classification
        ORDINAL: "ordinal"         # Ordinal regression (or multi-output classification)
    }
    
    if encoding_type not in ml_strategies:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Valid types: {VALID_ENCODING_TYPES}")
    
    return ml_strategies[encoding_type]


def calculate_uncertainty_score(predictions: np.ndarray, encoding_type: str) -> np.ndarray:
    """
    Calculate uncertainty scores for active learning based on encoding type.
    
    Args:
        predictions: Model predictions (1D for sequential, 2D for nominal/ordinal)
        encoding_type: Type of encoding used
        
    Returns:
        1D array of uncertainty scores (higher = more uncertain)
    """
    if encoding_type == SEQUENTIAL:
        # For sequential, use absolute distance from integer values as uncertainty
        rounded_preds = np.round(predictions)
        uncertainty = np.abs(predictions - rounded_preds)
        
    elif encoding_type == NOMINAL:
        # For nominal, use entropy of probability distribution
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs = predictions + epsilon
        probs = probs / np.sum(probs, axis=1, keepdims=True)  # Normalize
        entropy = -np.sum(probs * np.log(probs), axis=1)
        uncertainty = entropy
        
    elif encoding_type == ORDINAL:
        # For ordinal, use variance of cumulative distribution
        # Higher variance indicates more uncertainty
        uncertainty = np.var(predictions, axis=1)
        
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}. Valid types: {VALID_ENCODING_TYPES}")
    
    logger.debug(f"Calculated uncertainty scores for {encoding_type}: mean={np.mean(uncertainty):.3f}")
    return uncertainty


def get_active_learning_ranking(predictions: np.ndarray, encoding_type: str, 
                               strategy: str = "uncertainty") -> np.ndarray:
    """
    Get molecule ranking for active learning based on encoding type and strategy.
    
    Args:
        predictions: Model predictions
        encoding_type: Type of encoding used
        strategy: Active learning strategy ("uncertainty", "best_predictions")
        
    Returns:
        Array of indices sorted by ranking (best candidates first)
    """
    if strategy == "uncertainty":
        uncertainty_scores = calculate_uncertainty_score(predictions, encoding_type)
        # Sort by uncertainty descending (most uncertain first)
        ranking = np.argsort(-uncertainty_scores)
        
    elif strategy == "best_predictions":
        if encoding_type == SEQUENTIAL:
            # Sort by prediction ascending (best grades first: A=0, B=1, etc.)
            ranking = np.argsort(predictions)
            
        elif encoding_type == NOMINAL:
            # Sort by max probability descending (most confident predictions first)
            max_probs = np.max(predictions, axis=1)
            ranking = np.argsort(-max_probs)
            
        elif encoding_type == ORDINAL:
            # Convert ordinal to sequential-like score for ranking
            ordinal_scores = np.sum(predictions > 0.5, axis=1) - 1
            ranking = np.argsort(ordinal_scores)
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
            
    else:
        raise ValueError(f"Unknown active learning strategy: {strategy}")
    
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
    
    logger.info(f"Encoding {len(grades)} grades using {encoding_type} encoding")
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
    
    logger.info(f"Decoding {len(predictions)} predictions using {encoding_type} encoding")
    return decoding_function(predictions, label_mapping)