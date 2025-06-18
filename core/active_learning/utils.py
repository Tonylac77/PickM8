"""
Utilities for active learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_training_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about training data.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary with training statistics
    """
    graded_df = df[df['grade'].notna()]
    
    if len(graded_df) == 0:
        return {"total_molecules": len(df), "graded_molecules": 0}
    
    grade_counts = graded_df['grade'].value_counts().to_dict()
    
    stats = {
        "total_molecules": len(df),
        "graded_molecules": len(graded_df),
        "ungraded_molecules": len(df) - len(graded_df),
        "grade_distribution": grade_counts,
        "grading_percentage": (len(graded_df) / len(df)) * 100,
        "most_common_grade": graded_df['grade'].mode().iloc[0] if len(graded_df) > 0 else None,
        "least_common_grade": min(grade_counts.keys(), key=grade_counts.get) if grade_counts else None
    }
    
    return stats


def update_model_predictions(df: pd.DataFrame, predictions: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    Update DataFrame with ML predictions and uncertainties.
    
    Args:
        df: Molecules DataFrame  
        predictions: Dict mapping molecule_id to prediction dict
        
    Returns:
        Updated DataFrame
    """
    df = df.copy()
    timestamp = pd.Timestamp.now()
    
    # Ensure prediction column can handle string values
    if 'prediction' in df.columns:
        df['prediction'] = df['prediction'].astype('object')
    
    for mol_id, pred_data in predictions.items():
        mask = df['id'] == mol_id
        if mask.any():
            prediction_value = pred_data.get('prediction')
            uncertainty_value = pred_data.get('uncertainty', np.nan)
            
            # Handle both string and numeric predictions
            if prediction_value is not None:
                df.loc[mask, 'prediction'] = prediction_value
            else:
                df.loc[mask, 'prediction'] = None
                
            df.loc[mask, 'prediction_uncertainty'] = uncertainty_value
            df.loc[mask, 'prediction_timestamp'] = timestamp
    
    logger.info(f"Updated ML predictions for {len(predictions)} molecules")
    
    return df