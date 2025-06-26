"""Grading analysis functions."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

def add_grade(df: pd.DataFrame, molecule_id: int, grade: str) -> pd.DataFrame:
    """Add or update grade for a molecule."""
    df = df.copy()
    mask = df['id'] == molecule_id

    if not mask.any():
        logger.warning(f"Molecule ID {molecule_id} not found")
        return df

    df.loc[mask, 'grade'] = grade
    df.loc[mask, 'grade_timestamp'] = pd.Timestamp.now()

    logger.info(f"Added grade {grade} to molecule {molecule_id}")
    return df

def get_graded_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Get molecules that have been graded."""
    if 'grade' not in df.columns or len(df) == 0:
        return pd.DataFrame()
    return df[df['grade'].notna()].copy()

def get_ungraded_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Get molecules that haven't been graded."""
    if 'grade' not in df.columns or len(df) == 0:
        return df.copy()
    return df[df['grade'].isna()].copy()

def get_grading_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate grading statistics."""
    total = len(df)
    graded = df['grade'].notna().sum()

    stats = {
        'total_molecules': total,
        'graded_count': graded,
        'ungraded_count': total - graded,
        'grading_percentage': (graded / total * 100) if total > 0 else 0
    }

    # Grade distribution
    if graded > 0:
        grade_dist = df['grade'].value_counts().to_dict()
        stats['grade_distribution'] = grade_dist
        stats['most_common_grade'] = max(grade_dist, key=grade_dist.get)

    return stats

def has_trained_model(df: pd.DataFrame) -> bool:
    """Check if a trained model exists by looking for prediction data."""
    return 'prediction' in df.columns and df['prediction'].notna().any()

def filter_and_sort_molecules(
    df: pd.DataFrame,
    mode: str = 'all',
    sort_by: str = 'score',
    metadata: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Filter and sort molecules for display.

    Chain-of-Thought:
    - Simplified from original complex logic
    - Direct filtering without multiple service calls
    - Clear sort options
    - Respects score_direction from session metadata
    """
    if len(df) == 0:
        return df.copy()
    
    if mode == 'graded':
        filtered_df = get_graded_molecules(df)
    elif mode == 'ungraded':
        filtered_df = get_ungraded_molecules(df)
    else:
        filtered_df = df.copy()

    # Return early if no molecules after filtering
    if len(filtered_df) == 0:
        return filtered_df

    # Sort by specified column
    if sort_by == 'score' and 'score' in filtered_df.columns:
        # Determine sort direction based on score_direction in metadata
        score_direction = metadata.get('score_direction', 'Lower is better') if metadata else 'Lower is better'
        ascending = score_direction == 'Lower is better'
        filtered_df = filtered_df.sort_values('score', ascending=ascending)
    elif sort_by == 'grade_time' and 'grade_timestamp' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('grade_timestamp', ascending=False, na_position='last')
    elif sort_by == 'uncertainty' and 'prediction_uncertainty' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('prediction_uncertainty', ascending=False, na_position='last')
    elif sort_by == 'random':
        filtered_df = filtered_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    elif sort_by == 'best_prediction' and 'prediction' in filtered_df.columns:
        # Handle both new grade string predictions and legacy numeric predictions
        predictions = filtered_df['prediction'].dropna()
        
        if len(predictions) > 0:
            # Check if predictions are grade strings (new system)
            sample_pred = predictions.iloc[0]
            if isinstance(sample_pred, str) and sample_pred in ['A', 'B', 'C', 'D', 'F']:
                # New system: predictions are grade strings, sort alphabetically (A < B < C < D < F)
                filtered_df = filtered_df.sort_values('prediction', ascending=True, na_position='last')
            else:
                # Legacy system: predictions are numeric (lower values = better grades)
                try:
                    # Try to convert to numeric for proper sorting
                    filtered_df['_prediction_numeric'] = pd.to_numeric(filtered_df['prediction'], errors='coerce')
                    filtered_df = filtered_df.sort_values('_prediction_numeric', ascending=True, na_position='last')
                    filtered_df = filtered_df.drop(columns=['_prediction_numeric'])
                except:
                    # Fallback to string sorting
                    filtered_df = filtered_df.sort_values('prediction', ascending=True, na_position='last')

    return filtered_df.reset_index(drop=True)

def get_molecules_by_strategy(df: pd.DataFrame, strategy: str, metadata: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Get ungraded molecules sorted by the specified selection strategy.
    
    Args:
        df: Molecules DataFrame
        strategy: Selection strategy ('Random', 'Best Score', 'Best Predictions', 'Highest Uncertainty')
        metadata: Session metadata containing score_direction preference
        
    Returns:
        Filtered and sorted DataFrame of ungraded molecules
    """
    # Strategy mapping to sort parameters
    strategy_mapping = {
        'Random': 'random',
        'Best Score': 'score',
        'Best Predictions': 'best_prediction',
        'Highest Uncertainty': 'uncertainty'
    }
    
    sort_by = strategy_mapping.get(strategy, 'score')
    
    return filter_and_sort_molecules(df, mode='ungraded', sort_by=sort_by, metadata=metadata)