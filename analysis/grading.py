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
    elif sort_by == 'random':
        filtered_df = filtered_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    elif sort_by == 'best_prediction' and 'prediction' in filtered_df.columns:
        # Handle both new grade string predictions and legacy numeric predictions
        predictions = filtered_df['prediction'].dropna()
        
        if len(predictions) > 0:
            # Check if predictions are grade strings (new system)
            sample_pred = predictions.iloc[0]
            if isinstance(sample_pred, str) and sample_pred in ['A', 'B', 'C', 'D']:
                # New system: predictions are grade strings, sort alphabetically (A < B < C < D)
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
        strategy: Selection strategy ('Random', 'Best Score', 'Best Predictions')
        metadata: Session metadata containing score_direction preference
        
    Returns:
        Filtered and sorted DataFrame of ungraded molecules
    """
    # Strategy mapping to sort parameters
    strategy_mapping = {
        'Random': 'random',
        'Best Score': 'score',
        'Best Predictions': 'best_prediction',
    }
    
    sort_by = strategy_mapping.get(strategy, 'score')
    
    return filter_and_sort_molecules(df, mode='ungraded', sort_by=sort_by, metadata=metadata)

def get_best_ungraded_molecule(df: pd.DataFrame, strategy: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[pd.Series]:
    """
    Get the best ungraded molecule based on the specified selection strategy.
    
    This function provides stateless molecule selection - always returns the top-ranked
    ungraded molecule according to the strategy, eliminating navigation state issues.
    
    Args:
        df: Molecules DataFrame
        strategy: Selection strategy ('Random', 'Best Score', 'Best Predictions')
        metadata: Session metadata containing score_direction preference
        
    Returns:
        Best ungraded molecule as Series, or None if no ungraded molecules exist
    """
    filtered_df = get_molecules_by_strategy(df, strategy, metadata)
    return filtered_df.iloc[0] if len(filtered_df) > 0 else None

def reset_all_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reset all grades and related data to None, preserving molecular fingerprints and pose quality.
    
    This function clears:
    - Manual grades and timestamps
    - ML predictions
    - All prediction-related metadata
    
    Preserves:
    - Molecular fingerprints and structure data
    - Pose quality metrics (clashes, strain energy)
    - Protein-ligand interaction data
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        DataFrame with all grades and predictions cleared
    """
    df = df.copy()
    
    # Track what we're resetting for better logging
    grades_cleared = df['grade'].notna().sum() if 'grade' in df.columns else 0
    predictions_cleared = df['prediction'].notna().sum() if 'prediction' in df.columns else 0
    
    # Clear grading data
    if 'grade' in df.columns:
        df['grade'] = None
    if 'grade_timestamp' in df.columns:
        df['grade_timestamp'] = None
    
    # Clear prediction data - handle all prediction-related columns robustly
    prediction_columns = ['prediction', 'prediction_timestamp']
    for col in prediction_columns:
        if col in df.columns:
            df[col] = None
    
    reset_count = len(df)
    logger.info(f"Reset operation completed: {reset_count} molecules processed")
    if grades_cleared > 0:
        logger.info(f"  - Cleared {grades_cleared} manual grades")
    if predictions_cleared > 0:
        logger.info(f"  - Cleared {predictions_cleared} ML predictions")
    
    return df

def cleanup_model_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up model-related metadata while preserving essential session data.
    
    Removes:
    - Model configuration (config.model_config)
    - Label mapping from training
    - Any other ML-specific metadata
    
    Preserves:
    - Session identification and creation info
    - Molecular data processing configuration
    - Protein and ligand file information
    
    Args:
        metadata: Session metadata dictionary
        
    Returns:
        Cleaned metadata dictionary
    """
    metadata = metadata.copy()
    
    # Remove model-specific configuration
    if 'config' in metadata and isinstance(metadata['config'], dict):
        config = metadata['config'].copy()
        if 'model_config' in config:
            del config['model_config']
            logger.info("Removed model_config from session metadata")
        metadata['config'] = config
    
    # Remove label mapping
    if 'label_mapping' in metadata:
        del metadata['label_mapping']
        logger.info("Removed label_mapping from session metadata")
    
    # Log what was preserved
    essential_keys = ['session_id', 'protein_name', 'num_molecules', 'score_label', 'score_direction', 'created_date']
    preserved_keys = [key for key in essential_keys if key in metadata]
    logger.info(f"Preserved essential metadata keys: {preserved_keys}")
    
    return metadata