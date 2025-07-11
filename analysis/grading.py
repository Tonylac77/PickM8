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

def get_best_predicted_ungraded_molecule(
    df: pd.DataFrame, 
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[pd.Series]:
    """
    Get the best ungraded molecule based on ML predictions.
    
    Priority order: A > B > C > D
    Tiebreaker: score (based on score_direction in metadata)
    
    Args:
        df: Molecules DataFrame
        metadata: Session metadata containing score_direction
        
    Returns:
        Best ungraded molecule with prediction, or None if none exist
    """
    # Filter to ungraded molecules with predictions
    candidates = df[df['grade'].isna() & df['prediction'].notna()].copy()
    
    if len(candidates) == 0:
        return None
    
    # Create grade priority (A=1, B=2, C=3, D=4 for sorting)
    grade_priority = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    candidates['_grade_priority'] = candidates['prediction'].map(grade_priority).fillna(999)
    
    # Determine score sorting direction
    score_ascending = True  # Default: lower is better
    if metadata and metadata.get('score_direction') == 'Higher is better':
        score_ascending = False
    
    # Sort by grade priority first, then by score
    candidates = candidates.sort_values(
        by=['_grade_priority', 'score'],
        ascending=[True, score_ascending]
    )
    
    # Return the best candidate (without the temporary column)
    best_molecule = candidates.iloc[0].drop('_grade_priority')
    
    logger.debug(f"Selected molecule {best_molecule['name']} with prediction "
                f"{best_molecule['prediction']} and score {best_molecule['score']}")
    
    return best_molecule

def is_review_mode(df: pd.DataFrame) -> bool:
    """Check if all molecules have been graded (review mode)."""
    return df['grade'].notna().all()

def get_molecule_for_review(df: pd.DataFrame, index: int, metadata: Optional[Dict[str, Any]] = None) -> Optional[pd.Series]:
    """
    Get molecule at specific index for review mode.
    Molecules are ordered by grade (A>B>C>D) then by score.
    
    Args:
        df: Molecules DataFrame
        index: Index in the sorted list (0-based)
        metadata: Session metadata containing score_direction
        
    Returns:
        Molecule at the specified index, or None if index out of bounds
    """
    # Sort by grade first (A>B>C>D), then by score
    sorted_df = df.copy()
    
    # Create grade order for sorting
    grade_order = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    sorted_df['_grade_order'] = sorted_df['grade'].map(grade_order).fillna(999)
    
    # Determine score sorting
    score_ascending = True  # Default: lower is better
    if metadata and metadata.get('score_direction') == 'Higher is better':
        score_ascending = False
    
    # Sort by grade order, then score
    sorted_df = sorted_df.sort_values(
        by=['_grade_order', 'score'],
        ascending=[True, score_ascending]
    )
    sorted_df = sorted_df.drop(columns=['_grade_order'])
    
    # Return molecule at index
    if 0 <= index < len(sorted_df):
        return sorted_df.iloc[index]
    return None

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