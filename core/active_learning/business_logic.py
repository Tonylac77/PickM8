"""
Business logic functions for Active Learning operations using functional programming approach.
All functions are pure - no side effects, return results.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
from datetime import datetime

from .models import train_model_with_calibration
from .features import prepare_features_from_dataframe
from .selection import select_molecules_for_labeling
from .utils import get_training_statistics, update_model_predictions
from .encoding import encode_grades_for_training
from core.grading import get_graded_molecules

logger = logging.getLogger(__name__)


def execute_model_training_pipeline(
    df: pd.DataFrame,
    session_dir: str
) -> Dict[str, Any]:
    """
    Execute complete model training pipeline.
    Pure function orchestrating ML training business logic.
    
    Args:
        df: Molecules DataFrame
        session_dir: Session directory path
    
    Returns:
        Dictionary with training result and updated DataFrame
    """
    try:
        # Get graded molecules
        graded_df = get_graded_molecules(df)
        
        if len(graded_df) < 3:
            return {
                'success': False,
                'error': f"Need at least 3 graded molecules to train model. Currently have {len(graded_df)}",
                'error_type': 'insufficient_data',
                'graded_count': len(graded_df)
            }
        
        logger.info("Starting model training process...")
        
        # Prepare features and labels
        logger.info("Preparing features from graded molecules...")
        features, mol_ids = prepare_features_from_dataframe(graded_df)
        logger.info(f"Prepared features for {len(mol_ids)} molecules, feature shape: {features.shape if len(features) > 0 else 'empty'}")
        
        if len(features) == 0:
            return {
                'success': False,
                'error': "No valid features found for model training",
                'error_type': 'no_features',
                'graded_count': len(graded_df)
            }
        
        # Encode grades for training
        logger.info("Encoding grades for machine learning...")
        grades = graded_df.loc[mol_ids, 'grade'].values
        y_train = encode_grades_for_training(grades)
        logger.info(f"Encoded {len(y_train)} grades for training")
        
        # Train model
        logger.info("Training ML model with calibration...")
        model_result = train_model_with_calibration(features, y_train)
        
        if not model_result['success']:
            return {
                'success': False,
                'error': f"Model training failed: {model_result['error']}",
                'error_type': 'training_failed',
                'model_result': model_result
            }
        
        model = model_result['model']
        training_stats = get_training_statistics(model_result)
        logger.info(f"Model training completed. Stats: {training_stats}")
        
        # Generate predictions for all molecules
        logger.info("Generating predictions for all molecules...")
        prediction_result = update_model_predictions(df, model)
        
        if not prediction_result['success']:
            return {
                'success': False,
                'error': f"Prediction generation failed: {prediction_result['error']}",
                'error_type': 'prediction_failed',
                'prediction_result': prediction_result
            }
        
        updated_df = prediction_result['updated_df']
        prediction_stats = prediction_result['stats']
        logger.info(f"Predictions updated. Stats: {prediction_stats}")
        
        return {
            'success': True,
            'updated_df': updated_df,
            'training_stats': training_stats,
            'prediction_stats': prediction_stats,
            'graded_count': len(graded_df),
            'total_predictions': len(updated_df)
        }
        
    except Exception as e:
        logger.error(f"Model training pipeline error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f"Unexpected training error: {str(e)}",
            'error_type': 'unexpected',
            'exception_details': str(e)
        }


def execute_molecule_suggestion(
    df: pd.DataFrame,
    strategy: str,
    num_suggestions: int = 5
) -> Dict[str, Any]:
    """
    Execute molecule suggestion based on strategy.
    Pure function for molecule selection logic.
    
    Args:
        df: Molecules DataFrame
        strategy: Selection strategy ('random', 'uncertainty', 'diverse')
        num_suggestions: Number of molecules to suggest
    
    Returns:
        Dictionary with suggested molecules and selection info
    """
    try:
        # Get ungraded molecules
        ungraded_df = df[df['grade'].isna()]
        
        if len(ungraded_df) == 0:
            return {
                'success': True,
                'suggested_molecules': [],
                'strategy_used': strategy,
                'ungraded_count': 0,
                'message': 'All molecules have been graded'
            }
        
        # Check if ML strategies are available
        has_predictions = 'prediction_uncertainty' in df.columns and df['prediction_uncertainty'].notna().any()
        
        # Fallback to random if ML strategy requested but not available
        if strategy in ['uncertainty', 'diverse'] and not has_predictions:
            strategy = 'random'
        
        # Select molecules based on strategy
        if strategy == 'uncertainty' and has_predictions:
            suggested_molecules = select_molecules_for_labeling(
                ungraded_df, method='uncertainty', n_molecules=num_suggestions
            )
        elif strategy == 'diverse' and has_predictions:
            suggested_molecules = select_molecules_for_labeling(
                ungraded_df, method='diverse', n_molecules=num_suggestions
            )
        else:
            # Random selection
            sample_size = min(num_suggestions, len(ungraded_df))
            suggested_molecules = ungraded_df.sample(sample_size)['id'].tolist()
        
        return {
            'success': True,
            'suggested_molecules': suggested_molecules,
            'strategy_used': strategy,
            'ungraded_count': len(ungraded_df),
            'num_suggestions': len(suggested_molecules),
            'has_predictions': has_predictions
        }
        
    except Exception as e:
        logger.error(f"Molecule suggestion error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f"Suggestion generation failed: {str(e)}",
            'suggested_molecules': [],
            'strategy_used': strategy
        }


def validate_grading_action(
    df: pd.DataFrame,
    mol_id: int,
    grade: str
) -> Dict[str, Any]:
    """
    Validate molecule grading action.
    Pure function - returns validation result.
    
    Args:
        df: Molecules DataFrame
        mol_id: Molecule ID to grade
        grade: Grade to assign ('A', 'B', 'C', 'D', 'F')
    
    Returns:
        Dictionary with validation results
    """
    # Check if molecule exists
    mol_exists = mol_id in df['id'].values
    if not mol_exists:
        return {
            'is_valid': False,
            'error': f"Molecule with ID {mol_id} not found",
            'error_type': 'molecule_not_found'
        }
    
    # Validate grade
    valid_grades = ['A', 'B', 'C', 'D', 'F']
    if grade not in valid_grades:
        return {
            'is_valid': False,
            'error': f"Invalid grade '{grade}'. Must be one of: {valid_grades}",
            'error_type': 'invalid_grade'
        }
    
    # Get current grade
    mol_row = df[df['id'] == mol_id].iloc[0]
    current_grade = mol_row.get('grade')
    has_existing_grade = pd.notna(current_grade)
    
    return {
        'is_valid': True,
        'mol_id': mol_id,
        'new_grade': grade,
        'current_grade': current_grade,
        'has_existing_grade': has_existing_grade,
        'is_grade_change': has_existing_grade and current_grade != grade,
        'molecule_name': mol_row.get('name', f"Molecule {mol_id}")
    }


def calculate_grading_progress(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate grading progress statistics.
    Pure function - returns progress data.
    
    Args:
        df: Molecules DataFrame
    
    Returns:
        Dictionary containing progress statistics
    """
    total_molecules = len(df)
    graded_count = df['grade'].notna().sum()
    ungraded_count = df['grade'].isna().sum()
    
    # Grade distribution
    grade_counts = df['grade'].value_counts().to_dict()
    
    # Progress percentage
    progress_percentage = (graded_count / total_molecules * 100) if total_molecules > 0 else 0
    
    # ML training readiness
    ml_ready = graded_count >= 3
    ml_status = "Ready" if ml_ready else f"Need {3 - graded_count} more grades"
    
    return {
        'total_molecules': total_molecules,
        'graded_count': graded_count,
        'ungraded_count': ungraded_count,
        'progress_percentage': progress_percentage,
        'grade_distribution': grade_counts,
        'ml_ready': ml_ready,
        'ml_status': ml_status,
        'next_milestone': _get_next_milestone(graded_count)
    }


def prepare_molecule_navigation(
    df: pd.DataFrame,
    current_idx: int,
    filter_ungraded: bool = False
) -> Dict[str, Any]:
    """
    Prepare molecule navigation data.
    Pure function - returns navigation state.
    
    Args:
        df: Molecules DataFrame
        current_idx: Current molecule index
        filter_ungraded: Whether to filter to ungraded molecules only
    
    Returns:
        Dictionary containing navigation data
    """
    # Filter DataFrame if requested
    if filter_ungraded:
        display_df = df[df['grade'].isna()].reset_index(drop=True)
        # Map back to original indices
        original_indices = df[df['grade'].isna()].index.tolist()
    else:
        display_df = df.copy()
        original_indices = list(range(len(df)))
    
    # Validate current index
    if current_idx >= len(display_df):
        current_idx = 0
    elif current_idx < 0:
        current_idx = len(display_df) - 1 if len(display_df) > 0 else 0
    
    # Calculate navigation
    total_count = len(display_df)
    prev_idx = (current_idx - 1) % total_count if total_count > 0 else 0
    next_idx = (current_idx + 1) % total_count if total_count > 0 else 0
    
    # Get original DataFrame index
    original_idx = original_indices[current_idx] if current_idx < len(original_indices) else 0
    
    return {
        'current_idx': current_idx,
        'prev_idx': prev_idx,
        'next_idx': next_idx,
        'total_count': total_count,
        'original_idx': original_idx,
        'is_filtered': filter_ungraded,
        'can_navigate': total_count > 0,
        'position_text': f"{current_idx + 1} of {total_count}" if total_count > 0 else "No molecules"
    }


def _get_next_milestone(graded_count: int) -> Dict[str, Any]:
    """
    Get next grading milestone.
    Private helper function.
    
    Args:
        graded_count: Current number of graded molecules
    
    Returns:
        Dictionary with milestone information
    """
    milestones = [
        (3, "ML Training Available"),
        (5, "Basic Model Performance"),
        (10, "Good Model Performance"),
        (25, "Strong Model Performance"),
        (50, "Excellent Model Performance"),
        (100, "Outstanding Dataset")
    ]
    
    for count, description in milestones:
        if graded_count < count:
            return {
                'target_count': count,
                'description': description,
                'remaining': count - graded_count
            }
    
    # If all milestones passed
    return {
        'target_count': graded_count,
        'description': "All milestones achieved!",
        'remaining': 0
    }