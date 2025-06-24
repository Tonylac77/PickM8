"""
UI state management functions using functional programming approach.
All functions are pure - return new state data without side effects.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def prepare_session_state_data(
    session_id: str,
    molecules_df: Any,
    protein_content: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare session state data for Streamlit session state.
    Pure function - returns state data without side effects.
    
    Args:
        session_id: Session identifier
        molecules_df: Molecules DataFrame
        protein_content: PDB protein content
        metadata: Session metadata
    
    Returns:
        Dictionary containing session state data
    """
    return {
        'session_id': session_id,
        'molecules_df': molecules_df,
        'protein_content': protein_content,
        'metadata': metadata,
        'last_updated': metadata.get('created_date', ''),
        'num_molecules': len(molecules_df) if molecules_df is not None else 0,
        'num_graded': molecules_df['grade'].notna().sum() if molecules_df is not None and 'grade' in molecules_df.columns else 0
    }


def update_session_state(
    current_state: Dict[str, Any],
    updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update session state with new data.
    Pure function - returns new state without modifying input.
    
    Args:
        current_state: Current session state
        updates: Updates to apply
    
    Returns:
        New session state with updates applied
    """
    # Create copy to avoid modifying original
    new_state = current_state.copy()
    
    # Apply updates
    for key, value in updates.items():
        new_state[key] = value
    
    # Recalculate derived values if molecules_df was updated
    if 'molecules_df' in updates and updates['molecules_df'] is not None:
        molecules_df = updates['molecules_df']
        new_state['num_molecules'] = len(molecules_df)
        new_state['num_graded'] = molecules_df['grade'].notna().sum() if 'grade' in molecules_df.columns else 0
    
    return new_state


def clear_session_selections() -> Dict[str, Any]:
    """
    Get dictionary of session state keys to clear.
    Pure function - returns clear operations without side effects.
    
    Returns:
        Dictionary of keys to clear from session state
    """
    return {
        'selected_session_for_load': None,
        'selected_session_for_reprocess': None,
        'reprocess_in_progress': False,
        'processing_complete': False
    }


def prepare_navigation_state(
    session_id: str,
    has_molecules: bool,
    has_grades: bool,
    processing_complete: bool
) -> Dict[str, Any]:
    """
    Prepare navigation state data for UI decisions.
    Pure function - returns navigation state without side effects.
    
    Args:
        session_id: Current session ID
        has_molecules: Whether session has molecule data
        has_grades: Whether session has any grades
        processing_complete: Whether processing is complete
    
    Returns:
        Dictionary containing navigation state information
    """
    return {
        'can_navigate_to_active_learning': has_molecules and processing_complete,
        'can_navigate_to_results': has_molecules and processing_complete,
        'should_show_grading_progress': has_molecules and has_grades,
        'session_ready': processing_complete and has_molecules,
        'recommendations': _get_navigation_recommendations(
            has_molecules, has_grades, processing_complete
        )
    }


def _get_navigation_recommendations(
    has_molecules: bool,
    has_grades: bool, 
    processing_complete: bool
) -> Dict[str, str]:
    """
    Get navigation recommendations based on session state.
    Private helper function.
    
    Args:
        has_molecules: Whether session has molecule data
        has_grades: Whether session has any grades
        processing_complete: Whether processing is complete
    
    Returns:
        Dictionary of navigation recommendations
    """
    recommendations = {}
    
    if not processing_complete:
        recommendations['primary'] = "Complete molecule processing first"
        recommendations['secondary'] = "Wait for processing to finish"
    elif not has_molecules:
        recommendations['primary'] = "No molecules available"
        recommendations['secondary'] = "Check session data"
    elif not has_grades:
        recommendations['primary'] = "Start with Active Learning to grade molecules"
        recommendations['secondary'] = "Grade at least 5 molecules to train ML model"
    else:
        recommendations['primary'] = "Continue with Active Learning or view Results"
        recommendations['secondary'] = "ML training available with existing grades"
    
    return recommendations


def prepare_processing_state(
    total_molecules: int,
    processed_count: int,
    current_step: str,
    errors: list = None
) -> Dict[str, Any]:
    """
    Prepare processing state data for progress tracking.
    Pure function - returns processing state without side effects.
    
    Args:
        total_molecules: Total number of molecules to process
        processed_count: Number of molecules processed
        current_step: Current processing step
        errors: List of processing errors
    
    Returns:
        Dictionary containing processing state information
    """
    if errors is None:
        errors = []
    
    progress = processed_count / total_molecules if total_molecules > 0 else 0.0
    
    return {
        'total_molecules': total_molecules,
        'processed_count': processed_count,
        'progress_percentage': progress * 100,
        'progress_fraction': progress,
        'current_step': current_step,
        'has_errors': len(errors) > 0,
        'error_count': len(errors),
        'errors': errors.copy(),
        'is_complete': processed_count >= total_molecules and len(errors) == 0,
        'status_message': _get_processing_status_message(
            processed_count, total_molecules, current_step, len(errors)
        )
    }


def _get_processing_status_message(
    processed_count: int,
    total_molecules: int,
    current_step: str,
    error_count: int
) -> str:
    """
    Get processing status message based on current state.
    Private helper function.
    
    Args:
        processed_count: Number of molecules processed
        total_molecules: Total number of molecules
        current_step: Current processing step
        error_count: Number of errors
    
    Returns:
        Status message string
    """
    if error_count > 0:
        return f"Processing with {error_count} errors - {current_step}"
    elif processed_count >= total_molecules:
        return "Processing complete"
    else:
        return f"Processing {processed_count}/{total_molecules} - {current_step}"