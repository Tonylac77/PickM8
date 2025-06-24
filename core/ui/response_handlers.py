"""
UI response handler functions using functional programming approach.
All functions are pure - return response data without side effects.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def handle_creation_success(
    session_id: str,
    molecules_count: int,
    processing_summary: Dict[str, Any],
    protein_name: str
) -> Dict[str, Any]:
    """
    Prepare success response data for session creation.
    Pure function - returns response data without side effects.
    
    Args:
        session_id: Created session ID
        molecules_count: Number of molecules processed
        processing_summary: Summary of processing results
        protein_name: Name of protein file
    
    Returns:
        Dictionary containing success response data
    """
    return {
        'type': 'success',
        'title': 'Session Created Successfully',
        'message': f"Successfully created session and processed {molecules_count} molecules!",
        'details': {
            'session_id': session_id,
            'protein_name': protein_name,
            'molecules_count': molecules_count,
            'processing_summary': processing_summary
        },
        'show_balloons': True,
        'navigation_enabled': True,
        'next_steps': [
            "Navigate to Active Learning to start grading molecules",
            "View Results to analyze the processed data",
            "Grade at least 5 molecules to enable ML training"
        ]
    }


def handle_creation_error(
    error_message: str,
    error_details: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepare error response data for session creation failures.
    Pure function - returns response data without side effects.
    
    Args:
        error_message: Main error message
        error_details: Detailed error information
        suggestions: List of suggestions to resolve the error
    
    Returns:
        Dictionary containing error response data
    """
    if suggestions is None:
        suggestions = [
            "Check that files are valid and not corrupted",
            "Ensure protein file is in PDB format",
            "Ensure ligand file is in SDF format with valid molecules",
            "Try with smaller files if processing times out"
        ]
    
    return {
        'type': 'error',
        'title': 'Session Creation Failed',
        'message': error_message,
        'details': error_details,
        'suggestions': suggestions,
        'show_expander': error_details is not None,
        'navigation_enabled': False
    }


def handle_loading_success(
    session_id: str,
    protein_name: str,
    molecules_count: int,
    graded_count: int,
    last_modified: str
) -> Dict[str, Any]:
    """
    Prepare success response data for session loading.
    Pure function - returns response data without side effects.
    
    Args:
        session_id: Loaded session ID
        protein_name: Name of protein
        molecules_count: Number of molecules in session
        graded_count: Number of graded molecules
        last_modified: Last modification timestamp
    
    Returns:
        Dictionary containing success response data
    """
    progress_percentage = (graded_count / molecules_count * 100) if molecules_count > 0 else 0
    
    return {
        'type': 'success',
        'title': 'Session Loaded Successfully',
        'message': f"Loaded session: {protein_name}",
        'details': {
            'session_id': session_id,
            'protein_name': protein_name,
            'molecules_count': molecules_count,
            'graded_count': graded_count,
            'progress_percentage': progress_percentage,
            'last_modified': last_modified
        },
        'navigation_enabled': True,
        'next_steps': _get_loading_next_steps(graded_count, molecules_count)
    }


def handle_loading_error(
    session_id: str,
    error_message: str,
    protein_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Prepare error response data for session loading failures.
    Pure function - returns response data without side effects.
    
    Args:
        session_id: Session ID that failed to load
        error_message: Error message
        protein_name: Protein name if available
    
    Returns:
        Dictionary containing error response data
    """
    display_name = protein_name if protein_name else session_id[:8]
    
    return {
        'type': 'error',
        'title': 'Session Loading Failed',
        'message': f"Failed to load session: {display_name}",
        'details': error_message,
        'suggestions': [
            "Check if session files are intact",
            "Try restarting the application",
            "Contact support if the issue persists"
        ],
        'navigation_enabled': False
    }


def handle_processing_success(
    molecules_count: int,
    processing_summary: Dict[str, Any],
    fingerprint_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare success response data for molecule processing.
    Pure function - returns response data without side effects.
    
    Args:
        molecules_count: Number of molecules processed
        processing_summary: Summary of processing results
        fingerprint_stats: Fingerprint computation statistics
    
    Returns:
        Dictionary containing success response data
    """
    return {
        'type': 'success',
        'title': 'Processing Complete',
        'message': f"Successfully processed {molecules_count} molecules",
        'details': {
            'molecules_count': molecules_count,
            'processing_summary': processing_summary,
            'fingerprint_stats': fingerprint_stats
        },
        'show_statistics': True,
        'processing_complete': True
    }


def handle_processing_error(
    error_message: str,
    processed_count: int,
    total_count: int,
    failed_molecules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Prepare error response data for processing failures.
    Pure function - returns response data without side effects.
    
    Args:
        error_message: Main error message
        processed_count: Number of molecules successfully processed
        total_count: Total number of molecules
        failed_molecules: List of failed molecule IDs
    
    Returns:
        Dictionary containing error response data
    """
    if failed_molecules is None:
        failed_molecules = []
    
    return {
        'type': 'error',
        'title': 'Processing Failed',
        'message': error_message,
        'details': {
            'processed_count': processed_count,
            'total_count': total_count,
            'failed_count': len(failed_molecules),
            'failed_molecules': failed_molecules
        },
        'suggestions': [
            "Check molecule structures in SDF file",
            "Ensure protein file is valid PDB format",
            "Try processing with different fingerprint options",
            "Check system resources if processing large datasets"
        ],
        'partial_success': processed_count > 0,
        'processing_complete': False
    }


def handle_reprocessing_success(
    session_id: str,
    molecules_count: int,
    preserved_grades: int
) -> Dict[str, Any]:
    """
    Prepare success response data for session reprocessing.
    Pure function - returns response data without side effects.
    
    Args:
        session_id: Reprocessed session ID
        molecules_count: Number of molecules reprocessed
        preserved_grades: Number of grades preserved
    
    Returns:
        Dictionary containing success response data
    """
    return {
        'type': 'success',
        'title': 'Reprocessing Complete',
        'message': f"Successfully reprocessed {molecules_count} molecules",
        'details': {
            'session_id': session_id,
            'molecules_count': molecules_count,
            'preserved_grades': preserved_grades
        },
        'navigation_enabled': True,
        'next_steps': [
            "Continue with Active Learning using existing grades",
            "Review updated fingerprints and interactions",
            "Train new ML models with updated features"
        ] if preserved_grades > 0 else [
            "Navigate to Active Learning to start grading",
            "Grade molecules with new fingerprint data",
            "Train ML models once sufficient grades are available"
        ]
    }


def _get_loading_next_steps(graded_count: int, molecules_count: int) -> List[str]:
    """
    Get next steps based on grading progress.
    Private helper function.
    
    Args:
        graded_count: Number of graded molecules
        molecules_count: Total number of molecules
    
    Returns:
        List of next step recommendations
    """
    if graded_count == 0:
        return [
            "Navigate to Active Learning to start grading molecules",
            "Grade at least 5 molecules to enable ML training",
            "Use the grading interface to assess molecule quality"
        ]
    elif graded_count < 5:
        return [
            f"Continue grading molecules ({graded_count}/5 minimum for ML)",
            "Use Active Learning interface to grade more molecules",
            "ML training will be available after 5 grades"
        ]
    elif graded_count < molecules_count:
        return [
            "Continue with Active Learning to grade more molecules",
            "Train ML models to get predictions for ungraded molecules",
            "View Results to analyze current progress"
        ]
    else:
        return [
            "All molecules are graded - great work!",
            "View Results to analyze the complete dataset",
            "Export results for further analysis"
        ]