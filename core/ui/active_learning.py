"""
UI components for Active Learning page using functional programming approach.
All functions are pure - return display data without side effects.
"""

from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def prepare_training_diagnostics(graded_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare diagnostic information for model training failures.
    Pure function - returns diagnostic data without side effects.
    
    Args:
        graded_df: DataFrame of graded molecules
    
    Returns:
        Dictionary containing diagnostic information
    """
    fingerprint_status = []
    
    for idx, row in graded_df.iterrows():
        # Safe checking for fingerprint data (handles arrays and None)
        def check_fp_status(fp_data):
            if fp_data is None:
                return '❌'
            if isinstance(fp_data, (list, np.ndarray)):
                return '✅' if len(fp_data) > 0 else '❌'
            return '✅' if pd.notna(fp_data) else '❌'
        
        status = {
            'ID': row['id'],
            'Name': row['name'],
            'Grade': row['grade'],
            'Morgan FP': check_fp_status(row.get('morgan_fp')),
            'RDKit FP': check_fp_status(row.get('rdkit_fp')),
            'MapChiral FP': check_fp_status(row.get('mapchiral_fp')),
            'Interaction FP': check_fp_status(row.get('interaction_fp'))
        }
        fingerprint_status.append(status)
    
    # Calculate summary statistics
    total_molecules = len(graded_df)
    fp_columns = ['morgan_fp', 'rdkit_fp', 'mapchiral_fp', 'interaction_fp']
    missing_counts = {}
    
    for col in fp_columns:
        if col in graded_df.columns:
            missing_count = graded_df[col].isna().sum()
            missing_counts[col] = {
                'missing': missing_count,
                'available': total_molecules - missing_count,
                'percentage_available': ((total_molecules - missing_count) / total_molecules * 100) if total_molecules > 0 else 0
            }
    
    return {
        'fingerprint_status': fingerprint_status,
        'summary_stats': missing_counts,
        'total_molecules': total_molecules,
        'recommendations': _get_training_recommendations(missing_counts, total_molecules)
    }


def prepare_training_success_response(
    training_stats: Dict[str, Any],
    prediction_stats: Dict[str, Any],
    molecules_count: int
) -> Dict[str, Any]:
    """
    Prepare success response for model training.
    Pure function - returns response data without side effects.
    
    Args:
        training_stats: Training statistics
        prediction_stats: Prediction statistics  
        molecules_count: Number of molecules processed
    
    Returns:
        Dictionary containing success response data
    """
    return {
        'type': 'success',
        'title': 'Model Training Complete',
        'message': f"Successfully trained model and updated predictions for {molecules_count} molecules",
        'training_stats': training_stats,
        'prediction_stats': prediction_stats,
        'next_steps': [
            "Review molecules with high prediction uncertainty",
            "Grade suggested molecules to improve model performance",
            "Use predictions to prioritize molecule review",
            "Continue iterative active learning process"
        ],
        'show_statistics': True
    }


def prepare_training_error_response(
    error_message: str,
    graded_count: int,
    diagnostics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare error response for model training failures.
    Pure function - returns response data without side effects.
    
    Args:
        error_message: Main error message
        graded_count: Number of graded molecules
        diagnostics: Diagnostic information
    
    Returns:
        Dictionary containing error response data
    """
    return {
        'type': 'error',
        'title': 'Model Training Failed',
        'message': error_message,
        'graded_count': graded_count,
        'diagnostics': diagnostics,
        'suggestions': [
            "Ensure molecules have computed fingerprints",
            "Check that data processing completed successfully",
            "Verify graded molecules have valid molecular data",
            "Try reprocessing the session if fingerprints are missing"
        ],
        'show_diagnostics': diagnostics is not None
    }


def prepare_molecule_suggestion_data(
    df: pd.DataFrame,
    strategy: str,
    has_predictions: bool
) -> Dict[str, Any]:
    """
    Prepare data for molecule suggestion interface.
    Pure function - returns suggestion data without side effects.
    
    Args:
        df: Molecules DataFrame
        strategy: Selection strategy ('random', 'uncertainty', 'diverse')
        has_predictions: Whether ML predictions are available
    
    Returns:
        Dictionary containing suggestion data
    """
    ungraded_df = df[df['grade'].isna()]
    available_strategies = ['random']
    
    if has_predictions and len(ungraded_df) > 0:
        available_strategies.extend(['uncertainty', 'diverse'])
    
    current_strategy = strategy if strategy in available_strategies else 'random'
    
    # Get suggested molecules based on strategy
    suggested_molecules = []
    if len(ungraded_df) > 0:
        if current_strategy == 'random':
            suggested_molecules = ungraded_df.sample(min(5, len(ungraded_df)))['id'].tolist()
        elif current_strategy == 'uncertainty' and has_predictions:
            # Sort by highest uncertainty
            uncertainty_sorted = ungraded_df.sort_values('prediction_uncertainty', ascending=False, na_last=True)
            suggested_molecules = uncertainty_sorted.head(5)['id'].tolist()
        elif current_strategy == 'diverse' and has_predictions:
            # Simple diversity: select molecules with different predicted grades
            try:
                diverse_selection = ungraded_df.groupby('prediction').head(1)
                suggested_molecules = diverse_selection.head(5)['id'].tolist()
            except:
                # Fallback to random if diversity selection fails
                suggested_molecules = ungraded_df.sample(min(5, len(ungraded_df)))['id'].tolist()
    
    return {
        'available_strategies': available_strategies,
        'current_strategy': current_strategy,
        'suggested_molecules': suggested_molecules,
        'ungraded_count': len(ungraded_df),
        'can_use_ml_strategies': has_predictions and len(ungraded_df) > 0
    }


def prepare_grading_interface_data(
    df: pd.DataFrame,
    mol_idx: int,
    protein_content: str
) -> Dict[str, Any]:
    """
    Prepare data for molecule grading interface.
    Pure function - returns interface data without side effects.
    
    Args:
        df: Molecules DataFrame
        mol_idx: Current molecule index
        protein_content: Protein PDB content
    
    Returns:
        Dictionary containing grading interface data
    """
    if mol_idx >= len(df):
        mol_idx = 0
    
    mol_data = df.iloc[mol_idx]
    
    # Prepare navigation data
    total_molecules = len(df)
    prev_idx = (mol_idx - 1) % total_molecules
    next_idx = (mol_idx + 1) % total_molecules
    
    # Count grading progress
    graded_count = df['grade'].notna().sum()
    ungraded_count = df['grade'].isna().sum()
    
    # Get grade status
    current_grade = mol_data.get('grade')
    has_grade = pd.notna(current_grade)
    
    return {
        'mol_data': mol_data.to_dict(),
        'mol_idx': mol_idx,
        'molecule_id': mol_data['id'],
        'molecule_name': mol_data['name'],
        'mol_block': mol_data['mol_block'],
        'protein_content': protein_content,
        'current_grade': current_grade,
        'has_grade': has_grade,
        'navigation': {
            'current_idx': mol_idx,
            'total_molecules': total_molecules,
            'prev_idx': prev_idx,
            'next_idx': next_idx
        },
        'progress': {
            'graded_count': graded_count,
            'ungraded_count': ungraded_count,
            'total_count': total_molecules,
            'percentage': (graded_count / total_molecules * 100) if total_molecules > 0 else 0
        }
    }


def _get_training_recommendations(
    missing_counts: Dict[str, Any],
    total_molecules: int
) -> List[str]:
    """
    Get recommendations based on fingerprint availability.
    Private helper function.
    
    Args:
        missing_counts: Dictionary of missing fingerprint counts
        total_molecules: Total number of molecules
    
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    if total_molecules == 0:
        recommendations.append("No graded molecules available")
        return recommendations
    
    # Check each fingerprint type
    for fp_type, stats in missing_counts.items():
        if stats['percentage_available'] < 50:
            recommendations.append(f"Most {fp_type} fingerprints are missing - reprocess session")
        elif stats['percentage_available'] < 80:
            recommendations.append(f"Some {fp_type} fingerprints are missing - check processing logs")
    
    if not recommendations:
        recommendations.append("Fingerprint data looks good - check for other processing issues")
    
    recommendations.append("Consider reprocessing the session if issues persist")
    
    return recommendations