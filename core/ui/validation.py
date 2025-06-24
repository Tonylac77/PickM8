"""
UI validation functions using functional programming approach.
All functions are pure - no side effects, return validation results.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def validate_session_inputs(
    protein_file: Any,
    ligand_file: Any, 
    score_label: Optional[str],
    molecular_fp_types: List[str]
) -> Dict[str, Any]:
    """
    Validate all inputs required for session creation.
    Pure function - returns validation result without side effects.
    
    Args:
        protein_file: Uploaded protein file
        ligand_file: Uploaded ligand file
        score_label: Selected score column name
        molecular_fp_types: Selected molecular fingerprint types
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate protein file
    if not protein_file:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Protein file is required")
    
    # Validate ligand file
    if not ligand_file:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Ligand file is required")
    
    # Validate score selection
    if not score_label:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Score column selection is required")
    
    # Validate fingerprint selection
    if not molecular_fp_types:
        validation_result['is_valid'] = False
        validation_result['errors'].append("At least one molecular fingerprint type is required")
    
    return validation_result


def validate_file_uploads(protein_file: Any, ligand_file: Any) -> Dict[str, Any]:
    """
    Validate uploaded files for basic requirements.
    Pure function - returns validation result without side effects.
    
    Args:
        protein_file: Uploaded protein file
        ligand_file: Uploaded ligand file
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate protein file
    if protein_file:
        if not protein_file.name.lower().endswith('.pdb'):
            validation_result['warnings'].append("Protein file should have .pdb extension")
        
        # Check file size (basic check)
        try:
            content = protein_file.getvalue()
            if len(content) == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Protein file is empty")
            elif len(content) > 50 * 1024 * 1024:  # 50MB limit
                validation_result['warnings'].append("Protein file is very large (>50MB)")
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Error reading protein file: {str(e)}")
    
    # Validate ligand file
    if ligand_file:
        if not ligand_file.name.lower().endswith('.sdf'):
            validation_result['warnings'].append("Ligand file should have .sdf extension")
        
        # Check file size (basic check)
        try:
            content = ligand_file.getvalue()
            if len(content) == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Ligand file is empty")
            elif len(content) > 100 * 1024 * 1024:  # 100MB limit
                validation_result['warnings'].append("Ligand file is very large (>100MB)")
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Error reading ligand file: {str(e)}")
    
    return validation_result


def validate_processing_configuration(
    interaction_type: str,
    molecular_fp_types: List[str],
    compute_pose_quality: bool
) -> Dict[str, Any]:
    """
    Validate processing configuration options.
    Pure function - returns validation result without side effects.
    
    Args:
        interaction_type: Selected interaction analysis type
        molecular_fp_types: Selected molecular fingerprint types
        compute_pose_quality: Whether to compute pose quality
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate interaction type
    valid_interaction_types = ['plip', 'prolif']
    if interaction_type not in valid_interaction_types:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Invalid interaction type: {interaction_type}")
    
    # Validate molecular fingerprint types
    valid_fp_types = ['morgan', 'rdkit', 'mapchiral']
    invalid_fp_types = [fp for fp in molecular_fp_types if fp not in valid_fp_types]
    if invalid_fp_types:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Invalid fingerprint types: {invalid_fp_types}")
    
    if not molecular_fp_types:
        validation_result['is_valid'] = False
        validation_result['errors'].append("At least one molecular fingerprint type is required")
    
    # Warnings for performance
    if len(molecular_fp_types) > 2 and compute_pose_quality:
        validation_result['warnings'].append("Computing multiple fingerprint types with pose quality may be slow")
    
    if interaction_type == 'prolif':
        validation_result['warnings'].append("ProLIF analysis may take longer than PLIP")
    
    return validation_result


def validate_reprocessing_inputs(
    session_data: Dict[str, Any],
    molecular_fp_types: List[str],
    interaction_type: str,
    compute_pose_quality: bool
) -> Dict[str, Any]:
    """
    Validate inputs for session reprocessing.
    Pure function - returns validation result without side effects.
    
    Args:
        session_data: Loaded session data
        molecular_fp_types: Selected molecular fingerprint types
        interaction_type: Selected interaction type
        compute_pose_quality: Whether to compute pose quality
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate session data
    if not session_data:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Session data is required")
        return validation_result
    
    if 'molecules_df' not in session_data:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Session must contain molecules data")
        return validation_result
    
    if 'metadata' not in session_data:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Session must contain metadata")
        return validation_result
    
    # Validate processing configuration
    config_validation = validate_processing_configuration(
        interaction_type, molecular_fp_types, compute_pose_quality
    )
    
    if not config_validation['is_valid']:
        validation_result['is_valid'] = False
        validation_result['errors'].extend(config_validation['errors'])
    
    validation_result['warnings'].extend(config_validation['warnings'])
    
    # Check if reprocessing will preserve grades
    molecules_df = session_data['molecules_df']
    if 'grade' in molecules_df.columns:
        graded_count = molecules_df['grade'].notna().sum()
        if graded_count > 0:
            validation_result['warnings'].append(
                f"Reprocessing will preserve {graded_count} existing grades"
            )
    
    return validation_result