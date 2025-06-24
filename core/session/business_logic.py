"""
Business logic functions for session operations using functional programming approach.
All functions are pure - no side effects, return results.
"""

from typing import Dict, List, Any, Optional
import logging

from .service import (
    generate_session_id, create_new_session, save_session_data,
    prepare_file_for_processing, find_default_score_property
)
from .file_handler import process_score_column
from .processing import create_processing_configs, execute_processing_pipeline

logger = logging.getLogger(__name__)


def create_and_save_session(
    protein_file: Any,
    ligand_path: str,
    score_label: str,
    score_direction: str,
    available_properties: List[str],
    molecular_fp_types: List[str],
    interaction_type: str,
    compute_pose_quality: bool
) -> Dict[str, Any]:
    """
    Create and save a new session with all processing steps.
    Pure function orchestrating session creation business logic.
    
    Args:
        protein_file: Uploaded protein file
        ligand_path: Path to ligand SDF file
        score_label: Selected score column name
        score_direction: Score interpretation direction
        available_properties: Available SDF properties
        molecular_fp_types: Selected molecular fingerprint types
        interaction_type: Selected interaction type
        compute_pose_quality: Whether to compute pose quality
    
    Returns:
        Dictionary with operation result and data
    """
    try:
        # Generate session ID
        new_session_id = generate_session_id()
        
        # Load and validate molecules
        molecules_df = prepare_file_for_processing(ligand_path)
        if molecules_df is None:
            return {
                'success': False,
                'error': 'Failed to load SDF file',
                'error_type': 'file_loading'
            }
        
        # Process score column
        protein_content = protein_file.getvalue().decode('utf-8')
        score_result = process_score_column(molecules_df, score_label, score_direction)
        
        if not score_result['success']:
            return {
                'success': False,
                'error': score_result['error_message'],
                'error_type': 'score_processing',
                'available_columns': score_result.get('available_columns', [])
            }
        
        processed_molecules_df = score_result['molecules_df']
        score_range = score_result['score_range']
        
        # Create processing configurations
        processing_configs = create_processing_configs(
            molecular_fp_types, interaction_type, compute_pose_quality
        )
        
        # Execute processing pipeline
        processing_result = execute_processing_pipeline(
            processed_molecules_df, protein_content, processing_configs
        )
        
        if not processing_result['processing_summary']['success']:
            return {
                'success': False,
                'error': 'Processing pipeline failed',
                'error_type': 'processing',
                'processing_summary': processing_result['processing_summary']
            }
        
        processed_df = processing_result['processed_df']
        processing_summary = processing_result['processing_summary']
        
        # Create session data
        session_data = create_new_session(
            new_session_id, protein_file.name, protein_content, processed_df,
            score_label, score_direction, interaction_type, compute_pose_quality,
            available_properties
        )
        
        # Save session
        save_success = save_session_data(session_data)
        if not save_success:
            return {
                'success': False,
                'error': 'Failed to save session data',
                'error_type': 'saving'
            }
        
        return {
            'success': True,
            'session_id': new_session_id,
            'molecules_df': processed_df,
            'protein_content': protein_content,
            'processing_summary': processing_summary,
            'score_range': score_range,
            'molecules_count': len(processed_df)
        }
        
    except Exception as e:
        logger.error(f"Session creation error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f"Unexpected error: {str(e)}",
            'error_type': 'unexpected',
            'exception_details': str(e)
        }


def execute_reprocessing_pipeline(
    session_data: Dict[str, Any],
    molecular_fp_types: List[str],
    interaction_type: str,
    compute_pose_quality: bool
) -> Dict[str, Any]:
    """
    Execute reprocessing pipeline for existing session.
    Pure function orchestrating reprocessing business logic.
    
    Args:
        session_data: Loaded session data
        molecular_fp_types: Selected molecular fingerprint types
        interaction_type: Selected interaction type
        compute_pose_quality: Whether to compute pose quality
    
    Returns:
        Dictionary with operation result and data
    """
    try:
        from .service import reprocess_session_data
        
        # Prepare for reprocessing
        reprocessing_data = reprocess_session_data(
            session_data, molecular_fp_types, interaction_type, compute_pose_quality
        )
        
        # Create processing configurations
        processing_configs = create_processing_configs(
            molecular_fp_types, interaction_type, compute_pose_quality
        )
        
        # Execute processing pipeline
        processing_result = execute_processing_pipeline(
            reprocessing_data['molecules_df'],
            reprocessing_data['metadata']['protein_content'],
            processing_configs
        )
        
        if not processing_result['processing_summary']['success']:
            return {
                'success': False,
                'error': 'Reprocessing pipeline failed',
                'error_type': 'processing',
                'processing_summary': processing_result['processing_summary']
            }
        
        processed_df = processing_result['processed_df']
        processing_summary = processing_result['processing_summary']
        
        # Create updated session data
        updated_session_data = {
            'session_id': session_data['session_id'],
            'session_dir': reprocessing_data['session_dir'],
            'molecules_df': processed_df,
            'metadata': reprocessing_data['metadata']
        }
        
        # Save updated session
        save_success = save_session_data(updated_session_data)
        if not save_success:
            return {
                'success': False,
                'error': 'Failed to save reprocessed data',
                'error_type': 'saving'
            }
        
        # Count preserved grades
        preserved_grades = 0
        if 'grade' in processed_df.columns:
            preserved_grades = processed_df['grade'].notna().sum()
        
        return {
            'success': True,
            'session_id': session_data['session_id'],
            'molecules_df': processed_df,
            'protein_content': reprocessing_data['metadata']['protein_content'],
            'processing_summary': processing_summary,
            'molecules_count': len(processed_df),
            'preserved_grades': preserved_grades,
            'debug_info': reprocessing_data['debug_info']
        }
        
    except Exception as e:
        logger.error(f"Reprocessing error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f"Unexpected reprocessing error: {str(e)}",
            'error_type': 'unexpected',
            'exception_details': str(e)
        }


def validate_and_prepare_session_creation(
    protein_file: Any,
    ligand_file: Any,
    score_label: Optional[str],
    molecular_fp_types: List[str]
) -> Dict[str, Any]:
    """
    Validate inputs and prepare data for session creation.
    Pure function - returns validation and preparation results.
    
    Args:
        protein_file: Uploaded protein file
        ligand_file: Uploaded ligand file
        score_label: Selected score column name
        molecular_fp_types: Selected molecular fingerprint types
    
    Returns:
        Dictionary with validation results and prepared data
    """
    # Import here to avoid circular imports
    from core.ui.validation import validate_session_inputs, validate_file_uploads
    
    # Validate inputs
    input_validation = validate_session_inputs(
        protein_file, ligand_file, score_label, molecular_fp_types
    )
    
    if not input_validation['is_valid']:
        return {
            'success': False,
            'validation_errors': input_validation['errors'],
            'validation_warnings': input_validation['warnings']
        }
    
    # Validate files
    file_validation = validate_file_uploads(protein_file, ligand_file)
    
    # Create temporary file for ligand processing
    ligand_path = None
    if ligand_file:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.sdf', delete=False) as f:
            f.write(ligand_file.getvalue())
            ligand_path = f.name
    
    # Detect SDF properties if file is valid
    available_properties = []
    if ligand_path and file_validation['is_valid']:
        from .file_handler import detect_sdf_properties
        available_properties = detect_sdf_properties(ligand_path)
        
        # Auto-detect score property if not provided
        if not score_label:
            score_label, _ = find_default_score_property(available_properties)
    
    return {
        'success': file_validation['is_valid'],
        'ligand_path': ligand_path,
        'available_properties': available_properties,
        'detected_score_label': score_label,
        'validation_warnings': input_validation['warnings'] + file_validation['warnings'],
        'validation_errors': file_validation['errors']
    }