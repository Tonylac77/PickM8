"""
Processing pipeline functions using functional programming approach.
All functions are pure with no side effects.
"""

from typing import Dict, List, Any, Tuple
import logging

from core.fingerprints import (
    create_default_fingerprint_config,
    create_default_interaction_config,
)
from core.pose_analysis import (
    compute_pose_quality_batch,
    create_default_posecheck_config,
)
from utils.processing import compute_fingerprints_batch, get_fingerprint_statistics

logger = logging.getLogger(__name__)


def create_processing_configs(
    molecular_fp_types: List[str],
    interaction_type: str,
    compute_pose_quality: bool
) -> Dict[str, Any]:
    """
    Create processing configurations based on user selections.
    Pure function - returns configuration dictionaries.
    
    Args:
        molecular_fp_types: List of molecular fingerprint types to compute
        interaction_type: Type of interaction analysis (plip/prolif)
        compute_pose_quality: Whether to compute pose quality metrics
    
    Returns:
        Dictionary containing all processing configurations
    """
    # Create fingerprint configuration
    fp_config = create_default_fingerprint_config()
    fp_config['compute_morgan'] = 'morgan' in molecular_fp_types
    fp_config['compute_rdkit'] = 'rdkit' in molecular_fp_types
    fp_config['compute_mapchiral'] = 'mapchiral' in molecular_fp_types
    
    # Create interaction configuration
    interaction_config = create_default_interaction_config()
    interaction_config['interaction_type'] = interaction_type
    
    # Create pose configuration
    pose_config = create_default_posecheck_config()
    pose_config['calculate_clashes'] = compute_pose_quality
    pose_config['calculate_strain'] = compute_pose_quality
    
    return {
        'fingerprint_config': fp_config,
        'interaction_config': interaction_config,
        'pose_config': pose_config,
        'molecular_fp_types': molecular_fp_types,
        'interaction_type': interaction_type,
        'compute_pose_quality': compute_pose_quality
    }


def execute_processing_pipeline(
    molecules_df,
    protein_content: str,
    processing_configs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute the complete processing pipeline on molecules.
    Pure function - returns new DataFrame and processing statistics.
    
    Args:
        molecules_df: Input molecules DataFrame
        protein_content: PDB protein content
        processing_configs: Configuration dictionary from create_processing_configs
    
    Returns:
        Dictionary with processed DataFrame and statistics
    """
    # Extract configurations
    fp_config = processing_configs['fingerprint_config']
    interaction_config = processing_configs['interaction_config']
    pose_config = processing_configs['pose_config']
    
    # Start with copy of input DataFrame
    processed_df = molecules_df.copy()
    
    processing_steps = []
    
    try:
        # Step 1: Compute fingerprints
        processing_steps.append("Computing molecular and interaction fingerprints...")
        processed_df = compute_fingerprints_batch(
            processed_df, protein_content, fp_config, interaction_config
        )
        
        # Step 2: Compute pose quality metrics (if enabled)
        if processing_configs['compute_pose_quality']:
            processing_steps.append("Analyzing pose quality...")
            processed_df = compute_pose_quality_batch(processed_df, protein_content, pose_config)
        else:
            processing_steps.append("Skipping pose quality analysis...")
        
        # Step 3: Generate statistics
        processing_steps.append("Generating processing statistics...")
        fp_stats = get_fingerprint_statistics(processed_df)
        
        # Create processing summary
        processing_summary = {
            'success': True,
            'molecules_processed': len(processed_df),
            'processing_steps': processing_steps,
            'fingerprint_stats': fp_stats,
            'configurations_used': processing_configs
        }
        
        return {
            'processed_df': processed_df,
            'processing_summary': processing_summary
        }
        
    except Exception as e:
        logger.error(f"Error in processing pipeline: {e}")
        
        processing_summary = {
            'success': False,
            'error': str(e),
            'processing_steps': processing_steps,
            'configurations_used': processing_configs
        }
        
        return {
            'processed_df': molecules_df,  # Return original on error
            'processing_summary': processing_summary
        }


def create_processing_statistics_summary(fp_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a formatted summary of processing statistics.
    Pure function - formats statistics for display.
    
    Args:
        fp_stats: Fingerprint statistics from get_fingerprint_statistics
    
    Returns:
        Dictionary with formatted statistics for UI display
    """
    return {
        'total_molecules': fp_stats.get('total_molecules', 0),
        'fingerprint_percentages': {
            'morgan': fp_stats.get('morgan_fp_percentage', 0),
            'rdkit': fp_stats.get('rdkit_fp_percentage', 0),
            'mapchiral': fp_stats.get('mapchiral_fp_percentage', 0),
            'interaction': fp_stats.get('interaction_fp_percentage', 0)
        },
        'interaction_metrics': {
            'avg_interactions': fp_stats.get('avg_interactions_per_molecule', 0),
            'molecules_with_interactions': fp_stats.get('molecules_with_interactions', 0),
            'max_interactions': fp_stats.get('max_interactions', 0)
        }
    }


def validate_processing_inputs(
    molecules_df,
    protein_content: str,
    molecular_fp_types: List[str]
) -> Dict[str, Any]:
    """
    Validate inputs for processing pipeline. Pure function.
    
    Args:
        molecules_df: Molecules DataFrame to validate
        protein_content: Protein content to validate
        molecular_fp_types: Selected fingerprint types to validate
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Validate molecules DataFrame
    if molecules_df is None or len(molecules_df) == 0:
        validation_result['valid'] = False
        validation_result['errors'].append("No molecules provided for processing")
    
    # Validate protein content
    if not protein_content or not protein_content.strip():
        validation_result['valid'] = False
        validation_result['errors'].append("No protein content provided")
    elif not ('ATOM' in protein_content or 'HETATM' in protein_content):
        validation_result['warnings'].append("Protein content may not be valid PDB format")
    
    # Validate fingerprint types
    if not molecular_fp_types:
        validation_result['valid'] = False
        validation_result['errors'].append("No molecular fingerprint types selected")
    
    valid_fp_types = ['morgan', 'rdkit', 'mapchiral']
    invalid_types = [fp for fp in molecular_fp_types if fp not in valid_fp_types]
    if invalid_types:
        validation_result['valid'] = False
        validation_result['errors'].append(f"Invalid fingerprint types: {invalid_types}")
    
    return validation_result


def create_reprocessing_debug_info(
    original_df,
    prepared_df,
    processing_configs: Dict[str, Any]
) -> List[str]:
    """
    Create debug information for reprocessing operations.
    Pure function - returns debug information list.
    
    Args:
        original_df: Original molecules DataFrame
        prepared_df: DataFrame prepared for reprocessing
        processing_configs: Processing configurations
    
    Returns:
        List of debug information strings
    """
    debug_info = []
    
    # DataFrame information
    debug_info.append(f"Original columns: {list(original_df.columns) if original_df is not None else 'None'}")
    debug_info.append(f"Prepared columns: {list(prepared_df.columns) if prepared_df is not None else 'None'}")
    
    if original_df is not None and len(original_df) > 0:
        sample_row = original_df.iloc[0]
        debug_info.append(f"Original Morgan FP: {type(sample_row.get('morgan_fp', 'Missing'))}")
        debug_info.append(f"Original RDKit FP: {type(sample_row.get('rdkit_fp', 'Missing'))}")
        debug_info.append(f"Original MapChiral FP: {type(sample_row.get('mapchiral_fp', 'Missing'))}")
    
    # Configuration information
    debug_info.append(f"Selected fingerprints: {processing_configs.get('molecular_fp_types', [])}")
    debug_info.append(f"Interaction type: {processing_configs.get('interaction_type', 'Unknown')}")
    debug_info.append(f"Pose quality: {processing_configs.get('compute_pose_quality', False)}")
    
    return debug_info