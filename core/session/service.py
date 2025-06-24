"""
Session management service using functional programming approach.
All functions are pure with no side effects.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from utils.data_processing import (
    load_molecules_dataframe,
    load_session_metadata,
    save_molecules_dataframe,
    save_session_metadata,
    load_sdf_file
)

logger = logging.getLogger(__name__)


def generate_session_id() -> str:
    """Generate a new unique session ID."""
    return str(uuid.uuid4())


def create_session_directory_path(session_id: str, base_dir: str = "data/sessions") -> str:
    """Create session directory path from session ID."""
    return f"{base_dir}/{session_id}"


def create_new_session(
    session_id: str,
    protein_name: str,
    protein_content: str,
    molecules_df,
    score_label: str,
    score_direction: str,
    interaction_type: str,
    compute_pose_quality: bool,
    available_properties: List[str]
) -> Dict[str, Any]:
    """
    Create a new session with metadata. Pure function - returns session data without side effects.
    
    Args:
        session_id: Unique session identifier
        protein_name: Name of protein file
        protein_content: PDB file content
        molecules_df: Processed molecules DataFrame
        score_label: Selected score column name
        score_direction: Score interpretation direction
        interaction_type: Type of interaction analysis (plip/prolif)
        compute_pose_quality: Whether pose quality was computed
        available_properties: List of available SDF properties
    
    Returns:
        Dictionary containing session metadata
    """
    session_metadata = {
        'protein_name': protein_name,
        'protein_content': protein_content,
        'num_molecules': len(molecules_df),
        'score_label': score_label,
        'score_direction': score_direction,
        'created_date': datetime.now().isoformat(),
        'interaction_type': interaction_type,
        'compute_pose_quality': compute_pose_quality,
        'available_properties': available_properties
    }
    
    return {
        'session_id': session_id,
        'metadata': session_metadata,
        'molecules_df': molecules_df
    }


def load_session_data(session_id: str, base_dir: str = "data/sessions") -> Optional[Dict[str, Any]]:
    """
    Load session data by ID. Pure function - returns data without side effects.
    
    Args:
        session_id: Session identifier to load
        base_dir: Base directory for sessions
    
    Returns:
        Dictionary with session data or None if not found
    """
    session_dir = create_session_directory_path(session_id, base_dir)
    
    try:
        # Load molecules DataFrame
        molecules_df = load_molecules_dataframe(session_dir)
        if molecules_df is None:
            return None
        
        # Load session metadata
        metadata = load_session_metadata(session_dir)
        if metadata is None:
            return None
        
        return {
            'session_id': session_id,
            'session_dir': session_dir,
            'molecules_df': molecules_df,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return None


def prepare_reprocessing_data(
    molecules_df,
    molecular_fp_types: List[str],
    interaction_type: str,
    compute_pose_quality: bool
) -> Tuple[Any, List[str]]:
    """
    Prepare DataFrame for reprocessing by clearing computed columns.
    Pure function - returns new DataFrame without modifying input.
    
    Args:
        molecules_df: Original molecules DataFrame
        molecular_fp_types: Selected molecular fingerprint types
        interaction_type: Selected interaction type
        compute_pose_quality: Whether to compute pose quality
    
    Returns:
        Tuple of (prepared_df, debug_info)
    """
    # Create copy to avoid modifying original
    reprocess_df = molecules_df.copy()
    
    # Define columns to clear (preserve grades)
    computed_columns = [
        'morgan_fp', 'rdkit_fp', 'mapchiral_fp', 'interaction_fp',
        'interactions', 'num_interactions', 'clashes', 'strain_energy',
        'prediction', 'prediction_uncertainty', 'prediction_timestamp'
    ]
    
    debug_info = [
        f"Original columns: {list(molecules_df.columns)}",
        f"Selected fingerprints: {molecular_fp_types}",
        f"Interaction type: {interaction_type}",
        f"Pose quality: {compute_pose_quality}"
    ]
    
    # Clear computed columns (but preserve grades)
    for col in computed_columns:
        if col in reprocess_df.columns:
            if col not in ['grade', 'grade_timestamp']:
                reprocess_df[col] = None
    
    return reprocess_df, debug_info


def reprocess_session_data(
    session_data: Dict[str, Any],
    molecular_fp_types: List[str],
    interaction_type: str,
    compute_pose_quality: bool
) -> Dict[str, Any]:
    """
    Prepare session data for reprocessing. Pure function - returns new data structure.
    
    Args:
        session_data: Original session data
        molecular_fp_types: Selected molecular fingerprint types
        interaction_type: Selected interaction type
        compute_pose_quality: Whether to compute pose quality
    
    Returns:
        Dictionary with prepared reprocessing data
    """
    prepared_df, debug_info = prepare_reprocessing_data(
        session_data['molecules_df'],
        molecular_fp_types,
        interaction_type,
        compute_pose_quality
    )
    
    # Update metadata for reprocessing
    updated_metadata = session_data['metadata'].copy()
    updated_metadata.update({
        'interaction_type': interaction_type,
        'compute_pose_quality': compute_pose_quality,
        'last_reprocessed': datetime.now().isoformat(),
        'molecular_fingerprints': molecular_fp_types
    })
    
    return {
        'session_id': session_data['session_id'],
        'session_dir': session_data['session_dir'],
        'molecules_df': prepared_df,
        'metadata': updated_metadata,
        'debug_info': debug_info
    }


def get_session_list(base_dir: str = "data/sessions") -> List[Dict[str, Any]]:
    """
    Get list of existing sessions with metadata. Pure function.
    
    Args:
        base_dir: Base directory for sessions
    
    Returns:
        List of session information dictionaries
    """
    sessions_dir = Path(base_dir)
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            try:
                # Load session metadata
                metadata = load_session_metadata(str(session_dir))
                if metadata is None:
                    continue
                
                # Load molecules to get counts
                molecules_df = load_molecules_dataframe(str(session_dir))
                
                num_molecules = len(molecules_df) if molecules_df is not None else 0
                num_grades = 0
                
                if molecules_df is not None and 'grade' in molecules_df.columns:
                    num_grades = molecules_df['grade'].notna().sum()
                
                # Get last modified time
                molecule_file = session_dir / "molecules.pkl"
                last_modified = datetime.fromtimestamp(
                    molecule_file.stat().st_mtime if molecule_file.exists() 
                    else session_dir.stat().st_mtime
                )
                
                sessions.append({
                    'session_id': session_dir.name,
                    'session_id_short': session_dir.name[:8],
                    'protein_name': metadata.get('protein_name', 'Unknown'),
                    'num_molecules': num_molecules,
                    'num_grades': num_grades,
                    'last_modified': last_modified,
                    'score_label': metadata.get('score_label', 'score'),
                    'created_date': metadata.get('created_date', 'Unknown'),
                    'interaction_type': metadata.get('interaction_type', 'Unknown'),
                    'compute_pose_quality': metadata.get('compute_pose_quality', False)
                })
                
            except Exception as e:
                logger.warning(f"Could not read session {session_dir.name}: {str(e)}")
                continue
    
    # Sort by last modified (newest first)
    sessions.sort(key=lambda x: x['last_modified'], reverse=True)
    return sessions


def save_session_data(session_data: Dict[str, Any]) -> bool:
    """
    Save session data to disk. Returns success status.
    
    Args:
        session_data: Session data dictionary
    
    Returns:
        True if successful, False otherwise
    """
    try:
        session_dir = session_data.get('session_dir') or create_session_directory_path(
            session_data['session_id']
        )
        
        # Save molecules DataFrame
        save_molecules_dataframe(session_data['molecules_df'], session_dir)
        
        # Save session metadata
        save_session_metadata(session_dir, session_data['metadata'])
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving session data: {e}")
        return False


def prepare_file_for_processing(sdf_path: str):
    """
    Prepare SDF file for processing by loading and validating molecules.
    Pure function - returns molecules DataFrame or None.
    
    Args:
        sdf_path: Path to SDF file
    
    Returns:
        Molecules DataFrame or None if loading fails
    """
    try:
        return load_sdf_file(sdf_path)
    except Exception as e:
        logger.error(f"Error loading SDF file: {e}")
        return None


def find_default_score_property(available_properties: List[str]) -> Tuple[Optional[str], int]:
    """
    Find default score property from available properties.
    Pure function - returns best match and its index.
    
    Args:
        available_properties: List of available property names
    
    Returns:
        Tuple of (best matching property name or None, index in list)
    """
    if not available_properties:
        return None, 0
    
    # Common score property names to look for (ordered by preference)
    score_candidates = [
        'score', 'Score', 'SCORE',
        'docking_score', 'DockingScore', 'Docking_Score',
        'binding_affinity', 'BindingAffinity', 'Binding_Affinity',
        'energy', 'Energy', 'ENERGY',
        'glide_score', 'GlideScore', 'Glide_Score'
    ]
    
    # Look for exact matches first
    for candidate in score_candidates:
        if candidate in available_properties:
            index = available_properties.index(candidate)
            return candidate, index
    
    # Look for partial matches (case-insensitive)
    for candidate in score_candidates:
        for i, prop in enumerate(available_properties):
            if candidate.lower() in prop.lower():
                return prop, i
    
    # If no good match, return first property that looks numeric
    for i, prop in enumerate(available_properties):
        prop_lower = prop.lower()
        if any(keyword in prop_lower for keyword in ['score', 'energy', 'affinity', 'dock']):
            return prop, i
    
    # Default to first property
    return available_properties[0], 0


def create_processing_statistics_summary(
    processing_summary: Dict[str, Any],
    fingerprint_stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive processing statistics summary.
    Pure function - returns summary data.
    
    Args:
        processing_summary: Processing results summary
        fingerprint_stats: Fingerprint computation statistics
    
    Returns:
        Dictionary containing comprehensive statistics
    """
    summary = {
        'processing_success': processing_summary.get('success', False),
        'total_molecules': processing_summary.get('total_molecules', 0),
        'successful_molecules': processing_summary.get('successful_molecules', 0),
        'failed_molecules': processing_summary.get('failed_molecules', 0),
        'processing_time': processing_summary.get('processing_time', 0),
        'fingerprint_stats': fingerprint_stats or {}
    }
    
    # Calculate success rate
    if summary['total_molecules'] > 0:
        summary['success_rate'] = summary['successful_molecules'] / summary['total_molecules']
    else:
        summary['success_rate'] = 0.0
    
    return summary