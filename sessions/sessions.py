"""Session management functions using functional programming approach."""
import uuid
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generate a new unique session ID."""
    return str(uuid.uuid4())

def create_session(
    protein_file: Any,
    ligand_file: Any,
    score_label: str,
    score_direction: str,
    config: Dict[str, Any]
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Create a new session with processed molecules.

    Chain-of-Thought:
    - Direct function without service layer abstraction
    - Returns all needed data in one call
    - Caller handles persistence
    """
    # Work around Python's built-in io module conflict
    import sys
    import os
    from data_io import molecules
    from features.fingerprints import fingerprints
    from features.interactions import interactions
    from analysis import pose_quality

    # Generate session ID
    session_id = generate_session_id()

    # Load and process molecules
    df = molecules.load_sdf(ligand_file)
    df = molecules.process_score_column(df, score_label, score_direction)

    # Read protein content
    protein_content = protein_file.read().decode('utf-8')

    # Compute fingerprints
    if config.get('compute_fingerprints', True):
        logger.info("Computing molecular fingerprints...")
        try:
            df = fingerprints.compute_all_fingerprints(
                df, protein_content, config['fingerprint_config']
            )
            logger.info("Fingerprint computation completed successfully")
        except Exception as e:
            logger.error(f"Error computing fingerprints: {e}")
            raise RuntimeError(f"Fingerprint computation failed: {e}")

    # Compute interactions
    if config.get('compute_interactions', True):
        logger.info("Computing protein-ligand interactions...")
        try:
            df = interactions.compute_all_interactions(
                df, protein_content, config['interaction_config']
            )
            logger.info("Interaction computation completed successfully")
        except Exception as e:
            logger.error(f"Error computing interactions: {e}")
            raise RuntimeError(f"Interaction computation failed: {e}")

    # Compute GRADE descriptors
    if config.get('compute_grade_descriptors', False):
        logger.info("Computing GRADE descriptors...")
        try:
            from features.grade_descriptors import grade_descriptors
            from data_io.molecules.molecules import add_grade_columns
            
            # Add GRADE columns to DataFrame
            df = add_grade_columns(df, config['grade_config'].get('extended', False))
            
            # Compute GRADE descriptors
            df = grade_descriptors.compute_all_grade_descriptors(
                df, protein_content, config['grade_config']
            )
            logger.info("GRADE descriptor computation completed successfully")
        except Exception as e:
            logger.error(f"Error computing GRADE descriptors: {e}")
            raise RuntimeError(f"GRADE descriptor computation failed: {e}")

    # Analyze pose quality
    if config.get('compute_pose_quality', True):
        logger.info("Analyzing pose quality...")
        try:
            df = pose_quality.analyze_all_poses(
                df, protein_content, config['pose_config']
            )
            logger.info("Pose quality analysis completed successfully")
        except Exception as e:
            logger.error(f"Error analyzing pose quality: {e}")
            raise RuntimeError(f"Pose quality analysis failed: {e}")

    # Create metadata
    metadata = {
        'session_id': session_id,
        'protein_name': protein_file.name,
        'protein_content': protein_content,
        'num_molecules': len(df),
        'score_label': score_label,
        'score_direction': score_direction,
        'created_date': datetime.now().isoformat(),
        'config': config
    }

    return session_id, df, metadata

def save_session(session_id: str, df: pd.DataFrame, metadata: Dict[str, Any]) -> bool:
    """Save session data to disk."""
    try:
        session_dir = Path(f"data/sessions/{session_id}")
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrame
        df.to_pickle(session_dir / "molecules.pkl")

        # Save metadata
        with open(session_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved session {session_id}")
        return True
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return False

def load_session(session_id: str) -> Optional[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Load session data from disk."""
    try:
        session_dir = Path(f"data/sessions/{session_id}")

        # Load DataFrame
        df = pd.read_pickle(session_dir / "molecules.pkl")

        # Load metadata
        with open(session_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)

        return df, metadata
    except Exception as e:
        logger.error(f"Error loading session {session_id}: {e}")
        return None

def list_sessions() -> List[Dict[str, Any]]:
    """Get list of all sessions with summary info."""
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists():
        return []

    sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            try:
                # Load just metadata for summary
                with open(session_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)

                sessions.append({
                    'session_id': session_dir.name,
                    'created_date': metadata.get('created_date'),
                    'protein_name': metadata.get('protein_name'),
                    'num_molecules': metadata.get('num_molecules', 0)
                })
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load metadata for session {session_dir.name}: {e}")
                continue

    # Sort sessions by created_date, handling None values by putting them at the end
    def sort_key(session):
        date = session['created_date']
        if date is None:
            return (1, '')  # None dates go last
        return (0, date)
    
    return sorted(sessions, key=sort_key, reverse=True)