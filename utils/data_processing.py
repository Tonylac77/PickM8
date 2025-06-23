"""
Data processing utilities using functional programming approach.
Handles loading, saving, and transforming molecular data in a single DataFrame.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from rdkit import Chem
from rdkit.Chem import PandasTools
import pickle

logger = logging.getLogger(__name__)


def create_empty_molecules_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with the standard molecule schema."""
    columns = {
        # Core molecule data
        'id': 'int64',
        'name': 'object',
        'smiles': 'object', 
        'mol_block': 'object',
        'mol': 'object',  # RDKit Mol object for PandasTools compatibility
        
        # Original SDF scores/properties (will be dynamically added)
        'score': 'float64',
        
        # Computed fingerprints
        'morgan_fp': 'object',  # List of integers
        'rdkit_fp': 'object',   # List of integers
        'mapchiral_fp': 'object',  # List of integers
        'interaction_fp': 'object',  # JSON string
        'interactions': 'object',    # JSON string
        'num_interactions': 'int64',
        
        # User grades
        'grade': 'object',      # 'A', 'B', 'C', 'D', 'F', or None
        'grade_timestamp': 'datetime64[ns]',
        
        # Pose quality metrics
        'clashes': 'int64',
        'strain_energy': 'float64',
        
        # ML predictions
        'prediction': 'object',  # Can store grade letters or numeric values
        'prediction_uncertainty': 'float64',
        'prediction_timestamp': 'datetime64[ns]'
    }
    
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in columns.items()})


def load_sdf_file(sdf_path: str) -> pd.DataFrame:
    """
    Load molecules from SDF file into DataFrame.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        DataFrame with molecules and their properties
    """
    try:
        # Use RDKit PandasTools to load SDF
        df = PandasTools.LoadSDF(sdf_path, molColName='mol', includeFingerprints=False)
        
        # Ensure we have required columns
        if 'mol' not in df.columns:
            raise ValueError("No valid molecules found in SDF file")
            
        # Create standardized columns
        df['id'] = range(len(df))
        df['name'] = df.get('ID', df.get('Name', [f"mol_{i}" for i in range(len(df))]))
        df['smiles'] = df['mol'].apply(lambda m: Chem.MolToSmiles(m) if m else None)
        df['mol_block'] = df['mol'].apply(lambda m: Chem.MolToMolBlock(m) if m else None)
        
        # Initialize score column - will be set properly in main.py
        df['score'] = 0.0
            
        # Initialize computed columns
        df['morgan_fp'] = None
        df['rdkit_fp'] = None
        df['mapchiral_fp'] = None
        df['interaction_fp'] = None
        df['interactions'] = None
        df['num_interactions'] = 0
        
        # Initialize grade columns
        df['grade'] = None
        df['grade_timestamp'] = pd.NaT
        
        # Initialize pose quality columns
        df['clashes'] = 0
        df['strain_energy'] = 0.0
        
        # Initialize ML prediction columns
        df['prediction'] = None
        df['prediction_uncertainty'] = np.nan
        df['prediction_timestamp'] = pd.NaT
        
        logger.info(f"Loaded {len(df)} molecules from {sdf_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading SDF file {sdf_path}: {e}")
        raise


def load_pdb_file(pdb_path: str) -> str:
    """
    Load protein PDB file content.
    
    Args:
        pdb_path: Path to PDB file
        
    Returns:
        PDB file content as string
    """
    try:
        with open(pdb_path, 'r') as f:
            content = f.read()
        logger.info(f"Loaded PDB file {pdb_path}")
        return content
    except Exception as e:
        logger.error(f"Error loading PDB file {pdb_path}: {e}")
        raise


def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure computed numeric columns have proper dtypes.
    Note: Score column is already validated as numeric during data loading.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with corrected column types
    """
    df = df.copy()
    
    # Ensure computed numeric columns are properly typed
    numeric_columns = ['clashes', 'strain_energy', 'num_interactions', 'prediction_uncertainty']
    
    for col in numeric_columns:
        if col in df.columns:
            if col in ['clashes', 'num_interactions']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('Int64')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure prediction column remains as object type for storing grade letters
    if 'prediction' in df.columns:
        df['prediction'] = df['prediction'].astype('object')
    
    return df


def save_molecules_dataframe(df: pd.DataFrame, session_dir: str) -> None:
    """
    Save molecules DataFrame to session directory.
    
    Args:
        df: Molecules DataFrame
        session_dir: Session directory path
    """
    try:
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure proper column types before saving
        df = ensure_numeric_columns(df)
        
        # Save as pickle to preserve all data types including RDKit Mol objects
        pickle_path = session_path / "molecules.pkl"
        df.to_pickle(pickle_path)
        
        logger.info(f"Saved {len(df)} molecules to {pickle_path}")
        
    except Exception as e:
        logger.error(f"Error saving molecules DataFrame: {e}")
        raise


def load_molecules_dataframe(session_dir: str) -> Optional[pd.DataFrame]:
    """
    Load molecules DataFrame from session directory.
    
    Args:
        session_dir: Session directory path
        
    Returns:
        Molecules DataFrame or None if not found
    """
    try:
        pickle_path = Path(session_dir) / "molecules.pkl"
        
        if not pickle_path.exists():
            logger.warning(f"No molecules file found at {pickle_path}")
            return None
            
        df = pd.read_pickle(pickle_path)
        logger.info(f"Loaded {len(df)} molecules from {pickle_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading molecules DataFrame: {e}")
        return None


def add_grade_to_molecule(df: pd.DataFrame, molecule_id: int, grade: str) -> pd.DataFrame:
    """
    Add or update grade for a specific molecule.
    
    Args:
        df: Molecules DataFrame
        molecule_id: ID of molecule to grade
        grade: Grade ('A', 'B', 'C', 'D', 'F')
        
    Returns:
        Updated DataFrame
    """
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
    """
    Get subset of molecules that have been graded.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        DataFrame with only graded molecules
    """
    return df[df['grade'].notna()].copy()


def get_ungraded_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get subset of molecules that have not been graded.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        DataFrame with only ungraded molecules
    """
    return df[df['grade'].isna()].copy()


def update_pose_quality_metrics(df: pd.DataFrame, pose_metrics: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Update DataFrame with pose quality metrics.
    
    Args:
        df: Molecules DataFrame
        pose_metrics: Dict mapping molecule_id to metrics dict
        
    Returns:
        Updated DataFrame
    """
    df = df.copy()
    
    for mol_id, metrics in pose_metrics.items():
        mask = df['id'] == mol_id
        if mask.any():
            df.loc[mask, 'clashes'] = metrics.get('clashes', 0)
            df.loc[mask, 'strain_energy'] = metrics.get('strain_energy', 0.0)
    
    logger.info(f"Updated pose quality metrics for {len(pose_metrics)} molecules")
    return df


def update_ml_predictions(df: pd.DataFrame, predictions: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """
    Update DataFrame with ML predictions and uncertainties.
    
    Args:
        df: Molecules DataFrame  
        predictions: Dict mapping molecule_id to prediction dict
        
    Returns:
        Updated DataFrame
    """
    df = df.copy()
    timestamp = pd.Timestamp.now()
    
    for mol_id, pred_data in predictions.items():
        mask = df['id'] == mol_id
        if mask.any():
            df.loc[mask, 'prediction'] = pred_data.get('prediction', np.nan)
            df.loc[mask, 'prediction_uncertainty'] = pred_data.get('uncertainty', np.nan)
            df.loc[mask, 'prediction_timestamp'] = timestamp
    
    logger.info(f"Updated ML predictions for {len(predictions)} molecules")
    return df


def get_molecule_features(df: pd.DataFrame, include_predictions: bool = False) -> Tuple[np.ndarray, List[int]]:
    """
    Extract feature matrix and molecule IDs for ML training/prediction.
    
    Args:
        df: Molecules DataFrame
        include_predictions: Whether to include prediction features
        
    Returns:
        Tuple of (feature_matrix, molecule_ids)
    """
    # Get molecules with computed fingerprints
    valid_mask = (df['morgan_fp'].notna()) & (df['interaction_fp'].notna())
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return np.array([]), []
    
    features = []
    for _, row in valid_df.iterrows():
        # Combine molecular fingerprints
        feature_vector = []
        
        # Morgan fingerprint
        if row['morgan_fp'] is not None:
            feature_vector.extend(row['morgan_fp'])
        
        # RDKit fingerprint  
        if row['rdkit_fp'] is not None:
            feature_vector.extend(row['rdkit_fp'])
            
        # Interaction fingerprint (convert JSON to numeric)
        if row['interaction_fp'] is not None:
            try:
                ifp_data = json.loads(row['interaction_fp'])
                if isinstance(ifp_data, list):
                    feature_vector.extend(ifp_data)
                elif isinstance(ifp_data, dict):
                    # Convert dict values to list
                    feature_vector.extend(list(ifp_data.values()))
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Add additional features if requested
        if include_predictions and pd.notna(row['prediction']):
            feature_vector.append(row['prediction'])
            feature_vector.append(row['prediction_uncertainty'])
            
        features.append(feature_vector)
    
    # Ensure all feature vectors have the same length
    if features:
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]
    
    return np.array(features), valid_df['id'].tolist()


def save_session_metadata(session_dir: str, metadata: Dict[str, Any]) -> None:
    """
    Save session metadata to JSON file.
    
    Args:
        session_dir: Session directory path
        metadata: Metadata dictionary
    """
    try:
        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        
        metadata_path = session_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logger.info(f"Saved session metadata to {metadata_path}")
        
    except Exception as e:
        logger.error(f"Error saving session metadata: {e}")
        raise


def load_session_metadata(session_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load session metadata from JSON file.
    
    Args:
        session_dir: Session directory path
        
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        metadata_path = Path(session_dir) / "metadata.json"
        
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        logger.info(f"Loaded session metadata from {metadata_path}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error loading session metadata: {e}")
        return None