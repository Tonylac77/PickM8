"""Molecule data operations using functional programming."""
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from rdkit import Chem
from rdkit.Chem import PandasTools

logger = logging.getLogger(__name__)

def create_empty_dataframe() -> pd.DataFrame:
    """Create empty DataFrame with standard molecule schema."""
    return pd.DataFrame({
        'id': pd.Series(dtype='int64'),
        'name': pd.Series(dtype='object'),
        'smiles': pd.Series(dtype='object'),
        'mol_block': pd.Series(dtype='object'),
        'mol': pd.Series(dtype='object'),
        'score': pd.Series(dtype='float64'),
        'morgan_fp': pd.Series(dtype='object'),
        'rdkit_fp': pd.Series(dtype='object'),
        'mapchiral_fp': pd.Series(dtype='object'),
        'interaction_fp': pd.Series(dtype='object'),
        'interactions': pd.Series(dtype='object'),
        'num_interactions': pd.Series(dtype='int64'),
        'grade': pd.Series(dtype='object'),
        'grade_timestamp': pd.Series(dtype='datetime64[ns]'),
        'clashes': pd.Series(dtype='int64'),
        'strain_energy': pd.Series(dtype='float64'),
        'prediction': pd.Series(dtype='object'),
        'prediction_uncertainty': pd.Series(dtype='float64'),
        'prediction_timestamp': pd.Series(dtype='datetime64[ns]')
    })

def load_sdf(filepath) -> pd.DataFrame:
    """
    Load molecules from SDF file.

    Chain-of-Thought:
    - Single responsibility: just load SDF
    - Return clean DataFrame ready for processing
    - No side effects or hidden transformations
    - Handle both file paths (str) and Streamlit file upload objects
    """
    import tempfile
    import os
    
    try:
        # Handle Streamlit UploadedFile objects
        if hasattr(filepath, 'read'):
            # This is a file-like object (Streamlit UploadedFile)
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.sdf', delete=False) as tmp_file:
                # Write the content to temporary file
                filepath.seek(0)  # Reset file pointer to beginning
                tmp_file.write(filepath.read())
                tmp_file.flush()
                
                # Load from temporary file
                df = PandasTools.LoadSDF(tmp_file.name, molColName='mol', idName="ID")
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
        else:
            # This is a regular file path string
            df = PandasTools.LoadSDF(filepath, molColName='mol', idName="ID")

        if 'mol' not in df.columns or len(df) == 0:
            raise ValueError("No valid molecules found in SDF file")

        # Initialize required columns
        df['id'] = range(len(df))
        df['name'] = df.get('ID', [f"mol_{i}" for i in range(len(df))])
        df['smiles'] = df['mol'].apply(lambda m: Chem.MolToSmiles(m) if m else None)
        df['mol_block'] = df['mol'].apply(lambda m: Chem.MolToMolBlock(m) if m else None)

        # Initialize computed columns with defaults
        for col in ['morgan_fp', 'rdkit_fp', 'mapchiral_fp', 'interaction_fp',
                    'interactions', 'grade', 'prediction']:
            df[col] = None

        for col in ['num_interactions', 'clashes']:
            df[col] = 0

        for col in ['score', 'strain_energy', 'prediction_uncertainty']:
            df[col] = 0.0

        for col in ['grade_timestamp', 'prediction_timestamp']:
            df[col] = pd.NaT

        logger.info(f"Loaded {len(df)} molecules from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading SDF file: {e}")
        raise

def process_score_column(
    df: pd.DataFrame,
    score_label: str,
    score_direction: str
) -> pd.DataFrame:
    """Process and validate score column."""
    df = df.copy()

    if score_label not in df.columns:
        raise ValueError(f"Score column '{score_label}' not found")

    # Validate numeric scores
    df['score'] = pd.to_numeric(df[score_label], errors='coerce')

    if df['score'].isna().any():
        raise ValueError(f"Score column contains non-numeric values")

    logger.info(f"Processed scores from '{score_label}' column")
    return df

def detect_sdf_properties(filepath) -> List[str]:
    """Detect available properties in SDF file."""
    import tempfile
    import os
    
    try:
        # Handle Streamlit UploadedFile objects
        if hasattr(filepath, 'read'):
            # This is a file-like object (Streamlit UploadedFile)
            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.sdf', delete=False) as tmp_file:
                # Write the content to temporary file
                filepath.seek(0)  # Reset file pointer to beginning
                tmp_file.write(filepath.read())
                tmp_file.flush()
                
                # Load from temporary file
                temp_df = PandasTools.LoadSDF(tmp_file.name, molColName='mol')
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
        else:
            # This is a regular file path string
            temp_df = PandasTools.LoadSDF(filepath, molColName='mol')
            
        if len(temp_df) == 0:
            return []

        # Get property columns (exclude RDKit internal columns)
        properties = [col for col in temp_df.columns
                      if col not in ['mol', 'ID'] and not col.startswith('_')]

        return properties
    except Exception as e:
        logger.error(f"Error detecting SDF properties: {e}")
        return []

# Legacy functions for test compatibility
def save_molecules_dataframe(df: pd.DataFrame, dirpath: str) -> bool:
    """Save molecules DataFrame to directory (legacy function for tests)."""
    try:
        from pathlib import Path
        filepath = Path(dirpath) / "molecules.pkl"
        df.to_pickle(filepath)
        logger.info(f"Saved molecules DataFrame to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving molecules DataFrame: {e}")
        return False

def load_molecules_dataframe(dirpath: str) -> Optional[pd.DataFrame]:
    """Load molecules DataFrame from directory (legacy function for tests)."""
    try:
        from pathlib import Path
        filepath = Path(dirpath) / "molecules.pkl"
        if not filepath.exists():
            return None
        df = pd.read_pickle(filepath)
        logger.info(f"Loaded molecules DataFrame from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading molecules DataFrame: {e}")
        return None

def add_grade_to_molecule(df: pd.DataFrame, molecule_id: int, grade: str) -> pd.DataFrame:
    """Add grade to molecule (legacy function for tests - redirects to grading module)."""
    from analysis import grading
    return grading.add_grade(df, molecule_id, grade)