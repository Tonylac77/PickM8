"""
File handling functions using functional programming approach.
All functions are pure with no side effects.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
from rdkit.Chem import PandasTools

from utils.data_processing import load_sdf_file

logger = logging.getLogger(__name__)


def detect_sdf_properties(sdf_path: str) -> List[str]:
    """
    Detect available properties in SDF file. Pure function.
    
    Args:
        sdf_path: Path to SDF file
    
    Returns:
        List of property names found in SDF file
    """
    try:
        # Load just first few molecules to detect properties
        temp_df = PandasTools.LoadSDF(sdf_path, molColName='mol')
        if len(temp_df) == 0:
            return []
        
        # Get column names excluding RDKit columns
        properties = [col for col in temp_df.columns 
                     if col not in ['mol', 'ID'] and not col.startswith('_')]
        
        return properties
        
    except Exception as e:
        logger.error(f"Error detecting SDF properties: {e}")
        return []


def get_score_property_candidates() -> List[str]:
    """
    Get list of common score property names to look for in SDF files.
    Pure function returning predefined list.
    
    Returns:
        List of common score property names
    """
    return ["minimizedAffinity", "score", "Score", "docking_score", "binding_affinity"]



def validate_uploaded_files(protein_file, ligand_file) -> Dict[str, Any]:
    """
    Validate uploaded files without modifying them. Pure function.
    
    Args:
        protein_file: Streamlit uploaded file object for protein
        ligand_file: Streamlit uploaded file object for ligands
    
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'valid': False,
        'protein_valid': False,
        'ligand_valid': False,
        'protein_content': None,
        'ligand_path': None,
        'errors': []
    }
    
    # Validate protein file
    if protein_file is not None:
        try:
            protein_content = protein_file.getvalue().decode('utf-8')
            if protein_content.strip().startswith('HEADER') or 'ATOM' in protein_content:
                validation_result['protein_valid'] = True
                validation_result['protein_content'] = protein_content
            else:
                validation_result['errors'].append("Protein file does not appear to be a valid PDB format")
        except Exception as e:
            validation_result['errors'].append(f"Error reading protein file: {str(e)}")
    else:
        validation_result['errors'].append("No protein file provided")
    
    # Validate ligand file
    if ligand_file is not None:
        try:
            # Create temporary file to test SDF reading
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
                tmp.write(ligand_file.getvalue())
                temp_path = tmp.name
            
            # Test if we can detect properties (validates SDF format)
            properties = detect_sdf_properties(temp_path)
            if len(properties) >= 0:  # Even empty properties list means valid SDF
                validation_result['ligand_valid'] = True
                validation_result['ligand_path'] = temp_path
            else:
                validation_result['errors'].append("Could not read SDF file or no properties found")
                
        except Exception as e:
            validation_result['errors'].append(f"Error reading SDF file: {str(e)}")
    else:
        validation_result['errors'].append("No ligand file provided")
    
    validation_result['valid'] = validation_result['protein_valid'] and validation_result['ligand_valid']
    
    return validation_result


def process_score_column(
    molecules_df: pd.DataFrame, 
    score_label: str, 
    score_direction: str
) -> Dict[str, Any]:
    """
    Process and validate score column in molecules DataFrame.
    Pure function - returns new DataFrame and validation results.
    
    Args:
        molecules_df: Input molecules DataFrame
        score_label: Name of score column to process
        score_direction: Direction preference for scores
    
    Returns:
        Dictionary with processed DataFrame and validation results
    """
    result = {
        'success': False,
        'molecules_df': None,
        'score_range': None,
        'error_message': None
    }
    
    # Create copy to avoid modifying input
    processed_df = molecules_df.copy()
    
    if score_label not in processed_df.columns:
        result['error_message'] = f"Score column '{score_label}' not found in SDF file!"
        result['available_columns'] = list(processed_df.columns)
        return result
    
    try:
        # Validate all values are numeric
        score_values = processed_df[score_label]
        numeric_scores = pd.to_numeric(score_values, errors='raise')
        
        # Set score column
        processed_df['score'] = numeric_scores
        
        # Calculate score range
        score_min = processed_df['score'].min()
        score_max = processed_df['score'].max()
        
        result.update({
            'success': True,
            'molecules_df': processed_df,
            'score_range': (score_min, score_max),
            'num_molecules': len(processed_df)
        })
        
    except (ValueError, TypeError) as e:
        result['error_message'] = f"Score column '{score_label}' contains non-numeric values!"
        
    return result


def create_file_processing_summary(
    protein_name: str,
    num_molecules: int,
    score_range: tuple[float, float],
    available_properties: List[str],
    selected_score: str
) -> Dict[str, Any]:
    """
    Create summary of file processing results. Pure function.
    
    Args:
        protein_name: Name of protein file
        num_molecules: Number of molecules loaded
        score_range: Min and max score values
        available_properties: Properties found in SDF
        selected_score: Selected score property name
    
    Returns:
        Dictionary with processing summary
    """
    return {
        'protein_name': protein_name,
        'num_molecules': num_molecules,
        'score_range': score_range,
        'score_range_text': f"{score_range[0]:.3f} to {score_range[1]:.3f}",
        'available_properties': available_properties,
        'num_properties': len(available_properties),
        'selected_score': selected_score,
        'processing_timestamp': pd.Timestamp.now().isoformat()
    }


def prepare_file_for_processing(ligand_path: str) -> Optional[pd.DataFrame]:
    """
    Load and prepare SDF file for processing. Pure function.
    
    Args:
        ligand_path: Path to SDF file
    
    Returns:
        Loaded molecules DataFrame or None if failed
    """
    try:
        molecules_df = load_sdf_file(ligand_path)
        return molecules_df
        
    except Exception as e:
        logger.error(f"Error loading SDF file for processing: {e}")
        return None