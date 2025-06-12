#!/usr/bin/env python3
"""
ProLIF-based protein-ligand interaction fingerprint calculations.
Pure functional implementation for ProLIF interaction analysis.
"""

import numpy as np
import tempfile
import os
import logging
from rdkit import Chem

logger = logging.getLogger(__name__)

# ProLIF imports
try:
    import prolif as plf
    import MDAnalysis as mda
    PROLIF_AVAILABLE = True
except ImportError:
    PROLIF_AVAILABLE = False
    logger.warning("ProLIF not available. Install with: pip install prolif")


def is_prolif_available():
    """Check if ProLIF is available for use."""
    return PROLIF_AVAILABLE


def create_ligand_sdf(ligand_mol: 'Chem.Mol') -> str:
    """
    Create temporary SDF file for ligand.
    
    Args:
        ligand_mol: RDKit molecule object
        
    Returns:
        Path to temporary SDF file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp_sdf:
        sdf_writer = Chem.SDWriter(tmp_sdf.name)
        sdf_writer.write(ligand_mol)
        sdf_writer.close()
        return tmp_sdf.name


def extract_prolif_interactions(ifp_df) -> list:
    """
    Extract interaction data from ProLIF fingerprint DataFrame.
    
    Args:
        ifp_df: ProLIF fingerprint DataFrame
        
    Returns:
        List of interaction dictionaries
    """
    interaction_data = []
    
    for col in ifp_df.columns:
        if ifp_df[col].iloc[0] > 0:
            interaction_type = col[1] if len(col) > 1 else 'unknown'
            residue_info = col[0] if len(col) > 0 else 'unknown'
            
            interaction_data.append({
                'type': interaction_type,
                'residue': str(residue_info)
            })
    
    return interaction_data


def create_prolif_summary(interaction_data: list, ifp_array: np.ndarray) -> dict:
    """
    Create interaction summary from ProLIF data.
    
    Args:
        interaction_data: List of extracted interactions
        ifp_array: ProLIF fingerprint array
        
    Returns:
        Dictionary containing interaction summary
    """
    interactions_summary = {
        'total_interactions': int(np.sum(ifp_array > 0)),
        'interaction_types': {},
        'interactions': interaction_data
    }
    
    # Group by interaction type
    for interaction in interaction_data:
        int_type = interaction['type']
        if int_type in interactions_summary['interaction_types']:
            interactions_summary['interaction_types'][int_type] += 1
        else:
            interactions_summary['interaction_types'][int_type] = 1
    
    return interactions_summary


def calculate_prolif_interactions(protein_path: str, ligand_mol: 'Chem.Mol', ligand_name: str = "LIG") -> tuple:
    """
    Calculate ProLIF interactions for a protein-ligand complex.
    
    Args:
        protein_path: Path to protein PDB file
        ligand_mol: RDKit molecule object for the ligand
        ligand_name: Name for the ligand (default: "LIG")
        
    Returns:
        Tuple of (interaction_fingerprint, interaction_summary)
        
    Raises:
        ImportError: If ProLIF is not available
    """
    if not PROLIF_AVAILABLE:
        raise ImportError("ProLIF is not available. Install with: pip install prolif")
    
    # Create temporary SDF file for ligand
    tmp_sdf_path = create_ligand_sdf(ligand_mol)
    
    try:
        # Load structures with MDAnalysis
        protein_u = mda.Universe(protein_path)
        ligand_u = mda.Universe(tmp_sdf_path)
        
        # Create ProLIF fingerprint
        fp = plf.Fingerprint()
        
        # Calculate fingerprint
        ifp_df = fp.run(ligand_u.trajectory, ligand_u, protein_u)
        
        # Convert to numpy array format
        ifp_array = ifp_df.values.flatten()
        
        # Extract interaction details
        interaction_data = extract_prolif_interactions(ifp_df)
        
        # Create summary
        interactions_summary = create_prolif_summary(interaction_data, ifp_array)
        
        return ifp_array, interactions_summary
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_sdf_path):
            os.unlink(tmp_sdf_path)


def get_prolif_interaction_types() -> list:
    """
    Get list of interaction types supported by ProLIF.
    
    Returns:
        List of supported interaction type names
    """
    return [
        'HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiStacking',
        'Anionic', 'Cationic', 'CationPi', 'PiCation',
        'XBAcceptor', 'XBDonor'
    ]