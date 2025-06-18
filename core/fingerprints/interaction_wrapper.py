"""
Interaction fingerprint calculations.
Handles PLIP and ProLIF interaction fingerprints for protein-ligand complexes.
"""

import json
import tempfile
import os
import logging
from typing import Dict, Any, Tuple, Optional
from rdkit import Chem

# Import interaction calculation modules
try:
    from .interactions.functions import calculate_interactions
    INTERACTION_AVAILABLE = True
except ImportError:
    INTERACTION_AVAILABLE = False
    logging.warning("Interaction functions not available")

logger = logging.getLogger(__name__)


def compute_interaction_fingerprint(mol: Chem.Mol, protein_content: str, 
                                  interaction_config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], int]:
    """
    Compute interaction fingerprint for a molecule with protein.
    
    Args:
        mol: RDKit molecule object
        protein_content: PDB content as string
        interaction_config: Configuration for interaction calculation
        
    Returns:
        Tuple of (interaction_fp_json, interactions_json, num_interactions)
    """
    try:
        if not INTERACTION_AVAILABLE or mol is None:
            return None, None, 0
            
        # Create temporary protein file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(protein_content)
            protein_path = f.name
            
        try:
            # Calculate interactions using functional approach
            ifp_type = interaction_config.get('interaction_type', 'plip')
            interaction_fp, interaction_summary = calculate_interactions(
                protein_path, mol, "LIG", ifp_type, interaction_config
            )
            
            # Extract interaction details
            interaction_details = interaction_summary.get('interactions', [])
            
            # Convert to JSON strings
            interaction_fp_json = json.dumps(interaction_fp.tolist()) if interaction_fp is not None else None
            interactions_json = json.dumps(interaction_details) if interaction_details else None
            num_interactions = len(interaction_details) if interaction_details else 0
            
            return interaction_fp_json, interactions_json, num_interactions
            
        finally:
            # Clean up temporary file
            if os.path.exists(protein_path):
                os.unlink(protein_path)
        
    except Exception as e:
        logger.error(f"Error computing interaction fingerprint: {e}")
        return None, None, 0


def create_default_interaction_config() -> Dict[str, Any]:
    """
    Create default configuration for interaction calculation.
    
    Returns:
        Default interaction configuration
    """
    return {
        "interaction_type": "plip",  # or "prolif"
        "plip_config": {
            "hydrogen_bonds": True,
            "hydrophobic_contacts": True,
            "pi_stacking": True,
            "salt_bridges": True,
            "halogen_bonds": True
        },
        "prolif_config": {
            "interactions": ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", "Anionic", "Cationic"]
        }
    }