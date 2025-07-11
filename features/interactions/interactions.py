"""Protein-ligand interaction functions for the flat architecture."""
import json
import tempfile
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO
from io import StringIO
import string
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

# PLIP imports
try:
    from plip.structure.preparation import PDBComplex
    PLIP_AVAILABLE = True
except ImportError:
    PLIP_AVAILABLE = False
    logger.warning("PLIP not available. Install with: pip install plip")

# ProLIF imports
try:
    import prolif as plf
    import MDAnalysis as mda
    PROLIF_AVAILABLE = True
except ImportError:
    PROLIF_AVAILABLE = False
    logger.warning("ProLIF not available. Install with: pip install prolif")

# Set overall interaction availability
INTERACTION_AVAILABLE = PLIP_AVAILABLE or PROLIF_AVAILABLE

def is_plip_available():
    """Check if PLIP is available for use."""
    return PLIP_AVAILABLE

def is_prolif_available():
    """Check if ProLIF is available for use."""
    return PROLIF_AVAILABLE

# BioPython complex creation function
def create_complex_with_biopython(protein_path: str, ligand_mol: 'Chem.Mol', ligand_name: str) -> Optional[str]:
    """
    Merges a protein PDB file and an RDKit ligand molecule into a single
    complex PDB file using BioPython.

    Args:
        protein_path: The file path to the protein's PDB file.
        ligand_mol: The RDKit Mol object for the ligand.
        ligand_name: The name for the ligand (used for residue name, max 3 chars).

    Returns:
        The file path to the newly created complex PDB file, or None on failure.
    """
    ligand_pdb_block = Chem.MolToPDBBlock(ligand_mol)
    ligand_io = StringIO(ligand_pdb_block)

    complex_path = None
    try:
        parser = PDBParser(QUIET=True)
        protein_struct = parser.get_structure('protein', protein_path)
        ligand_struct = parser.get_structure('ligand', ligand_io)

        protein_model = protein_struct[0]
        ligand_chain = next(ligand_struct.get_chains())
        ligand_residue = next(ligand_chain.get_residues())

        ligand_residue.resname = ligand_name[:3].upper()

        existing_chain_ids = {chain.id for chain in protein_model}
        new_chain_id = 'L'
        if new_chain_id in existing_chain_ids:
            available_ids = (c for c in string.ascii_uppercase if c not in existing_chain_ids)
            try:
                new_chain_id = next(available_ids)
            except StopIteration:
                logger.error("No available chain IDs left (A-Z are all taken).")
                return None

        ligand_chain.id = new_chain_id
        protein_model.add(ligand_chain)

        io = PDBIO()
        io.set_structure(protein_struct)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False, encoding='utf-8') as complex_file:
            complex_path = complex_file.name
            io.save(complex_file)

        logger.info(f"Successfully created complex with BioPython: {complex_path}")
        return complex_path

    except Exception as e:
        logger.warning(f"BioPython complex creation failed: {e}", exc_info=True)
        if complex_path and os.path.exists(complex_path):
            os.unlink(complex_path)
        return None

# PLIP functions
def extract_plip_interactions(interaction_set) -> list:
    """
    Extract interaction data from PLIP interaction set.
    
    Args:
        interaction_set: PLIP interaction set object
        
    Returns:
        List of interaction tuples (restype, resnr, reschain, interaction_type)
    """
    interaction_data = []
    
    for hbond in interaction_set.hbonds_ldon + interaction_set.hbonds_pdon:
        interaction_data.append((
            hbond.restype, hbond.resnr, hbond.reschain, 'hydrogen_bond'
        ))
    
    for hydrophobic in interaction_set.hydrophobic_contacts:
        interaction_data.append((
            hydrophobic.restype, hydrophobic.resnr, hydrophobic.reschain, 'hydrophobic'
        ))
    
    for pistack in interaction_set.pistacking:
        interaction_data.append((
            pistack.restype, pistack.resnr, pistack.reschain, 'pi_stacking'
        ))
    
    for saltbridge in interaction_set.saltbridge_lneg + interaction_set.saltbridge_pneg:
        interaction_data.append((
            saltbridge.restype, saltbridge.resnr, saltbridge.reschain, 'salt_bridge'
        ))
    
    for halogen in interaction_set.halogen_bonds:
        interaction_data.append((
            halogen.restype, halogen.resnr, halogen.reschain, 'halogen_bond'
        ))
    
    for picat in interaction_set.pication_laro + interaction_set.pication_paro:
        interaction_data.append((
            picat.restype, picat.resnr, picat.reschain, 'pi_cation'
        ))
    
    for wbridge in interaction_set.water_bridges:
        interaction_data.append((
            wbridge.restype, wbridge.resnr, wbridge.reschain, 'water_bridge'
        ))
    
    for metal in interaction_set.metal_complexes:
        interaction_data.append((
            metal.restype, metal.resnr, metal.reschain, 'metal_coordination'
        ))
    
    return interaction_data

def create_plip_interaction_summary(interaction_set, interaction_data: list) -> dict:
    """
    Create interaction summary from PLIP data.
    
    Args:
        interaction_set: PLIP interaction set object
        interaction_data: List of extracted interactions
        
    Returns:
        Dictionary containing interaction summary
    """
    return {
        'total_interactions': len(interaction_data),
        'interaction_types': {
            'hydrogen_bonds': len(interaction_set.hbonds_ldon + interaction_set.hbonds_pdon),
            'hydrophobic': len(interaction_set.hydrophobic_contacts),
            'pi_stacking': len(interaction_set.pistacking),
            'salt_bridges': len(interaction_set.saltbridge_lneg + interaction_set.saltbridge_pneg),
            'halogen_bonds': len(interaction_set.halogen_bonds),
            'pi_cation': len(interaction_set.pication_laro + interaction_set.pication_paro),
            'water_bridges': len(interaction_set.water_bridges),
            'metal_coordination': len(interaction_set.metal_complexes)
        },
        'interactions': [
            {
                'type': inter[3],
                'restype': inter[0],
                'resnr': inter[1],
                'reschain': inter[2]
            } for inter in interaction_data
        ]
    }

def interactions_to_fingerprint(interaction_data: list, fp_size: int = 1024) -> np.ndarray:
    """
    Convert interaction data to binary fingerprint.
    
    Args:
        interaction_data: List of interaction tuples
        fp_size: Size of the fingerprint vector
        
    Returns:
        Binary fingerprint as numpy array
    """
    ifp = np.zeros(fp_size, dtype=int)
    
    for interaction in interaction_data:
        restype, resnr, reschain, int_type = interaction
        interaction_str = f"{restype}_{resnr}_{reschain}_{int_type}"
        hash_val = hash(interaction_str) % fp_size
        ifp[hash_val] = 1
    
    return ifp

def calculate_plip_interactions(protein_path: str, ligand_mol: 'Chem.Mol', ligand_name: str = "LIG") -> tuple:
    """
    Calculate PLIP interactions for a protein-ligand complex.
    
    Args:
        protein_path: Path to protein PDB file
        ligand_mol: RDKit molecule object for the ligand
        ligand_name: Name for the ligand (default: "LIG")
        
    Returns:
        Tuple of (interaction_fingerprint, interaction_summary)
        
    Raises:
        ImportError: If PLIP is not available
        ValueError: If no ligand is found in the complex
    """
    if not PLIP_AVAILABLE:
        raise ImportError("PLIP is not available. Install with: pip install plip")
    
    # Create complex PDB using BioPython
    complex_path = create_complex_with_biopython(protein_path, ligand_mol, ligand_name)
    if complex_path is None:
        raise RuntimeError("Failed to create protein-ligand complex. BioPython complex creation failed.")
    
    try:
        # Initialize PLIP
        mol = PDBComplex()
        mol.load_pdb(complex_path)
        
        # Find the ligand
        ligand_id = None
        for lig in mol.ligands:
            if lig.hetid in ['LIG', 'UNL', 'UNK']:
                ligand_id = ':'.join([lig.hetid, lig.chain, str(lig.position)])
                break
        
        if not ligand_id and mol.ligands:
            lig = mol.ligands[-1]
            ligand_id = ':'.join([lig.hetid, lig.chain, str(lig.position)])
        
        if not ligand_id:
            logger.warning("No ligand found in the complex")
            return np.zeros(1024), {'total_interactions': 0, 'interaction_types': {}, 'interactions': []}
        
        logger.debug(f"Analyzing ligand: {ligand_id}")
        
        # Analyze the complex
        mol.analyze()
        
        # Get the interaction set for our ligand
        if ligand_id not in mol.interaction_sets:
            logger.warning(f"No interactions found for ligand {ligand_id}")
            return np.zeros(1024), {'total_interactions': 0, 'interaction_types': {}, 'interactions': []}
        
        interaction_set = mol.interaction_sets[ligand_id]
        
        # Extract interaction data
        interaction_data = extract_plip_interactions(interaction_set)
        
        # Convert to interaction fingerprint
        ifp = interactions_to_fingerprint(interaction_data)
        
        # Create interaction summary
        interactions_summary = create_plip_interaction_summary(interaction_set, interaction_data)
        
        logger.debug(f"Found {len(interaction_data)} interactions")
        
        return ifp, interactions_summary
        
    finally:
        # Clean up temporary file
        if os.path.exists(complex_path):
            os.unlink(complex_path)

def get_plip_interaction_types() -> list:
    """
    Get list of interaction types supported by PLIP.
    
    Returns:
        List of supported interaction type names
    """
    return [
        'hydrogen_bond', 'hydrophobic', 'pi_stacking', 'salt_bridge',
        'halogen_bond', 'pi_cation', 'water_bridge', 'metal_coordination'
    ]

# ProLIF functions
def create_ligand_pdb(ligand_mol: 'Chem.Mol') -> str:
    """
    Create temporary PDB file for ligand.
    
    Args:
        ligand_mol: RDKit molecule object
        
    Returns:
        Path to temporary PDB file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
        pdb_block = Chem.MolToPDBBlock(ligand_mol)
        tmp_pdb.write(pdb_block)
        return tmp_pdb.name

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
    
    # Create temporary PDB file for ligand
    tmp_pdb_path = create_ligand_pdb(ligand_mol)
    
    try:
        # Load structures with MDAnalysis
        protein_u = mda.Universe(protein_path)
        ligand_u = mda.Universe(tmp_pdb_path)
        
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
        if os.path.exists(tmp_pdb_path):
            os.unlink(tmp_pdb_path)

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

# Main interaction computation functions
def compute_interaction_fingerprint(
    mol: Chem.Mol, 
    protein_content: str, 
    interaction_config: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], int]:
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
            # Calculate interactions using specified method
            ifp_type = interaction_config.get('interaction_type', 'plip')
            ligand_name = interaction_config.get('ligand_name', 'LIG')
            
            if ifp_type.lower() == 'plip' and PLIP_AVAILABLE:
                interaction_fp, interaction_summary = calculate_plip_interactions(
                    protein_path, mol, ligand_name
                )
            elif ifp_type.lower() == 'prolif' and PROLIF_AVAILABLE:
                interaction_fp, interaction_summary = calculate_prolif_interactions(
                    protein_path, mol, ligand_name
                )
            else:
                logger.warning(f"Interaction type '{ifp_type}' not available")
                return None, None, 0
            
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

def _compute_interactions_worker(args):
    """Worker function for parallel interaction computation."""
    idx, mol, protein_content, config = args
    
    if mol is None:
        return idx, None, None, 0
    
    # Compute interaction fingerprint
    interaction_fp, interactions_json, num_interactions = compute_interaction_fingerprint(
        mol, protein_content, config
    )
    
    return idx, interaction_fp, interactions_json, num_interactions

def compute_all_interactions(
    df: pd.DataFrame,
    protein_content: str,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute all protein-ligand interactions for DataFrame with parallelization.
    
    Chain-of-Thought:
    - Single entry point for all interaction computation
    - Process molecules in parallel for efficiency
    - Returns updated DataFrame (functional style)
    """
    df = df.copy()

    # Prepare data for parallel processing
    valid_molecules = []
    for idx, row in df.iterrows():
        if row['mol'] is not None:
            valid_molecules.append((idx, row['mol'], protein_content, config))

    if not valid_molecules:
        return df

    # Determine number of workers
    max_workers = config.get('max_workers', min(mp.cpu_count(), len(valid_molecules)))

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_interactions_worker, mol_data): mol_data[0] 
            for mol_data in valid_molecules
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, interaction_fp, interactions_json, num_interactions = future.result()
                df.at[idx, 'interaction_fp'] = interaction_fp
                df.at[idx, 'interactions'] = interactions_json
                df.at[idx, 'num_interactions'] = num_interactions
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error computing interactions for molecule {idx}: {e}")
                df.at[idx, 'interaction_fp'] = None
                df.at[idx, 'interactions'] = None
                df.at[idx, 'num_interactions'] = 0

    logger.info(f"Computed interactions for {len(df)} molecules using {max_workers} workers")
    return df

def create_default_interaction_config() -> Dict[str, Any]:
    """
    Create default configuration for interaction calculation.
    
    Returns:
        Default interaction configuration
    """
    return {
        "interaction_type": "plip",  # or "prolif"
        "ligand_name": "LIG",
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