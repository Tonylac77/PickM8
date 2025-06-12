#!/usr/bin/env python3
"""
PLIP-based protein-ligand interaction fingerprint calculations.
Pure functional implementation for PLIP interaction analysis.
"""

import numpy as np
import tempfile
import os
import logging
from pathlib import Path
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO
from io import StringIO
import string

logger = logging.getLogger(__name__)

# PLIP imports
try:
    from plip.structure.preparation import PDBComplex
    PLIP_AVAILABLE = True
except ImportError:
    PLIP_AVAILABLE = False
    logger.warning("PLIP not available. Install with: pip install plip")


def is_plip_available():
    """Check if PLIP is available for use."""
    return PLIP_AVAILABLE


def create_complex_with_biopython(protein_path: str, ligand_mol: 'Chem.Mol', ligand_name: str) -> str | None:
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


def create_complex_manually(protein_path: str, ligand_mol: 'Chem.Mol', ligand_name: str) -> str:
    """
    Fallback method to create complex without BioPython dependencies.
    
    Args:
        protein_path: Path to protein PDB file
        ligand_mol: RDKit molecule object
        ligand_name: Name for the ligand
        
    Returns:
        Path to the created complex PDB file
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
        with open(protein_path, 'r') as f:
            protein_lines = f.readlines()
        
        for line in protein_lines:
            if not line.startswith('END'):
                tmp_pdb.write(line)
        
        if not any(line.startswith('TER') for line in protein_lines[-5:]):
            tmp_pdb.write('TER\n')
        
        ligand_pdb = Chem.MolToPDBBlock(ligand_mol)
        ligand_lines = ligand_pdb.strip().split('\n')
        
        for line in ligand_lines:
            if line.startswith(('ATOM', 'HETATM')):
                if line.startswith('ATOM'):
                    line = 'HETATM' + line[6:]
                
                resname = line[17:20].strip()
                if not resname or resname == 'UNL':
                    line = line[:17] + 'LIG' + line[20:]
                
                if line[21] == ' ':
                    line = line[:21] + 'Z' + line[22:]
                
                tmp_pdb.write(line + '\n')
            elif line.startswith('CONECT'):
                tmp_pdb.write(line + '\n')
        
        tmp_pdb.write('END\n')
        return tmp_pdb.name


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


def create_interaction_summary(interaction_set, interaction_data: list) -> dict:
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
    
    # Create complex PDB
    complex_path = create_complex_with_biopython(protein_path, ligand_mol, ligand_name)
    if complex_path is None:
        complex_path = create_complex_manually(protein_path, ligand_mol, ligand_name)
    
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
        interactions_summary = create_interaction_summary(interaction_set, interaction_data)
        
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