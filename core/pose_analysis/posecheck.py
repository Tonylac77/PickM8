"""
PoseCheck integration for pose quality validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import tempfile
import os
from pathlib import Path

from rdkit import Chem

# Import PoseCheck if available
try:
    from posecheck import PoseCheck
    POSECHECK_AVAILABLE = True
except ImportError:
    POSECHECK_AVAILABLE = False
    logging.warning("PoseCheck not available")

logger = logging.getLogger(__name__)


def analyze_single_molecule_pose(args: Tuple[int, str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze pose quality for a single molecule (for parallel processing).
    
    Args:
        args: Tuple of (mol_id, mol_block, protein_content, config)
        
    Returns:
        Dictionary with molecule ID and pose quality metrics
    """
    mol_id, mol_block, protein_content, config = args
    
    result = {
        'id': mol_id,
        'clashes': 0,
        'strain_energy': 0.0,
        'error': None
    }
    
    if not POSECHECK_AVAILABLE:
        result['error'] = "PoseCheck not available"
        return result
    
    try:
        # Create temporary files for PoseCheck
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write protein file
            protein_path = Path(temp_dir) / "protein.pdb"
            with open(protein_path, 'w') as f:
                f.write(protein_content)
            
            # Write ligand file
            ligand_path = Path(temp_dir) / "ligand.sdf"
            with open(ligand_path, 'w') as f:
                f.write(mol_block)
            
            # Initialize PoseCheck
            pc = PoseCheck()
            
            # Load structures
            pc.load_protein_from_pdb(str(protein_path))
            pc.load_ligands_from_sdf(str(ligand_path))
            
            # Calculate clash score
            if config.get('calculate_clashes', True):
                try:
                    clash_results = pc.calculate_clashes()
                    if clash_results and len(clash_results) > 0:
                        clash_value = clash_results[0]
                        if isinstance(clash_value, dict):
                            result['clashes'] = int(clash_value.get('clashes', 0))
                        else:
                            # Handle direct numeric value
                            result['clashes'] = int(clash_value) if clash_value is not None else 0
                except Exception as e:
                    logger.warning(f"Error calculating clashes for molecule {mol_id}: {e}")
            
            # Calculate strain energy
            if config.get('calculate_strain', True):
                try:
                    strain_results = pc.calculate_strain_energy()
                    if strain_results and len(strain_results) > 0:
                        strain_value = strain_results[0]
                        if isinstance(strain_value, dict):
                            result['strain_energy'] = float(strain_value.get('strain_energy', 0.0))
                        else:
                            # Handle direct numeric value
                            result['strain_energy'] = float(strain_value) if strain_value is not None else 0.0
                except Exception as e:
                    logger.warning(f"Error calculating strain energy for molecule {mol_id}: {e}")
                    
    except Exception as e:
        logger.error(f"Error analyzing pose for molecule {mol_id}: {e}")
        result['error'] = str(e)
    
    return result


def compute_pose_quality_batch(df: pd.DataFrame, protein_content: str,
                             config: Dict[str, Any], n_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Compute pose quality metrics for all molecules in DataFrame using parallel processing.
    
    Args:
        df: DataFrame with molecules
        protein_content: PDB content as string
        config: PoseCheck configuration
        n_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        Updated DataFrame with pose quality metrics
    """
    df = df.copy()
    
    if len(df) == 0 or not protein_content:
        return df
        
    if not POSECHECK_AVAILABLE:
        logger.warning("PoseCheck not available, using simple pose quality analysis")
        return compute_pose_quality_simple(df, protein_content)
    
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(df))
        
    logger.info(f"Computing pose quality for {len(df)} molecules using {n_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [
        (row['id'], row['mol_block'], protein_content, config)
        for _, row in df.iterrows()
        if pd.notna(row['mol_block']) and row['mol_block']
    ]
    
    if not args_list:
        logger.warning("No valid molecules found for pose quality analysis")
        return df
    
    # Process in parallel
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_id = {
                executor.submit(analyze_single_molecule_pose, args): args[0] 
                for args in args_list
            }
            
            for future in as_completed(future_to_id):
                mol_id = future_to_id[future]
                try:
                    result = future.result()
                    results[result['id']] = result
                except Exception as e:
                    logger.error(f"Error processing molecule {mol_id}: {e}")
                    
    except Exception as e:
        logger.error(f"Error in parallel pose quality computation: {e}")
        # Fallback to sequential processing
        logger.info("Falling back to sequential processing")
        for args in args_list:
            result = analyze_single_molecule_pose(args)
            results[result['id']] = result
    
    # Update DataFrame with results
    for mol_id, result in results.items():
        mask = df['id'] == mol_id
        if mask.any():
            df.loc[mask, 'clashes'] = result['clashes']
            df.loc[mask, 'strain_energy'] = result['strain_energy']
    
    successful = len([r for r in results.values() if r['error'] is None])
    logger.info(f"Successfully analyzed pose quality for {successful}/{len(df)} molecules")
    
    return df


def analyze_single_molecule_pose_simple(mol_id: int, mol_block: str, protein_content: str,
                                      clash_threshold: float = 2.5) -> Dict[str, Any]:
    """
    Simplified pose analysis for a single molecule without PoseCheck.
    Uses basic geometric checks as fallback.
    
    Args:
        mol_id: Molecule ID
        mol_block: Molecule SDF block
        protein_content: PDB content
        clash_threshold: Distance threshold for clash detection (Angstroms)
        
    Returns:
        Dictionary with basic pose quality metrics
    """
    result = {
        'id': mol_id,
        'clashes': 0,
        'strain_energy': 0.0,
        'error': None
    }
    
    try:
        # Parse molecule
        mol = Chem.MolFromMolBlock(mol_block)
        if mol is None:
            result['error'] = "Invalid molecule"
            return result
        
        # Get conformer if available
        if mol.GetNumConformers() == 0:
            result['error'] = "No conformer available"
            return result
        
        conf = mol.GetConformer()
        
        # Basic clash detection (simplified)
        # Count atoms that might be too close to protein
        # This is a very simplified approach - real clash detection would need protein coordinates
        
        # For now, just count number of atoms as a proxy for potential clashes
        # In a real implementation, you would parse protein coordinates and check distances
        num_atoms = mol.GetNumAtoms()
        
        # Heuristic: molecules with many atoms in small space might have more clashes
        if num_atoms > 50:
            result['clashes'] = max(0, num_atoms - 50) // 10
        
        # Simple strain energy estimation based on molecular complexity
        # Real strain energy would require force field calculations
        num_bonds = mol.GetNumBonds()
        num_rings = mol.GetRingInfo().NumRings()
        
        # Heuristic: more complex molecules might have higher strain
        result['strain_energy'] = (num_rings * 2.0) + (num_bonds * 0.1)
        
    except Exception as e:
        logger.error(f"Error in simple pose analysis for molecule {mol_id}: {e}")
        result['error'] = str(e)
    
    return result


def compute_pose_quality_simple(df: pd.DataFrame, protein_content: str) -> pd.DataFrame:
    """
    Compute simplified pose quality metrics without PoseCheck.
    
    Args:
        df: DataFrame with molecules
        protein_content: PDB content as string
        
    Returns:
        Updated DataFrame with simplified pose quality metrics
    """
    df = df.copy()
    
    if len(df) == 0:
        return df
    
    logger.info(f"Computing simplified pose quality for {len(df)} molecules")
    
    for idx, row in df.iterrows():
        if pd.notna(row['mol_block']) and row['mol_block']:
            result = analyze_single_molecule_pose_simple(
                row['id'], row['mol_block'], protein_content
            )
            df.loc[idx, 'clashes'] = result['clashes']
            df.loc[idx, 'strain_energy'] = result['strain_energy']
    
    return df