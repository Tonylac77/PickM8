"""PoseCheck integration and pose quality analysis."""
import pandas as pd
import numpy as np
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Import PoseCheck if available
try:
    from posecheck import PoseCheck
    POSECHECK_AVAILABLE = True
except ImportError:
    POSECHECK_AVAILABLE = False
    logger.warning("PoseCheck not available")

def analyze_single_pose(mol_block: str, protein_content: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze pose quality for a single molecule.
    
    Args:
        mol_block: Molecule SDF block
        protein_content: PDB content as string
        config: PoseCheck configuration
        
    Returns:
        Dictionary with pose quality metrics
    """
    result = {
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
                            result['clashes'] = int(clash_value)
                except Exception as e:
                    logger.warning(f"Error calculating clashes: {e}")
            
            # Calculate strain energy
            if config.get('calculate_strain', True):
                try:
                    strain_results = pc.calculate_strain_energy()
                    if strain_results and len(strain_results) > 0:
                        strain_value = strain_results[0]
                        if isinstance(strain_value, dict):
                            result['strain_energy'] = float(strain_value.get('strain_energy', 0.0))
                        else:
                            result['strain_energy'] = float(strain_value)
                except Exception as e:
                    logger.warning(f"Error calculating strain energy: {e}")
                    
    except Exception as e:
        logger.error(f"Error in pose quality analysis: {e}")
        result['error'] = str(e)
    
    return result

def _analyze_pose_worker(args):
    """Worker function for parallel pose analysis."""
    idx, mol_block, protein_content, config = args
    
    if mol_block is None:
        return idx, 0, 0.0
    
    pose_result = analyze_single_pose(mol_block, protein_content, config)
    return idx, pose_result['clashes'], pose_result['strain_energy']

def analyze_all_poses(
    df: pd.DataFrame,
    protein_content: str,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Analyze pose quality for all poses in DataFrame with parallelization.
    
    Chain-of-Thought:
    - Single entry point for all pose quality analysis
    - Process molecules in parallel for efficiency
    - Returns updated DataFrame (functional style)
    """
    df = df.copy()
    
    if not config.get('enabled', True):
        return df
    
    # Prepare data for parallel processing
    valid_molecules = []
    for idx, row in df.iterrows():
        if row['mol_block'] is not None:
            valid_molecules.append((idx, row['mol_block'], protein_content, config))
    
    if not valid_molecules:
        return df
    
    # Determine number of workers
    max_workers = config.get('max_workers', min(mp.cpu_count(), len(valid_molecules)))
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_analyze_pose_worker, mol_data): mol_data[0] 
            for mol_data in valid_molecules
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, clashes, strain_energy = future.result()
                df.at[idx, 'clashes'] = clashes
                df.at[idx, 'strain_energy'] = strain_energy
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error analyzing pose for molecule {idx}: {e}")
                df.at[idx, 'clashes'] = 0
                df.at[idx, 'strain_energy'] = 0.0
    
    logger.info(f"Analyzed pose quality for {len(df)} molecules using {max_workers} workers")
    return df

def get_pose_quality_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about pose quality metrics.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary with pose quality statistics
    """
    if len(df) == 0:
        return {"total_molecules": 0}
    
    clash_data = df['clashes'].dropna()
    strain_data = df['strain_energy'].dropna()
    
    stats = {
        "total_molecules": len(df),
        "molecules_with_clash_data": len(clash_data),
        "molecules_with_strain_data": len(strain_data),
        "posecheck_available": POSECHECK_AVAILABLE
    }
    
    if len(clash_data) > 0:
        stats.update({
            "avg_clashes": float(clash_data.mean()),
            "max_clashes": int(clash_data.max()),
            "min_clashes": int(clash_data.min()),
            "molecules_with_clashes": int((clash_data > 0).sum()),
            "clash_free_molecules": int((clash_data == 0).sum())
        })
    
    if len(strain_data) > 0:
        stats.update({
            "avg_strain_energy": float(strain_data.mean()),
            "max_strain_energy": float(strain_data.max()),
            "min_strain_energy": float(strain_data.min()),
            "high_strain_molecules": int((strain_data > 10.0).sum())
        })
    
    return stats