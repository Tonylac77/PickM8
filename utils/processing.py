"""
Processing utilities for molecular data pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from core.fingerprints import (
    compute_morgan_fingerprint, compute_rdkit_fingerprint, compute_mapchiral_fingerprint,
    compute_interaction_fingerprint
)
from core.pose_analysis import compute_pose_quality_batch

logger = logging.getLogger(__name__)


def process_molecule_fingerprints(args):
    """Process a single molecule for fingerprints."""
    mol_id, mol, protein_content, fp_config, interaction_config = args
    
    result = {
        'id': mol_id,
        'morgan_fp': None,
        'rdkit_fp': None,
        'mapchiral_fp': None,
        'interaction_fp': None,
        'interactions': None,
        'num_interactions': 0
    }
    
    try:
        # Compute molecular fingerprints
        if fp_config.get('compute_morgan', True):
            result['morgan_fp'] = compute_morgan_fingerprint(
                mol, 
                radius=fp_config.get('morgan_radius', 2),
                n_bits=fp_config.get('morgan_bits', 2048)
            )
            
        if fp_config.get('compute_rdkit', True):
            result['rdkit_fp'] = compute_rdkit_fingerprint(
                mol,
                n_bits=fp_config.get('rdkit_bits', 2048)
            )
            
        if fp_config.get('compute_mapchiral', True):
            result['mapchiral_fp'] = compute_mapchiral_fingerprint(
                mol,
                max_radius=fp_config.get('mapchiral_max_radius', 2),
                n_permutations=fp_config.get('mapchiral_n_permutations', 2048),
                mapping=fp_config.get('mapchiral_mapping', False)
            )
            
        # Compute interaction fingerprints
        if fp_config.get('compute_interactions', True) and protein_content:
            ifp, interactions, num_int = compute_interaction_fingerprint(
                mol, protein_content, interaction_config
            )
            result['interaction_fp'] = ifp
            result['interactions'] = interactions
            result['num_interactions'] = num_int
            
    except Exception as e:
        logger.error(f"Error processing molecule {mol_id}: {e}")
        
    return result


def compute_fingerprints_batch(df: pd.DataFrame, protein_content: str, 
                             fp_config: Dict[str, Any], interaction_config: Dict[str, Any],
                             n_workers: Optional[int] = None) -> pd.DataFrame:
    """
    Compute fingerprints for all molecules in DataFrame using parallel processing.
    
    Args:
        df: DataFrame with molecules
        protein_content: PDB content as string
        fp_config: Fingerprint computation configuration
        interaction_config: Interaction calculation configuration
        n_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        Updated DataFrame with computed fingerprints
    """
    df = df.copy()
    
    if len(df) == 0:
        return df
        
    # Set number of workers
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(df))
        
    logger.info(f"Computing fingerprints for {len(df)} molecules using {n_workers} workers")
    
    # Prepare arguments for parallel processing
    args_list = [
        (row['id'], row['mol'], protein_content, fp_config, interaction_config)
        for _, row in df.iterrows()
        if row['mol'] is not None
    ]
    
    if not args_list:
        logger.warning("No valid molecules found for fingerprint computation")
        return df
    
    # Process in parallel
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_id = {
                executor.submit(process_molecule_fingerprints, args): args[0] 
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
        logger.error(f"Error in parallel fingerprint computation: {e}")
        # Fallback to sequential processing
        logger.info("Falling back to sequential processing")
        for args in args_list:
            result = process_molecule_fingerprints(args)
            results[result['id']] = result
    
    # Update DataFrame with results
    for mol_id, result in results.items():
        mask = df['id'] == mol_id
        if mask.any():
            # Get the row index for proper assignment
            row_idx = df.index[mask].tolist()[0]
            df.at[row_idx, 'morgan_fp'] = result['morgan_fp']
            df.at[row_idx, 'rdkit_fp'] = result['rdkit_fp']
            df.at[row_idx, 'mapchiral_fp'] = result['mapchiral_fp']
            df.at[row_idx, 'interaction_fp'] = result['interaction_fp']
            df.at[row_idx, 'interactions'] = result['interactions']
            df.at[row_idx, 'num_interactions'] = result['num_interactions']
    
    successful = len([r for r in results.values() if r['morgan_fp'] is not None])
    logger.info(f"Successfully computed fingerprints for {successful}/{len(df)} molecules")
    
    return df


def get_fingerprint_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about computed fingerprints.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary with fingerprint statistics
    """
    total_molecules = len(df)
    
    if total_molecules == 0:
        return {"total_molecules": 0}
    
    stats = {
        "total_molecules": total_molecules,
        "morgan_fp_computed": df['morgan_fp'].notna().sum(),
        "rdkit_fp_computed": df['rdkit_fp'].notna().sum(),
        "mapchiral_fp_computed": df['mapchiral_fp'].notna().sum() if 'mapchiral_fp' in df.columns else 0,
        "interaction_fp_computed": df['interaction_fp'].notna().sum(),
        "molecules_with_interactions": (df['num_interactions'] > 0).sum(),
        "avg_interactions_per_molecule": df['num_interactions'].mean(),
        "max_interactions": df['num_interactions'].max(),
        "min_interactions": df['num_interactions'].min()
    }
    
    # Calculate completion percentages
    stats["morgan_fp_percentage"] = (stats["morgan_fp_computed"] / total_molecules) * 100
    stats["rdkit_fp_percentage"] = (stats["rdkit_fp_computed"] / total_molecules) * 100
    stats["mapchiral_fp_percentage"] = (stats["mapchiral_fp_computed"] / total_molecules) * 100
    stats["interaction_fp_percentage"] = (stats["interaction_fp_computed"] / total_molecules) * 100
    
    return stats