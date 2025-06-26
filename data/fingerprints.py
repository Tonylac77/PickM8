"""Fingerprint computation functions."""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Try to import MapChiral
try:
    from mapchiral.mapchiral import encode as mapchiral_encode
    MAPCHIRAL_AVAILABLE = True
except ImportError:
    MAPCHIRAL_AVAILABLE = False
    logger.warning("MapChiral not available")

def is_mapchiral_available() -> bool:
    """Check if MapChiral is available (legacy function for tests)."""
    return MAPCHIRAL_AVAILABLE

def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Optional[List[int]]:
    """Compute Morgan fingerprint for a molecule."""
    try:
        if mol is None:
            return None

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp.ToBitString())
    except Exception as e:
        logger.error(f"Error computing Morgan fingerprint: {e}")
        return None

def compute_rdkit_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> Optional[List[int]]:
    """Compute RDKit fingerprint for a molecule."""
    try:
        if mol is None:
            return None

        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        return [int(bit) for bit in fp.ToBitString()]
    except Exception as e:
        logger.error(f"Error computing RDKit fingerprint: {e}")
        return None

def compute_mapchiral_fingerprint(
    mol: Chem.Mol,
    max_radius: int = 2,
    n_permutations: int = 2048
) -> Optional[List[float]]:
    """Compute MapChiral fingerprint if available."""
    try:
        if mol is None or not MAPCHIRAL_AVAILABLE:
            return None

        fp = mapchiral_encode(mol, max_radius=max_radius, n_permutations=n_permutations)
        return fp.tolist() if fp is not None else None
    except Exception as e:
        logger.error(f"Error computing MapChiral fingerprint: {e}")
        return None

def _compute_fingerprints_worker(args):
    """Worker function for parallel fingerprint computation."""
    idx, mol, config = args
    
    if mol is None:
        return idx, None, None, None
    
    # Prepare computation arguments
    compute_morgan = config.get('compute_morgan', True)
    compute_rdkit = config.get('compute_rdkit', True)
    compute_mapchiral = config.get('compute_mapchiral', True)
    
    morgan_fp = None
    rdkit_fp = None
    mapchiral_fp = None
    
    if compute_morgan:
        morgan_fp = compute_morgan_fingerprint(
            mol,
            radius=config.get('morgan_radius', 2),
            n_bits=config.get('morgan_bits', 2048)
        )
    
    if compute_rdkit:
        rdkit_fp = compute_rdkit_fingerprint(
            mol,
            n_bits=config.get('rdkit_bits', 2048)
        )
    
    if compute_mapchiral:
        mapchiral_fp = compute_mapchiral_fingerprint(
            mol,
            max_radius=config.get('mapchiral_max_radius', 2),
            n_permutations=config.get('mapchiral_n_permutations', 2048)
        )
    
    return idx, morgan_fp, rdkit_fp, mapchiral_fp

def compute_all_fingerprints(
    df: pd.DataFrame,
    protein_content: str,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute all molecular fingerprints for DataFrame with parallelization.

    Chain-of-Thought:
    - Single entry point for all fingerprint computation
    - Parallel processing for efficiency
    - Returns updated DataFrame (functional style)
    """
    df = df.copy()

    # Prepare data for parallel processing
    valid_molecules = []
    for idx, row in df.iterrows():
        if row['mol'] is not None:
            valid_molecules.append((idx, row['mol'], config))

    if not valid_molecules:
        return df

    # Determine number of workers
    max_workers = config.get('max_workers', min(mp.cpu_count(), len(valid_molecules)))

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_fingerprints_worker, mol_data): mol_data[0] 
            for mol_data in valid_molecules
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, morgan_fp, rdkit_fp, mapchiral_fp = future.result()
                df.at[idx, 'morgan_fp'] = morgan_fp
                df.at[idx, 'rdkit_fp'] = rdkit_fp
                df.at[idx, 'mapchiral_fp'] = mapchiral_fp
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error computing fingerprints for molecule {idx}: {e}")
                df.at[idx, 'morgan_fp'] = None
                df.at[idx, 'rdkit_fp'] = None
                df.at[idx, 'mapchiral_fp'] = None

    logger.info(f"Computed fingerprints for {len(df)} molecules using {max_workers} workers")
    return df

def get_fingerprint_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate fingerprint computation statistics."""
    total = len(df)
    if total == 0:
        return {"total_molecules": 0}

    stats = {
        "total_molecules": total,
        "morgan_fp_computed": df['morgan_fp'].notna().sum(),
        "rdkit_fp_computed": df['rdkit_fp'].notna().sum(),
        "mapchiral_fp_computed": df['mapchiral_fp'].notna().sum() if 'mapchiral_fp' in df.columns else 0,
        "mapchiral_available": MAPCHIRAL_AVAILABLE
    }

    # Calculate percentages
    for fp_type in ['morgan_fp', 'rdkit_fp', 'mapchiral_fp']:
        if fp_type in stats:
            computed = stats[f"{fp_type}_computed"]
            stats[f"{fp_type}_percentage"] = (computed / total * 100) if total > 0 else 0

    return stats