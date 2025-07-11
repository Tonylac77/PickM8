"""Fingerprint computation functions using scikit-fingerprints."""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

#disable rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

# Try to import scikit-fingerprints
try:
    from skfp.fingerprints import (
        E3FPFingerprint,
        ECFPFingerprint,
        ElectroShapeFingerprint,
        FunctionalGroupsFingerprint,
        MACCSFingerprint,
        RDKitFingerprint,
        PatternFingerprint,
        PharmacophoreFingerprint
    )
    SKFP_AVAILABLE = True
except ImportError:
    SKFP_AVAILABLE = False
    logger.warning("scikit-fingerprints not available")

# Try to import MapChiral (legacy support)
try:
    from mapchiral.mapchiral import encode as mapchiral_encode
    MAPCHIRAL_AVAILABLE = True
except ImportError:
    MAPCHIRAL_AVAILABLE = False
    logger.warning("MapChiral not available")

def is_mapchiral_available() -> bool:
    """Check if MapChiral is available (legacy function for tests)."""
    return MAPCHIRAL_AVAILABLE

def is_skfp_available() -> bool:
    """Check if scikit-fingerprints is available."""
    return SKFP_AVAILABLE


def compute_mapchiral_fingerprint(
    mol: Chem.Mol,
    max_radius: int = 2,
    n_permutations: int = 2048
) -> Optional[List[float]]:
    """Compute MapChiral fingerprint if available (legacy support)."""
    try:
        if mol is None or not MAPCHIRAL_AVAILABLE:
            return None

        fp = mapchiral_encode(mol, max_radius=max_radius, n_permutations=n_permutations)
        return fp.tolist() if fp is not None else None
    except Exception as e:
        logger.error(f"Error computing MapChiral fingerprint: {e}")
        return None

def compute_e3fp_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute E3FP fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = E3FPFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing E3FP fingerprint: {e}")
        return None

def compute_ecfp_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute ECFP fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = ECFPFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing ECFP fingerprint: {e}")
        return None

def compute_electroshape_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[float]]:
    """Compute ElectroShape fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = ElectroShapeFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing ElectroShape fingerprint: {e}")
        return None

def compute_functional_groups_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute FunctionalGroups fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = FunctionalGroupsFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing FunctionalGroups fingerprint: {e}")
        return None

def compute_maccs_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute MACCS fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = MACCSFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing MACCS fingerprint: {e}")
        return None

def compute_pattern_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute Pattern fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = PatternFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing Pattern fingerprint: {e}")
        return None

def compute_pharmacophore_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute Pharmacophore fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = PharmacophoreFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing Pharmacophore fingerprint: {e}")
        return None

def compute_skfp_rdkit_fingerprint(mol: Chem.Mol, **kwargs) -> Optional[List[int]]:
    """Compute RDKit fingerprint using scikit-fingerprints."""
    try:
        if mol is None or not SKFP_AVAILABLE:
            return None
        
        fp = RDKitFingerprint(**kwargs)
        result = fp.transform([mol])
        return result[0].tolist() if result is not None and len(result) > 0 else None
    except Exception as e:
        logger.error(f"Error computing RDKit fingerprint (scikit-fingerprints): {e}")
        return None

def _compute_fingerprints_worker(args):
    """Worker function for parallel fingerprint computation."""
    idx, mol, config = args
    
    if mol is None:
        return idx, {fp_type: None for fp_type in get_available_fingerprint_types()}
    
    results = {}
    
    # Legacy MapChiral fingerprint (kept for backward compatibility)
    if config.get('compute_mapchiral', False):
        results['mapchiral_fp'] = compute_mapchiral_fingerprint(
            mol,
            max_radius=config.get('mapchiral_max_radius', 2),
            n_permutations=config.get('mapchiral_n_permutations', 2048)
        )
    
    # scikit-fingerprints
    if config.get('compute_e3fp', False):
        results['e3fp_fp'] = compute_e3fp_fingerprint(
            mol,
            **config.get('e3fp_params', {})
        )
    
    if config.get('compute_ecfp', False):
        results['ecfp_fp'] = compute_ecfp_fingerprint(
            mol,
            **config.get('ecfp_params', {})
        )
    
    if config.get('compute_electroshape', False):
        results['electroshape_fp'] = compute_electroshape_fingerprint(
            mol,
            **config.get('electroshape_params', {})
        )
    
    if config.get('compute_functional_groups', False):
        results['functional_groups_fp'] = compute_functional_groups_fingerprint(
            mol,
            **config.get('functional_groups_params', {})
        )
    
    if config.get('compute_maccs', False):
        results['maccs_fp'] = compute_maccs_fingerprint(
            mol,
            **config.get('maccs_params', {})
        )
    
    if config.get('compute_pattern', False):
        results['pattern_fp'] = compute_pattern_fingerprint(
            mol,
            **config.get('pattern_params', {})
        )
    
    if config.get('compute_pharmacophore', False):
        results['pharmacophore_fp'] = compute_pharmacophore_fingerprint(
            mol,
            **config.get('pharmacophore_params', {})
        )
    
    return idx, results

def get_available_fingerprint_types() -> List[str]:
    """Get list of available fingerprint types."""
    return [
        'mapchiral_fp',  # Legacy (kept for backward compatibility)
        'e3fp_fp', 'ecfp_fp', 'electroshape_fp', 'functional_groups_fp',
        'maccs_fp', 'pattern_fp', 'pharmacophore_fp'  # scikit-fingerprints
    ]

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
    - Supports both legacy and scikit-fingerprints
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
                idx, fp_results = future.result()
                # Update DataFrame with all fingerprint results
                for fp_type, fp_value in fp_results.items():
                    if fp_type in df.columns:
                        df.at[idx, fp_type] = fp_value
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error computing fingerprints for molecule {idx}: {e}")
                # Set all fingerprint columns to None on error
                for fp_type in get_available_fingerprint_types():
                    if fp_type in df.columns:
                        df.at[idx, fp_type] = None

    logger.info(f"Computed fingerprints for {len(df)} molecules using {max_workers} workers")
    return df

def get_fingerprint_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate fingerprint computation statistics."""
    total = len(df)
    if total == 0:
        return {"total_molecules": 0}

    stats = {
        "total_molecules": total,
        "mapchiral_available": MAPCHIRAL_AVAILABLE,
        "skfp_available": SKFP_AVAILABLE
    }

    # Calculate statistics for all fingerprint types
    for fp_type in get_available_fingerprint_types():
        if fp_type in df.columns:
            computed = df[fp_type].notna().sum()
            stats[f"{fp_type}_computed"] = computed
            stats[f"{fp_type}_percentage"] = (computed / total * 100) if total > 0 else 0

    return stats