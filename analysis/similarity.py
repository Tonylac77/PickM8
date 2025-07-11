"""
Molecular similarity analysis module for PickM8.

This module provides functions to calculate molecular similarity using 
fingerprint-based Tanimoto similarity calculations.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple, Dict, Any
from rdkit import DataStructs

logger = logging.getLogger(__name__)


def get_enabled_fingerprint_columns(config: Dict[str, Any]) -> List[str]:
    """
    Get list of enabled fingerprint columns based on configuration.
    
    Args:
        config: Configuration dictionary containing fingerprint settings
        
    Returns:
        List of enabled fingerprint column names
    """
    enabled_columns = []
    
    # Map configuration keys to DataFrame column names
    fingerprint_mapping = {
        'compute_ecfp': 'ecfp_fp',
        'compute_e3fp': 'e3fp_fp',
        'compute_electroshape': 'electroshape_fp',
        'compute_functional_groups': 'functional_groups_fp',
        'compute_maccs': 'maccs_fp',
        'compute_pattern': 'pattern_fp',
        'compute_pharmacophore': 'pharmacophore_fp',
        'compute_mapchiral': 'mapchiral_fp'  # Legacy support
    }
    
    fingerprint_config = config.get('fingerprint_config', {})
    
    for config_key, column_name in fingerprint_mapping.items():
        if fingerprint_config.get(config_key, False):
            enabled_columns.append(column_name)
    
    return enabled_columns


def concatenate_fingerprints(row: pd.Series, fingerprint_columns: List[str]) -> Optional[np.ndarray]:
    """
    Concatenate multiple fingerprints into a single feature vector.
    
    Args:
        row: DataFrame row containing fingerprint data
        fingerprint_columns: List of fingerprint column names to concatenate
        
    Returns:
        Concatenated fingerprint as numpy array, or None if no valid fingerprints
    """
    fingerprints = []
    
    for col in fingerprint_columns:
        if col in row.index and row[col] is not None:
            fp = row[col]
            
            # Convert to numpy array if it's a list
            if isinstance(fp, list):
                fp = np.array(fp)
            elif isinstance(fp, np.ndarray):
                fp = fp.copy()
            else:
                continue  # Skip invalid fingerprint types
            
            # Ensure 1D array
            if fp.ndim > 1:
                fp = fp.flatten()
            
            fingerprints.append(fp)
    
    if not fingerprints:
        return None
    
    # Concatenate all fingerprints
    try:
        concatenated = np.concatenate(fingerprints)
        return concatenated
    except Exception as e:
        logger.error(f"Error concatenating fingerprints: {e}")
        return None


def calculate_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity between two fingerprints using RDKit.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tanimoto similarity score (0.0 to 1.0)
    """
    if fp1 is None or fp2 is None:
        return 0.0
    
    # Ensure both fingerprints have the same length
    if len(fp1) != len(fp2):
        logger.warning(f"Fingerprint length mismatch: {len(fp1)} vs {len(fp2)}")
        return 0.0
    
    try:
        # Convert numpy arrays to RDKit bit vectors for efficient Tanimoto calculation
        
        # Check if fingerprints are binary (most common case)
        is_binary1 = np.all(np.isin(fp1, [0, 1]))
        is_binary2 = np.all(np.isin(fp2, [0, 1]))
        
        if is_binary1 and is_binary2:
            # Create explicit bit vectors for binary fingerprints
            bv1 = DataStructs.CreateFromBitString(''.join(str(int(b)) for b in fp1))
            bv2 = DataStructs.CreateFromBitString(''.join(str(int(b)) for b in fp2))
            
            # Calculate Tanimoto similarity using RDKit
            return DataStructs.TanimotoSimilarity(bv1, bv2)
        else:
            # For continuous fingerprints, use bulk Tanimoto calculation
            return DataStructs.BulkTanimotoSimilarity(
                DataStructs.CreateFromFloatArray(fp1),
                [DataStructs.CreateFromFloatArray(fp2)]
            )[0]
        
    except Exception as e:
        logger.error(f"Error calculating Tanimoto similarity with RDKit: {e}")
        # Fallback to manual calculation for continuous fingerprints
        return _calculate_tanimoto_manual(fp1, fp2)


def _calculate_tanimoto_manual(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Manual Tanimoto calculation for continuous fingerprints (fallback).
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tanimoto similarity score (0.0 to 1.0)
    """
    # Check if fingerprints are binary (contain only 0s and 1s)
    is_binary1 = np.all(np.isin(fp1, [0, 1]))
    is_binary2 = np.all(np.isin(fp2, [0, 1]))
    
    if is_binary1 and is_binary2:
        # Use Jaccard index for binary fingerprints
        # Handle case where both fingerprints are all zeros
        if np.sum(fp1) == 0 and np.sum(fp2) == 0:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = np.sum(fp1 & fp2)
        union = np.sum(fp1 | fp2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    else:
        # Use continuous Tanimoto formula for continuous fingerprints
        # T(A,B) = A·B / (|A|² + |B|² - A·B)
        dot_product = np.dot(fp1, fp2)
        norm_a_squared = np.dot(fp1, fp1)
        norm_b_squared = np.dot(fp2, fp2)
        
        denominator = norm_a_squared + norm_b_squared - dot_product
        
        if denominator == 0:
            return 1.0 if np.allclose(fp1, fp2) else 0.0
        
        return dot_product / denominator


def find_most_similar_molecule(
    target_mol: pd.Series,
    df: pd.DataFrame,
    fingerprint_columns: List[str],
    exclude_self: bool = True
) -> Optional[Tuple[pd.Series, float]]:
    """
    Find the most similar molecule to the target molecule in the DataFrame.
    
    Args:
        target_mol: Target molecule (pandas Series)
        df: DataFrame containing all molecules
        fingerprint_columns: List of fingerprint columns to use for similarity
        exclude_self: Whether to exclude the target molecule itself
        
    Returns:
        Tuple of (most_similar_molecule, similarity_score) or None if no matches
    """
    if len(df) == 0:
        return None
    
    # Get target fingerprint
    target_fp = concatenate_fingerprints(target_mol, fingerprint_columns)
    if target_fp is None:
        logger.warning("Target molecule has no valid fingerprints")
        return None
    
    best_similarity = -1.0
    best_molecule = None
    target_id = target_mol.get('id', None)
    
    # Compare with all molecules in the DataFrame
    for idx, row in df.iterrows():
        # Skip self if requested
        if exclude_self and target_id is not None and row.get('id') == target_id:
            continue
        
        # Get candidate fingerprint
        candidate_fp = concatenate_fingerprints(row, fingerprint_columns)
        if candidate_fp is None:
            continue
        
        # Calculate similarity
        similarity = calculate_tanimoto_similarity(target_fp, candidate_fp)
        
        # Update best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_molecule = row
    
    if best_molecule is None:
        return None
    
    return best_molecule, best_similarity


def get_similarity_statistics(
    df: pd.DataFrame,
    fingerprint_columns: List[str],
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate similarity statistics for the dataset.
    
    Args:
        df: DataFrame containing molecules
        fingerprint_columns: List of fingerprint columns to use
        sample_size: Optional sample size for large datasets
        
    Returns:
        Dictionary containing similarity statistics
    """
    if len(df) < 2:
        return {"error": "Need at least 2 molecules for similarity analysis"}
    
    # Sample if dataset is large
    if sample_size and len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    similarities = []
    valid_pairs = 0
    
    # Calculate pairwise similarities
    for i in range(len(sample_df)):
        for j in range(i + 1, len(sample_df)):
            mol1 = sample_df.iloc[i]
            mol2 = sample_df.iloc[j]
            
            fp1 = concatenate_fingerprints(mol1, fingerprint_columns)
            fp2 = concatenate_fingerprints(mol2, fingerprint_columns)
            
            if fp1 is not None and fp2 is not None:
                similarity = calculate_tanimoto_similarity(fp1, fp2)
                similarities.append(similarity)
                valid_pairs += 1
    
    if not similarities:
        return {"error": "No valid fingerprint pairs found"}
    
    similarities = np.array(similarities)
    
    return {
        "total_molecules": len(df),
        "sampled_molecules": len(sample_df),
        "valid_pairs": valid_pairs,
        "mean_similarity": float(np.mean(similarities)),
        "median_similarity": float(np.median(similarities)),
        "std_similarity": float(np.std(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "fingerprint_columns": fingerprint_columns
    }


def find_similar_molecules(
    target_mol: pd.Series,
    df: pd.DataFrame,
    fingerprint_columns: List[str],
    n_similar: int = 5,
    min_similarity: float = 0.1,
    exclude_self: bool = True
) -> List[Tuple[pd.Series, float]]:
    """
    Find N most similar molecules to the target molecule.
    
    Args:
        target_mol: Target molecule (pandas Series)
        df: DataFrame containing all molecules
        fingerprint_columns: List of fingerprint columns to use
        n_similar: Number of similar molecules to return
        min_similarity: Minimum similarity threshold
        exclude_self: Whether to exclude the target molecule itself
        
    Returns:
        List of (molecule, similarity_score) tuples, sorted by similarity (descending)
    """
    if len(df) == 0:
        return []
    
    # Get target fingerprint
    target_fp = concatenate_fingerprints(target_mol, fingerprint_columns)
    if target_fp is None:
        logger.warning("Target molecule has no valid fingerprints")
        return []
    
    similarities = []
    target_id = target_mol.get('id', None)
    
    # Compare with all molecules in the DataFrame
    for idx, row in df.iterrows():
        # Skip self if requested
        if exclude_self and target_id is not None and row.get('id') == target_id:
            continue
        
        # Get candidate fingerprint
        candidate_fp = concatenate_fingerprints(row, fingerprint_columns)
        if candidate_fp is None:
            continue
        
        # Calculate similarity
        similarity = calculate_tanimoto_similarity(target_fp, candidate_fp)
        
        # Only include if above threshold
        if similarity >= min_similarity:
            similarities.append((row, similarity))
    
    # Sort by similarity (descending) and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n_similar]


def validate_fingerprint_data(df: pd.DataFrame, fingerprint_columns: List[str]) -> Dict[str, Any]:
    """
    Validate fingerprint data quality in the DataFrame.
    
    Args:
        df: DataFrame containing molecules
        fingerprint_columns: List of fingerprint columns to check
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        "total_molecules": len(df),
        "fingerprint_columns": fingerprint_columns,
        "column_stats": {},
        "valid_molecules": 0,
        "invalid_molecules": 0
    }
    
    for col in fingerprint_columns:
        if col not in df.columns:
            results["column_stats"][col] = {"status": "missing", "available": 0}
            continue
        
        available = df[col].notna().sum()
        results["column_stats"][col] = {
            "status": "available",
            "available": available,
            "missing": len(df) - available,
            "coverage": available / len(df) if len(df) > 0 else 0
        }
    
    # Count molecules with at least one valid fingerprint
    valid_count = 0
    for idx, row in df.iterrows():
        fp = concatenate_fingerprints(row, fingerprint_columns)
        if fp is not None:
            valid_count += 1
    
    results["valid_molecules"] = valid_count
    results["invalid_molecules"] = len(df) - valid_count
    results["overall_coverage"] = valid_count / len(df) if len(df) > 0 else 0
    
    return results


def calculate_mol_similarity(mol1, mol2, fp_type: str = 'morgan') -> float:
    """
    Calculate molecular similarity directly from RDKit molecule objects.
    
    This is more efficient than using pre-computed fingerprints as it generates
    fingerprints on-demand using RDKit's optimized functions.
    
    Args:
        mol1: First RDKit molecule object
        mol2: Second RDKit molecule object  
        fp_type: Type of fingerprint to use ('morgan', 'rdkit', 'maccs')
        
    Returns:
        Tanimoto similarity score (0.0 to 1.0)
    """
    if mol1 is None or mol2 is None:
        return 0.0
    
    try:
        from rdkit.Chem import rdMolDescriptors, MACCSkeys
        
        if fp_type == 'morgan':
            # Morgan fingerprints (ECFP equivalent)
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        elif fp_type == 'rdkit':
            # RDKit fingerprints
            fp1 = rdMolDescriptors.RDKFingerprint(mol1)
            fp2 = rdMolDescriptors.RDKFingerprint(mol2)
        elif fp_type == 'maccs':
            # MACCS keys
            fp1 = MACCSkeys.GenMACCSKeys(mol1)
            fp2 = MACCSkeys.GenMACCSKeys(mol2)
        else:
            # Default to Morgan
            fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
            fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        # Calculate Tanimoto similarity
        return DataStructs.TanimotoSimilarity(fp1, fp2)
        
    except Exception as e:
        logger.error(f"Error calculating molecular similarity: {e}")
        return 0.0


def find_most_similar_molecule_by_mol(
    target_mol_obj,
    df: pd.DataFrame,
    fp_type: str = 'morgan',
    exclude_self: bool = True,
    target_id: Optional[int] = None
) -> Optional[Tuple[pd.Series, float]]:
    """
    Find most similar molecule using RDKit molecule objects directly.
    
    This is more efficient than using pre-computed fingerprints.
    
    Args:
        target_mol_obj: Target RDKit molecule object
        df: DataFrame containing molecules with 'mol' column
        fp_type: Type of fingerprint to use ('morgan', 'rdkit', 'maccs')
        exclude_self: Whether to exclude the target molecule itself
        target_id: ID of target molecule to exclude
        
    Returns:
        Tuple of (most_similar_molecule, similarity_score) or None if no matches
    """
    if target_mol_obj is None or len(df) == 0:
        return None
    
    if 'mol' not in df.columns:
        logger.warning("No 'mol' column found in DataFrame")
        return None
    
    best_similarity = -1.0
    best_molecule = None
    
    for idx, row in df.iterrows():
        # Skip self if requested
        if exclude_self and target_id is not None and row.get('id') == target_id:
            continue
        
        candidate_mol = row['mol']
        if candidate_mol is None:
            continue
        
        # Calculate similarity
        similarity = calculate_mol_similarity(target_mol_obj, candidate_mol, fp_type)
        
        # Update best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_molecule = row
    
    if best_molecule is None:
        return None
    
    return best_molecule, best_similarity
