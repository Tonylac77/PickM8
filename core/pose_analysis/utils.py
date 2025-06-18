"""
Utilities for pose quality analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


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
            "high_strain_molecules": int((strain_data > 10.0).sum())  # Arbitrary threshold
        })
    
    return stats


def filter_by_pose_quality(df: pd.DataFrame, max_clashes: Optional[int] = None,
                         max_strain_energy: Optional[float] = None) -> pd.DataFrame:
    """
    Filter molecules based on pose quality criteria.
    
    Args:
        df: Molecules DataFrame
        max_clashes: Maximum allowed clashes
        max_strain_energy: Maximum allowed strain energy
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    if max_clashes is not None:
        filtered_df = filtered_df[filtered_df['clashes'] <= max_clashes]
        
    if max_strain_energy is not None:
        filtered_df = filtered_df[filtered_df['strain_energy'] <= max_strain_energy]
    
    logger.info(f"Filtered from {len(df)} to {len(filtered_df)} molecules based on pose quality")
    
    return filtered_df


def rank_by_pose_quality(df: pd.DataFrame, clash_weight: float = 1.0, 
                        strain_weight: float = 0.1) -> pd.DataFrame:
    """
    Rank molecules by pose quality score.
    
    Args:
        df: Molecules DataFrame
        clash_weight: Weight for clash score in ranking
        strain_weight: Weight for strain energy in ranking
        
    Returns:
        DataFrame with pose quality ranking
    """
    df = df.copy()
    
    # Calculate composite pose quality score (lower is better)
    df['pose_quality_score'] = (
        df['clashes'] * clash_weight + 
        df['strain_energy'] * strain_weight
    )
    
    # Rank by pose quality (1 = best quality)
    df['pose_quality_rank'] = df['pose_quality_score'].rank(method='min')
    
    return df.sort_values('pose_quality_rank')


def validate_pose_quality_data(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Validate pose quality data and return molecules with issues.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary mapping issue types to lists of molecule IDs
    """
    issues = {
        "missing_clash_data": [],
        "missing_strain_data": [],
        "negative_clashes": [],
        "extreme_strain_values": []
    }
    
    for _, row in df.iterrows():
        mol_id = row['id']
        
        # Check for missing data
        if pd.isna(row['clashes']):
            issues["missing_clash_data"].append(mol_id)
            
        if pd.isna(row['strain_energy']):
            issues["missing_strain_data"].append(mol_id)
        
        # Check for invalid values
        if not pd.isna(row['clashes']) and row['clashes'] < 0:
            issues["negative_clashes"].append(mol_id)
            
        if not pd.isna(row['strain_energy']) and abs(row['strain_energy']) > 1000:
            issues["extreme_strain_values"].append(mol_id)
    
    return issues