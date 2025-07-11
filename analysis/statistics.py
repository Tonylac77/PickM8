"""Statistical analysis functions."""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def calculate_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive dataset statistics.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary with dataset statistics
    """
    if len(df) == 0:
        return {"total_molecules": 0}
    
    stats = {
        "total_molecules": len(df),
        "score_statistics": {},
        "grading_statistics": {},
        "fingerprint_statistics": {},
        "interaction_statistics": {},
        "pose_quality_statistics": {}
    }
    
    # Score statistics
    if 'score' in df.columns:
        score_data = df['score'].dropna()
        if len(score_data) > 0:
            stats["score_statistics"] = {
                "mean": float(score_data.mean()),
                "median": float(score_data.median()),
                "std": float(score_data.std()),
                "min": float(score_data.min()),
                "max": float(score_data.max()),
                "count": len(score_data)
            }
    
    # Grading statistics
    if 'grade' in df.columns:
        graded_count = df['grade'].notna().sum()
        stats["grading_statistics"] = {
            "graded_count": graded_count,
            "ungraded_count": len(df) - graded_count,
            "grading_percentage": (graded_count / len(df) * 100) if len(df) > 0 else 0
        }
        
        if graded_count > 0:
            grade_dist = df['grade'].value_counts().to_dict()
            stats["grading_statistics"]["grade_distribution"] = grade_dist
    
    # Fingerprint statistics
    fingerprint_cols = ['morgan_fp', 'rdkit_fp', 'mapchiral_fp', 'interaction_fp']
    fp_stats = {}
    for col in fingerprint_cols:
        if col in df.columns:
            computed = df[col].notna().sum()
            fp_stats[f"{col}_computed"] = computed
            fp_stats[f"{col}_percentage"] = (computed / len(df) * 100) if len(df) > 0 else 0
    stats["fingerprint_statistics"] = fp_stats
    
    # Interaction statistics
    if 'num_interactions' in df.columns:
        interaction_data = df['num_interactions'].dropna()
        if len(interaction_data) > 0:
            stats["interaction_statistics"] = {
                "mean_interactions": float(interaction_data.mean()),
                "max_interactions": int(interaction_data.max()),
                "molecules_with_interactions": int((interaction_data > 0).sum())
            }
    
    # Pose quality statistics
    if 'clashes' in df.columns:
        clash_data = df['clashes'].dropna()
        if len(clash_data) > 0:
            stats["pose_quality_statistics"]["clash_statistics"] = {
                "mean_clashes": float(clash_data.mean()),
                "max_clashes": int(clash_data.max()),
                "clash_free_molecules": int((clash_data == 0).sum())
            }
    
    if 'strain_energy' in df.columns:
        strain_data = df['strain_energy'].dropna()
        if len(strain_data) > 0:
            stats["pose_quality_statistics"]["strain_statistics"] = {
                "mean_strain": float(strain_data.mean()),
                "max_strain": float(strain_data.max()),
                "high_strain_molecules": int((strain_data > 10.0).sum())
            }
    
    return stats

def calculate_correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary with correlation data
    """
    numeric_cols = ['score', 'num_interactions', 'clashes', 'strain_energy']
    available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
    
    if len(available_cols) < 2:
        return {"error": "Not enough numeric columns for correlation analysis"}
    
    correlation_data = df[available_cols].corr()
    
    return {
        "correlation_matrix": correlation_data.to_dict(),
        "columns": available_cols,
        "sample_size": len(df[available_cols].dropna())
    }