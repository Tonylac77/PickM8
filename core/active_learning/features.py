"""
Feature extraction for active learning.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def prepare_features_from_dataframe(df: pd.DataFrame, include_pose_metrics: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Extract feature matrix from molecules DataFrame for ML training/prediction.
    
    Args:
        df: Molecules DataFrame
        include_pose_metrics: Whether to include pose quality metrics as features
        
    Returns:
        Tuple of (feature_matrix, molecule_ids)
    """
    # Get molecules with computed fingerprints
    valid_mask = (df['morgan_fp'].notna()) & (df['interaction_fp'].notna())
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return np.array([]), []
    
    features = []
    for _, row in valid_df.iterrows():
        feature_vector = []
        
        # Morgan fingerprint
        if isinstance(row['morgan_fp'], list):
            feature_vector.extend([int(x) for x in row['morgan_fp']])
        
        # RDKit fingerprint  
        if isinstance(row['rdkit_fp'], list):
            feature_vector.extend([int(x) for x in row['rdkit_fp']])
            
        # Interaction fingerprint (convert JSON to numeric)
        if row['interaction_fp'] is not None:
            try:
                ifp_data = json.loads(row['interaction_fp'])
                if isinstance(ifp_data, list):
                    feature_vector.extend([float(x) for x in ifp_data])
                elif isinstance(ifp_data, dict):
                    feature_vector.extend([float(x) for x in ifp_data.values()])
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        
        # Add pose quality metrics if requested
        if include_pose_metrics:
            feature_vector.append(float(row.get('clashes', 0)))
            feature_vector.append(float(row.get('strain_energy', 0.0)))
            feature_vector.append(float(row.get('num_interactions', 0)))
        
        # Add original score if available
        if pd.notna(row.get('score')):
            feature_vector.append(float(row['score']))
            
        features.append(feature_vector)
    
    # Ensure all feature vectors have the same length
    if features:
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]
    
    return np.array(features), valid_df['id'].tolist()


def get_molecule_features(df: pd.DataFrame, include_predictions: bool = False) -> Tuple[np.ndarray, List[int]]:
    """
    Extract feature matrix and molecule IDs for ML training/prediction.
    
    Args:
        df: Molecules DataFrame
        include_predictions: Whether to include prediction features
        
    Returns:
        Tuple of (feature_matrix, molecule_ids)
    """
    # Get molecules with computed fingerprints
    valid_mask = (df['morgan_fp'].notna()) & (df['interaction_fp'].notna())
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        return np.array([]), []
    
    features = []
    for _, row in valid_df.iterrows():
        # Combine molecular fingerprints
        feature_vector = []
        
        # Morgan fingerprint
        if row['morgan_fp'] is not None:
            feature_vector.extend(row['morgan_fp'])
        
        # RDKit fingerprint  
        if row['rdkit_fp'] is not None:
            feature_vector.extend(row['rdkit_fp'])
            
        # Interaction fingerprint (convert JSON to numeric)
        if row['interaction_fp'] is not None:
            try:
                ifp_data = json.loads(row['interaction_fp'])
                if isinstance(ifp_data, list):
                    feature_vector.extend(ifp_data)
                elif isinstance(ifp_data, dict):
                    # Convert dict values to list
                    feature_vector.extend(list(ifp_data.values()))
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Add additional features if requested
        if include_predictions and pd.notna(row['prediction']):
            feature_vector.append(row['prediction'])
            feature_vector.append(row['prediction_uncertainty'])
            
        features.append(feature_vector)
    
    # Ensure all feature vectors have the same length
    if features:
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]
    
    return np.array(features), valid_df['id'].tolist()