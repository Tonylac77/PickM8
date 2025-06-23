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
    logger.info(f"Preparing features from DataFrame with {len(df)} molecules")
    
    # Get molecules with computed fingerprints - check which fingerprint types are available
    fingerprint_columns = []
    
    # Check for molecular fingerprints (at least one required)
    if 'morgan_fp' in df.columns and df['morgan_fp'].notna().any():
        fingerprint_columns.append('morgan_fp')
        logger.info("Found Morgan fingerprints")
    
    if 'rdkit_fp' in df.columns and df['rdkit_fp'].notna().any():
        fingerprint_columns.append('rdkit_fp') 
        logger.info("Found RDKit fingerprints")
        
    if 'mapchiral_fp' in df.columns and df['mapchiral_fp'].notna().any():
        fingerprint_columns.append('mapchiral_fp')
        logger.info("Found MapChiral fingerprints")
    
    # Check for interaction fingerprints
    has_interaction_fp = 'interaction_fp' in df.columns and df['interaction_fp'].notna().any()
    if has_interaction_fp:
        logger.info("Found interaction fingerprints")
    
    # Require at least one molecular fingerprint
    if not fingerprint_columns:
        logger.warning("No molecular fingerprints found (morgan_fp, rdkit_fp, or mapchiral_fp)")
        return np.array([]), []
    
    # Build validation mask - require at least one molecular fingerprint and interaction fingerprint
    valid_mask = df[fingerprint_columns[0]].notna()  # Start with first available fingerprint
    for fp_col in fingerprint_columns[1:]:
        valid_mask |= df[fp_col].notna()  # OR with other available fingerprints
    
    if has_interaction_fp:
        valid_mask &= df['interaction_fp'].notna()  # AND with interaction fingerprints
    else:
        logger.warning("No interaction fingerprints found - proceeding with molecular fingerprints only")
    
    valid_df = df[valid_mask].copy()
    
    logger.info(f"Found {len(valid_df)} molecules with valid fingerprints (from {fingerprint_columns})")
    
    if len(valid_df) == 0:
        logger.warning("No molecules with valid fingerprints found")
        return np.array([]), []
    
    features = []
    skipped_molecules = []
    
    for idx, row in valid_df.iterrows():
        try:
            feature_vector = []
            
            # Add molecular fingerprints that are available
            # Morgan fingerprint
            if 'morgan_fp' in fingerprint_columns and pd.notna(row.get('morgan_fp')):
                if isinstance(row['morgan_fp'], list):
                    feature_vector.extend([int(x) for x in row['morgan_fp']])
                else:
                    logger.warning(f"Invalid Morgan fingerprint type for molecule {row['id']}: {type(row['morgan_fp'])}")
            
            # RDKit fingerprint  
            if 'rdkit_fp' in fingerprint_columns and pd.notna(row.get('rdkit_fp')):
                if isinstance(row['rdkit_fp'], list):
                    feature_vector.extend([int(x) for x in row['rdkit_fp']])
                else:
                    logger.warning(f"Invalid RDKit fingerprint type for molecule {row['id']}: {type(row['rdkit_fp'])}")
            
            # MapChiral fingerprint
            if 'mapchiral_fp' in fingerprint_columns and pd.notna(row.get('mapchiral_fp')):
                if isinstance(row['mapchiral_fp'], list):
                    feature_vector.extend([float(x) for x in row['mapchiral_fp']])
                else:
                    logger.warning(f"Invalid MapChiral fingerprint type for molecule {row['id']}: {type(row['mapchiral_fp'])}")
                
            # Interaction fingerprint (convert JSON to numeric) - only if available
            if has_interaction_fp and pd.notna(row.get('interaction_fp')):
                try:
                    ifp_data = json.loads(row['interaction_fp'])
                    if isinstance(ifp_data, list):
                        feature_vector.extend([float(x) for x in ifp_data])
                    elif isinstance(ifp_data, dict):
                        feature_vector.extend([float(x) for x in ifp_data.values()])
                    else:
                        logger.warning(f"Unexpected interaction fingerprint format for molecule {row['id']}: {type(ifp_data)}")
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.warning(f"Error parsing interaction fingerprint for molecule {row['id']}: {e}")
            
            # Add pose quality metrics if requested
            if include_pose_metrics:
                clashes = float(row.get('clashes', 0))
                strain_energy = float(row.get('strain_energy', 0.0))
                num_interactions = float(row.get('num_interactions', 0))
                feature_vector.extend([clashes, strain_energy, num_interactions])
            
            # Add original score if available
            if pd.notna(row.get('score')):
                score = float(row['score'])
                feature_vector.append(score)
            
            if len(feature_vector) == 0:
                logger.warning(f"No features extracted for molecule {row['id']}")
                skipped_molecules.append(row['id'])
                continue
                
            features.append(feature_vector)
            
        except Exception as e:
            logger.error(f"Error processing molecule {row['id']}: {e}")
            skipped_molecules.append(row['id'])
            continue
    
    if skipped_molecules:
        logger.warning(f"Skipped {len(skipped_molecules)} molecules due to feature extraction errors: {skipped_molecules}")
    
    # Remove skipped molecules from valid_df
    if skipped_molecules:
        valid_df = valid_df[~valid_df['id'].isin(skipped_molecules)]
    
    # Ensure all feature vectors have the same length
    if features:
        feature_lengths = [len(f) for f in features]
        max_len = max(feature_lengths)
        min_len = min(feature_lengths)
        
        logger.info(f"Feature vector lengths: min={min_len}, max={max_len}")
        
        if min_len != max_len:
            logger.warning(f"Inconsistent feature vector lengths detected, padding to {max_len}")
            features = [f + [0.0] * (max_len - len(f)) for f in features]
    
    feature_matrix = np.array(features)
    mol_ids = valid_df['id'].tolist()
    
    logger.info(f"Final feature matrix shape: {feature_matrix.shape}, molecule IDs: {len(mol_ids)}")
    
    return feature_matrix, mol_ids


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