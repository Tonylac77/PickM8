"""
Feature engineering module for improving ML performance in molecular screening.
Implements variance-based filtering, feature importance, and hybrid fingerprints.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Any, Optional
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import json

logger = logging.getLogger(__name__)


def remove_low_variance_features(X: np.ndarray, threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove features with low variance to improve ML performance.
    
    Args:
        X: Feature matrix
        threshold: Variance threshold (features with variance below this are removed)
        
    Returns:
        Tuple of (filtered_features, selected_indices)
    """
    if X.shape[0] < 2:  # Need at least 2 samples for variance
        return X, np.arange(X.shape[1])
    
    # Create selector
    selector = VarianceThreshold(threshold=threshold)
    
    try:
        # Fit and transform
        X_filtered = selector.fit_transform(X)
        selected_indices = selector.get_support(indices=True)
        
        logger.info(f"Removed {X.shape[1] - len(selected_indices)} low-variance features. "
                   f"Kept {len(selected_indices)} features.")
        
        return X_filtered, selected_indices
    except Exception as e:
        logger.warning(f"Variance filtering failed: {e}. Returning original features.")
        return X, np.arange(X.shape[1])


def calculate_feature_importance(X: np.ndarray, y: np.ndarray, 
                               method: str = 'random_forest',
                               task: str = 'classification') -> np.ndarray:
    """
    Calculate feature importance scores using various methods.
    
    Args:
        X: Feature matrix
        y: Target labels
        method: Method to use ('random_forest', 'mutual_info')
        task: 'classification' or 'regression'
        
    Returns:
        Array of feature importance scores
    """
    if method == 'random_forest':
        # Use Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
    elif method == 'mutual_info':
        # Use mutual information
        if task == 'classification':
            importances = mutual_info_classif(X, y, random_state=42)
        else:
            importances = mutual_info_regression(X, y, random_state=42)
    else:
        # Default to uniform importance
        importances = np.ones(X.shape[1]) / X.shape[1]
    
    return importances


def select_top_features(X: np.ndarray, importances: np.ndarray, 
                       n_features: Optional[int] = None,
                       importance_threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top features based on importance scores.
    
    Args:
        X: Feature matrix
        importances: Feature importance scores
        n_features: Number of top features to select
        importance_threshold: Minimum importance threshold
        
    Returns:
        Tuple of (selected_features, selected_indices)
    """
    if n_features is None and importance_threshold is None:
        # Keep top 75% by default
        n_features = int(0.75 * X.shape[1])
    
    if importance_threshold is not None:
        # Select by threshold
        selected_indices = np.where(importances >= importance_threshold)[0]
    else:
        # Select top n features
        selected_indices = np.argsort(importances)[-n_features:]
    
    X_selected = X[:, selected_indices]
    
    logger.info(f"Selected {len(selected_indices)} features based on importance.")
    
    return X_selected, selected_indices


def create_interaction_enhanced_features(morgan_fp: List[int], 
                                       interaction_fp: List[float],
                                       combination_method: str = 'concatenate') -> np.ndarray:
    """
    Create enhanced features by combining molecular and interaction fingerprints.
    
    Args:
        morgan_fp: Morgan fingerprint
        interaction_fp: Interaction fingerprint
        combination_method: How to combine ('concatenate', 'multiply', 'weighted')
        
    Returns:
        Combined feature vector
    """
    morgan_array = np.array(morgan_fp, dtype=np.float32)
    interaction_array = np.array(interaction_fp, dtype=np.float32)
    
    if combination_method == 'concatenate':
        # Simple concatenation
        combined = np.concatenate([morgan_array, interaction_array])
        
    elif combination_method == 'multiply':
        # Element-wise multiplication for same-sized vectors
        min_len = min(len(morgan_array), len(interaction_array))
        combined = np.concatenate([
            morgan_array[:min_len] * interaction_array[:min_len],
            morgan_array[min_len:],
            interaction_array[min_len:]
        ])
        
    elif combination_method == 'weighted':
        # Weighted combination with learned weights
        weight_morgan = 0.7
        weight_interaction = 0.3
        combined = np.concatenate([
            weight_morgan * morgan_array,
            weight_interaction * interaction_array
        ])
    else:
        combined = np.concatenate([morgan_array, interaction_array])
    
    return combined


def apply_dimensionality_reduction(X: np.ndarray, method: str = 'pca',
                                  n_components: Optional[int] = None,
                                  variance_threshold: float = 0.95) -> Tuple[np.ndarray, Any]:
    """
    Apply dimensionality reduction to features.
    
    Args:
        X: Feature matrix
        method: Reduction method ('pca', 'truncated_svd')
        n_components: Number of components (None for automatic)
        variance_threshold: Cumulative variance to preserve
        
    Returns:
        Tuple of (reduced_features, reducer_object)
    """
    if n_components is None:
        # Automatically determine components to preserve variance
        n_components = min(X.shape[0], X.shape[1])
    
    if method == 'pca' and X.shape[0] >= X.shape[1]:
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        # Use TruncatedSVD for sparse or wide matrices
        reducer = TruncatedSVD(n_components=min(n_components, X.shape[1]-1), random_state=42)
    
    X_reduced = reducer.fit_transform(X)
    
    # Find components explaining desired variance
    if hasattr(reducer, 'explained_variance_ratio_'):
        cumsum = np.cumsum(reducer.explained_variance_ratio_)
        n_components_selected = np.argmax(cumsum >= variance_threshold) + 1
        X_reduced = X_reduced[:, :n_components_selected]
        
        logger.info(f"Reduced dimensions from {X.shape[1]} to {n_components_selected} "
                   f"preserving {cumsum[n_components_selected-1]:.2%} variance")
    
    return X_reduced, reducer


def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Any]:
    """
    Normalize features for better ML performance.
    
    Args:
        X: Feature matrix
        method: Normalization method ('standard', 'minmax')
        
    Returns:
        Tuple of (normalized_features, scaler_object)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, scaler


def create_hybrid_fingerprint_features(df: pd.DataFrame,
                                     use_morgan: bool = True,
                                     use_rdkit: bool = True,
                                     use_interaction: bool = True,
                                     use_mapchiral: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    Create hybrid fingerprint features with enhanced combinations.
    
    Args:
        df: Molecules DataFrame
        use_morgan: Include Morgan fingerprints
        use_rdkit: Include RDKit fingerprints  
        use_interaction: Include interaction fingerprints
        use_mapchiral: Include MapChiral fingerprints
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    features = []
    feature_names = []
    
    for idx, row in df.iterrows():
        mol_features = []
        
        # Add molecular fingerprints
        if use_morgan and row['morgan_fp'] is not None:
            morgan_fp = [int(x) for x in row['morgan_fp']]
            mol_features.extend(morgan_fp)
            if idx == 0:  # Only add names once
                feature_names.extend([f'morgan_{i}' for i in range(len(morgan_fp))])
        
        if use_rdkit and row['rdkit_fp'] is not None:
            rdkit_fp = [int(x) for x in row['rdkit_fp']]
            mol_features.extend(rdkit_fp)
            if idx == 0:
                feature_names.extend([f'rdkit_{i}' for i in range(len(rdkit_fp))])
        
        # Add interaction fingerprints with special handling
        if use_interaction and row['interaction_fp'] is not None:
            try:
                ifp_data = json.loads(row['interaction_fp'])
                if isinstance(ifp_data, list):
                    mol_features.extend(ifp_data)
                    if idx == 0:
                        feature_names.extend([f'interaction_{i}' for i in range(len(ifp_data))])
                        
                    # Add interaction count as meta-feature
                    if 'num_interactions' in row:
                        mol_features.append(row['num_interactions'])
                        if idx == 0:
                            feature_names.append('num_interactions')
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        
        # Add pose quality features if available
        if 'clashes' in row and pd.notna(row['clashes']):
            mol_features.append(float(row['clashes']))
            if idx == 0:
                feature_names.append('clashes')
                
        if 'strain_energy' in row and pd.notna(row['strain_energy']):
            mol_features.append(float(row['strain_energy']))
            if idx == 0:
                feature_names.append('strain_energy')
        
        # Add MapChiral if available
        if use_mapchiral and 'mapchiral_fp' in row.columns and row['mapchiral_fp'] is not None:
            mapchiral_fp = row['mapchiral_fp']
            if isinstance(mapchiral_fp, list):
                mol_features.extend(mapchiral_fp)
                if idx == 0:
                    feature_names.extend([f'mapchiral_{i}' for i in range(len(mapchiral_fp))])
        
        features.append(mol_features)
    
    # Ensure all feature vectors have same length
    if features:
        max_len = max(len(f) for f in features)
        features = [f + [0.0] * (max_len - len(f)) for f in features]
        
        # Pad feature names if needed
        if len(feature_names) < max_len:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), max_len)])
    
    return np.array(features, dtype=np.float32), feature_names


def engineer_features_for_training(df: pd.DataFrame,
                                 variance_threshold: float = 0.01,
                                 use_importance_selection: bool = True,
                                 use_dimensionality_reduction: bool = False,
                                 normalize: bool = True,
                                 config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Complete feature engineering pipeline for ML training.
    
    Args:
        df: Graded molecules DataFrame
        variance_threshold: Threshold for variance filtering
        use_importance_selection: Whether to use feature importance
        use_dimensionality_reduction: Whether to apply PCA/SVD
        normalize: Whether to normalize features
        config: Additional configuration
        
    Returns:
        Tuple of (engineered_features, feature_metadata)
    """
    # Get configuration
    if config is None:
        config = {}
    
    use_morgan = config.get('use_morgan_fp', True)
    use_rdkit = config.get('use_rdkit_fp', True)
    use_interaction = config.get('use_interaction_fp', True)
    use_mapchiral = config.get('use_mapchiral_fp', False)
    
    # Create hybrid features
    X, feature_names = create_hybrid_fingerprint_features(
        df, use_morgan, use_rdkit, use_interaction, use_mapchiral
    )
    
    if len(X) == 0:
        return np.array([]), {}
    
    metadata = {
        'original_shape': X.shape,
        'feature_names': feature_names,
        'transformations': []
    }
    
    # Step 1: Remove low variance features
    if variance_threshold > 0:
        X, selected_indices = remove_low_variance_features(X, variance_threshold)
        metadata['variance_selected_indices'] = selected_indices
        metadata['feature_names'] = [feature_names[i] for i in selected_indices]
        metadata['transformations'].append('variance_filter')
    
    # Step 2: Feature importance selection (requires labels)
    if use_importance_selection and 'grade' in df.columns:
        # Encode grades for importance calculation
        from active_learning.encodings import encode_sequential
        y_encoded, _ = encode_sequential(df['grade'].tolist())
        
        importances = calculate_feature_importance(X, y_encoded)
        X, important_indices = select_top_features(X, importances, n_features=int(0.8 * X.shape[1]))
        
        # Update metadata
        if 'variance_selected_indices' in metadata:
            # Map through variance selection
            metadata['important_indices'] = metadata['variance_selected_indices'][important_indices]
        else:
            metadata['important_indices'] = important_indices
            
        metadata['feature_names'] = [metadata['feature_names'][i] for i in important_indices]
        metadata['feature_importances'] = importances[important_indices]
        metadata['transformations'].append('importance_selection')
    
    # Step 3: Normalize features
    if normalize:
        X, scaler = normalize_features(X)
        metadata['scaler'] = scaler
        metadata['transformations'].append('normalization')
    
    # Step 4: Dimensionality reduction (optional)
    if use_dimensionality_reduction and X.shape[1] > 50:
        X, reducer = apply_dimensionality_reduction(X, n_components=50)
        metadata['reducer'] = reducer
        metadata['transformations'].append('dimensionality_reduction')
        metadata['reduced_shape'] = X.shape
    
    metadata['final_shape'] = X.shape
    
    logger.info(f"Feature engineering complete: {metadata['original_shape']} -> {metadata['final_shape']}")
    logger.info(f"Applied transformations: {metadata['transformations']}")
    
    return X, metadata


def apply_feature_engineering_for_prediction(X_raw: np.ndarray, 
                                           feature_metadata: Dict[str, Any]) -> np.ndarray:
    """
    Apply the same feature engineering transformations used during training.
    
    Args:
        X_raw: Raw feature matrix
        feature_metadata: Metadata from training feature engineering
        
    Returns:
        Transformed feature matrix
    """
    X = X_raw.copy()
    
    # Apply transformations in order
    for transform in feature_metadata.get('transformations', []):
        
        if transform == 'variance_filter' and 'variance_selected_indices' in feature_metadata:
            X = X[:, feature_metadata['variance_selected_indices']]
            
        elif transform == 'importance_selection' and 'important_indices' in feature_metadata:
            # Important indices are relative to variance-filtered features
            X = X[:, feature_metadata['important_indices']]
            
        elif transform == 'normalization' and 'scaler' in feature_metadata:
            X = feature_metadata['scaler'].transform(X)
            
        elif transform == 'dimensionality_reduction' and 'reducer' in feature_metadata:
            X = feature_metadata['reducer'].transform(X)
    
    return X