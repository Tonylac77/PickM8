"""
Molecule selection strategies for active learning.
"""

import pandas as pd
import numpy as np
from typing import List
import logging
import streamlit as st

from .features import prepare_features_from_dataframe

logger = logging.getLogger(__name__)


def select_by_score(df: pd.DataFrame, n_molecules: int) -> List[int]:
    """
    Select molecules based on score with direction preference.
    
    Args:
        df: Molecules DataFrame
        n_molecules: Number of molecules to select
        
    Returns:
        List of molecule IDs
    """
    # Get score direction from session state metadata if available
    score_direction = "Lower is better"  # Default
    
    # Check if we have access to session metadata
    if hasattr(st.session_state, 'session_id') and st.session_state.session_id:
        try:
            from utils.data_processing import load_session_metadata
            session_dir = f"data/sessions/{st.session_state.session_id}"
            metadata = load_session_metadata(session_dir)
            if metadata:
                score_direction = metadata.get('score_direction', 'Lower is better')
        except Exception:
            pass  # Use default if metadata not available
    
    # Select based on score direction without modifying actual scores
    if score_direction == "Higher is better":
        # For "higher is better", select molecules with highest scores
        return df.nlargest(n_molecules, 'score')['id'].tolist()
    else:
        # For "lower is better", select molecules with lowest scores
        return df.nsmallest(n_molecules, 'score')['id'].tolist()


def select_molecules_for_labeling(df: pd.DataFrame, n_molecules: int = 10,
                                strategy: str = "uncertainty", **kwargs) -> List[int]:
    """
    Select molecules for manual labeling using active learning strategy.
    
    Args:
        df: Molecules DataFrame
        n_molecules: Number of molecules to select
        strategy: Selection strategy ('uncertainty', 'predicted_grade', 'diverse', 'random')
        **kwargs: Strategy-specific parameters
        
    Returns:
        List of molecule IDs to label
    """
    # Get ungraded molecules
    ungraded = df[df['grade'].isna()].copy()
    
    if len(ungraded) == 0:
        return []
    
    n_molecules = min(n_molecules, len(ungraded))
    
    if strategy == "uncertainty":
        # Select molecules with highest prediction uncertainty
        if 'prediction_uncertainty' in ungraded.columns:
            uncertainty_available = ungraded['prediction_uncertainty'].notna()
            if uncertainty_available.any():
                candidates = ungraded[uncertainty_available].copy()
                candidates = candidates.nlargest(n_molecules, 'prediction_uncertainty')
                return candidates['id'].tolist()
        
        # Fallback to score-based selection (scores are always valid numeric)
        # Use score direction from session metadata if available
        return select_by_score(ungraded, n_molecules)
    
    elif strategy == "predicted_grade":
        # Select molecules with highest predicted grades (A > B > C > D > F)
        if 'prediction' in ungraded.columns:
            prediction_available = ungraded['prediction'].notna()
            if prediction_available.any():
                candidates = ungraded[prediction_available].copy()
                
                # Create grade ranking (A=5, B=4, C=3, D=2, F=1)
                grade_to_score = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
                candidates['pred_score'] = candidates['prediction'].map(grade_to_score).fillna(0)
                
                # Select highest predicted grades
                candidates = candidates.nlargest(n_molecules, 'pred_score')
                return candidates['id'].tolist()
        
        # Fallback to uncertainty if no predictions
        if 'prediction_uncertainty' in ungraded.columns:
            uncertainty_available = ungraded['prediction_uncertainty'].notna()
            if uncertainty_available.any():
                candidates = ungraded[uncertainty_available].copy()
                candidates = candidates.nlargest(n_molecules, 'prediction_uncertainty')
                return candidates['id'].tolist()
        
        # Final fallback to score-based selection
        return select_by_score(ungraded, n_molecules)
    
    elif strategy == "diverse":
        # Select diverse molecules based on features
        try:
            features, mol_ids = prepare_features_from_dataframe(ungraded)
            if len(features) > 0:
                # Use simple distance-based diversity selection
                selected_indices = select_diverse_samples(features, n_molecules)
                return [mol_ids[i] for i in selected_indices]
        except Exception as e:
            logger.warning(f"Error in diverse selection: {e}")
        
        # Fallback to random selection
        return ungraded.sample(n=n_molecules, random_state=42)['id'].tolist()
    
    elif strategy == "random":
        return ungraded.sample(n=n_molecules, random_state=42)['id'].tolist()
    
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")


def select_diverse_samples(features: np.ndarray, n_samples: int) -> List[int]:
    """
    Select diverse samples using greedy farthest-first strategy.
    
    Args:
        features: Feature matrix
        n_samples: Number of samples to select
        
    Returns:
        List of indices of selected samples
    """
    if len(features) <= n_samples:
        return list(range(len(features)))
    
    # Start with random sample
    selected = [np.random.randint(len(features))]
    
    for _ in range(n_samples - 1):
        # Calculate distances from all unselected points to selected points
        distances = []
        for i in range(len(features)):
            if i in selected:
                distances.append(-1)  # Mark as selected
            else:
                # Find minimum distance to any selected point
                min_dist = float('inf')
                for j in selected:
                    dist = np.linalg.norm(features[i] - features[j])
                    min_dist = min(min_dist, dist)
                distances.append(min_dist)
        
        # Select point with maximum minimum distance
        next_idx = np.argmax(distances)
        selected.append(next_idx)
    
    return selected