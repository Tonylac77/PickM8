"""
Grading utilities for molecular analysis.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging
import streamlit as st

logger = logging.getLogger(__name__)


def add_grade_to_molecule(df: pd.DataFrame, molecule_id: int, grade: str) -> pd.DataFrame:
    """
    Add or update grade for a specific molecule.
    
    Args:
        df: Molecules DataFrame
        molecule_id: ID of molecule to grade
        grade: Grade ('A', 'B', 'C', 'D', 'F')
        
    Returns:
        Updated DataFrame
    """
    df = df.copy()
    mask = df['id'] == molecule_id
    
    if not mask.any():
        logger.warning(f"Molecule ID {molecule_id} not found")
        return df
        
    df.loc[mask, 'grade'] = grade
    df.loc[mask, 'grade_timestamp'] = pd.Timestamp.now()
    
    logger.info(f"Added grade {grade} to molecule {molecule_id}")
    return df


def get_graded_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get subset of molecules that have been graded.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        DataFrame with only graded molecules
    """
    return df[df['grade'].notna()].copy()


def get_ungraded_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get subset of molecules that have not been graded.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        DataFrame with only ungraded molecules
    """
    return df[df['grade'].isna()].copy()


def filter_and_sort_molecules(df: pd.DataFrame, mode: str, sort_method: str, 
                             suggested_molecules: Optional[list] = None) -> pd.DataFrame:
    """
    Filter and sort molecules based on current mode and sort method.
    When suggested molecules are provided, they are prioritized in annotate mode.
    
    Args:
        df: Molecules DataFrame
        mode: Filter mode ('annotate', 'review', 'all')
        sort_method: Sort method ('random', 'highest_score', 'uncertainty', 'predicted_grade', 'prediction')
        suggested_molecules: List of molecule IDs to prioritize (optional)
        
    Returns:
        Filtered and sorted DataFrame
    """
    if mode == "annotate":
        # Show ungraded molecules first
        filtered_df = get_ungraded_molecules(df)
        
        # If we have suggested molecules and model predictions, prioritize suggested molecules
        if (suggested_molecules and 'prediction_uncertainty' in df.columns and 
            df['prediction_uncertainty'].notna().any()):
            
            # Split into suggested and non-suggested ungraded molecules
            suggested_mask = filtered_df['id'].isin(suggested_molecules)
            suggested_df = filtered_df[suggested_mask].copy()
            other_df = filtered_df[~suggested_mask].copy()
            
            # Sort suggested based on strategy
            if len(suggested_df) > 0:
                if sort_method == "predicted_grade" and 'prediction' in suggested_df.columns:
                    # Create grade ranking for sorting
                    grade_to_score = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
                    suggested_df['pred_score'] = suggested_df['prediction'].map(grade_to_score).fillna(0)
                    suggested_df = suggested_df.sort_values('pred_score', ascending=False)
                    suggested_df = suggested_df.drop('pred_score', axis=1)
                else:
                    # Default to uncertainty sorting
                    suggested_df = suggested_df.sort_values('prediction_uncertainty', ascending=False)
            
            # Sort others by score
            if len(other_df) > 0:
                other_df = other_df.sort_values('score', ascending=True)
            
            # Combine: suggested first, then others
            filtered_df = pd.concat([suggested_df, other_df], ignore_index=True)
            
        else:
            # Normal sorting when no suggestions available
            if sort_method == "random":
                filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
            elif sort_method == "highest_score":
                # Get score direction from session state metadata if available
                score_direction = "Lower is better"  # Default
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'session_id') and st.session_state.session_id:
                    try:
                        from utils.data_processing import load_session_metadata
                        session_dir = f"data/sessions/{st.session_state.session_id}"
                        metadata = load_session_metadata(session_dir)
                        if metadata:
                            score_direction = metadata.get('score_direction', 'Lower is better')
                    except Exception:
                        pass  # Use default if metadata not available
                
                if score_direction == "Higher is better":
                    filtered_df = filtered_df.sort_values('score', ascending=False)
                else:
                    filtered_df = filtered_df.sort_values('score', ascending=True)
            elif sort_method == "uncertainty" and 'prediction_uncertainty' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('prediction_uncertainty', ascending=False, na_position='last')
            elif sort_method == "predicted_grade" and 'prediction' in filtered_df.columns:
                # Create grade ranking for sorting
                grade_to_score = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
                filtered_df['pred_score'] = filtered_df['prediction'].map(grade_to_score).fillna(0)
                filtered_df = filtered_df.sort_values('pred_score', ascending=False, na_position='last')
                filtered_df = filtered_df.drop('pred_score', axis=1)
            else:
                # Fallback to random
                filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
                
    elif mode == "review":
        # Show graded molecules
        filtered_df = get_graded_molecules(df)
        # Sort by grade timestamp (newest first)
        if 'grade_timestamp' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('grade_timestamp', ascending=False, na_position='last')
        else:
            filtered_df = filtered_df.sort_values('score', ascending=True)
    else:
        filtered_df = df.copy()
        # Sort molecules
        if sort_method == "score":
            filtered_df = filtered_df.sort_values('score', ascending=True)
        elif sort_method == "uncertainty" and 'prediction_uncertainty' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('prediction_uncertainty', ascending=False, na_position='last')
        elif sort_method == "prediction" and 'prediction' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('prediction', ascending=False, na_position='last')
        else:
            # Fallback to score
            filtered_df = filtered_df.sort_values('score', ascending=True)
    
    return filtered_df.reset_index(drop=True)