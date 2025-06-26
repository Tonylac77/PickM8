"""Progress tracking UI components."""
import streamlit as st
from typing import Dict, Any

def display_progress_metrics(stats: Dict[str, Any]):
    """Display progress metrics in a nice layout."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Molecules", 
            stats.get('total_molecules', 0)
        )
    
    with col2:
        st.metric(
            "Graded", 
            stats.get('graded_count', 0),
            f"{stats.get('grading_percentage', 0):.1f}%"
        )
    
    with col3:
        st.metric(
            "Remaining", 
            stats.get('ungraded_count', 0)
        )
    
    # Progress bar
    progress = stats.get('grading_percentage', 0) / 100
    st.progress(progress)

def display_processing_status(processing_stats: Dict[str, Any]):
    """Display processing status."""
    st.subheader("Processing Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Fingerprints Computed", 
                 processing_stats.get('morgan_fp_computed', 0))
        st.metric("Interactions Computed", 
                 processing_stats.get('interaction_fp_computed', 0))
    
    with col2:
        st.metric("Pose Quality Analyzed", 
                 processing_stats.get('molecules_with_clash_data', 0))
        st.metric("Success Rate", 
                 f"{processing_stats.get('success_rate', 0):.1f}%")