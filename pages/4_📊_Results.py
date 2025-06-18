"""
Results & Analysis Interface for PickM8
Using functional data processing approach.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import tempfile

# Import new functional utilities
from utils.data_processing import (
    load_molecules_dataframe, load_session_metadata
)
from core.grading import (
    get_graded_molecules, get_ungraded_molecules
)
from utils.export_utils import (
    export_to_csv, export_to_sdf, export_graded_molecules_only,
    export_predictions_summary, export_pose_quality_report,
    export_interaction_analysis, create_export_package,
    validate_export_data
)
from core.pose_analysis import get_pose_quality_statistics
from core.active_learning import get_training_statistics

st.set_page_config(page_title="Results - PickM8", page_icon="📊", layout="wide")


def create_grade_distribution_plots(df: pd.DataFrame):
    """Create grade distribution visualizations."""
    graded_df = get_graded_molecules(df)
    
    if len(graded_df) == 0:
        st.info("No grades available yet.")
        return
    
    # Grade counts
    grade_counts = graded_df['grade'].value_counts().reset_index()
    grade_counts.columns = ['grade', 'count']
    grade_counts = grade_counts.sort_values('grade')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            grade_counts, x='grade', y='count',
            title='Grade Distribution',
            color='grade',
            color_discrete_map={
                'A': '#2E8B57',  # SeaGreen
                'B': '#1E90FF',  # DodgerBlue
                'C': '#FFD700',  # Gold
                'D': '#FF8C00',  # DarkOrange
                'F': '#DC143C'   # Crimson
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            grade_counts, values='count', names='grade',
            title='Grade Proportions',
            color='grade',
            color_discrete_map={
                'A': '#2E8B57', 'B': '#1E90FF', 'C': '#FFD700',
                'D': '#FF8C00', 'F': '#DC143C'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)


def create_grading_progress_plot(df: pd.DataFrame):
    """Create grading progress over time plot."""
    graded_df = get_graded_molecules(df)
    
    if len(graded_df) == 0 or 'grade_timestamp' not in graded_df.columns:
        return
    
    # Sort by timestamp and create cumulative count
    progress_df = graded_df.sort_values('grade_timestamp').reset_index(drop=True)
    progress_df['cumulative'] = range(1, len(progress_df) + 1)
    
    fig = px.line(
        progress_df, x='grade_timestamp', y='cumulative',
        title='Grading Progress Over Time',
        labels={'cumulative': 'Cumulative Grades', 'grade_timestamp': 'Time'}
    )
    fig.update_traces(line_color='#1f77b4', line_width=3)
    st.plotly_chart(fig, use_container_width=True)


def create_score_vs_grade_plot(df: pd.DataFrame):
    """Create score vs grade scatter plot."""
    graded_df = get_graded_molecules(df)
    
    if len(graded_df) == 0:
        return
    
    fig = px.box(
        graded_df, x='grade', y='score',
        title='Score Distribution by Grade',
        color='grade',
        color_discrete_map={
            'A': '#2E8B57', 'B': '#1E90FF', 'C': '#FFD700',
            'D': '#FF8C00', 'F': '#DC143C'
        }
    )
    st.plotly_chart(fig, use_container_width=True)


def create_prediction_analysis_plots(df: pd.DataFrame):
    """Create prediction analysis visualizations."""
    pred_df = df[df['prediction'].notna()].copy()
    
    if len(pred_df) == 0:
        st.info("No predictions available. Train a model first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        fig_hist = px.histogram(
            pred_df, x='prediction',
            title='Prediction Distribution',
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Uncertainty distribution
        if 'prediction_uncertainty' in pred_df.columns:
            fig_unc = px.histogram(
                pred_df, x='prediction_uncertainty',
                title='Prediction Uncertainty Distribution',
                nbins=20
            )
            st.plotly_chart(fig_unc, use_container_width=True)
    
    # Prediction vs actual grade (if available)
    graded_pred_df = pred_df[pred_df['grade'].notna()]
    if len(graded_pred_df) > 0:
        col3, col4 = st.columns(2)
        
        with col3:
            fig_scatter = px.scatter(
                graded_pred_df, x='prediction', y='score', color='grade',
                title='Prediction vs Score (Colored by Grade)',
                color_discrete_map={
                    'A': '#2E8B57', 'B': '#1E90FF', 'C': '#FFD700',
                    'D': '#FF8C00', 'F': '#DC143C'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col4:
            if 'prediction_uncertainty' in graded_pred_df.columns:
                fig_unc_grade = px.box(
                    graded_pred_df, x='grade', y='prediction_uncertainty',
                    title='Prediction Uncertainty by Grade',
                    color='grade',
                    color_discrete_map={
                        'A': '#2E8B57', 'B': '#1E90FF', 'C': '#FFD700',
                        'D': '#FF8C00', 'F': '#DC143C'
                    }
                )
                st.plotly_chart(fig_unc_grade, use_container_width=True)


def create_active_learning_progression_plots(df: pd.DataFrame):
    """Create active learning progression visualizations."""
    graded_df = get_graded_molecules(df)
    
    if len(graded_df) == 0:
        st.info("No graded molecules available yet to analyze active learning progression.")
        return
    
    # If we don't have timestamps, we can't show progression
    if 'grade_timestamp' not in graded_df.columns or graded_df['grade_timestamp'].isna().all():
        st.info("No timestamp data available for active learning progression analysis.")
        return
    
    # Sort by timestamp
    graded_df_sorted = graded_df.sort_values('grade_timestamp').reset_index(drop=True)
    graded_df_sorted['grading_order'] = range(1, len(graded_df_sorted) + 1)
    
    # Create grade encoding for numerical analysis
    grade_to_score = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    graded_df_sorted['grade_numeric'] = graded_df_sorted['grade'].map(grade_to_score)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Progression of grades over time
        fig_progression = px.scatter(
            graded_df_sorted, 
            x='grading_order', 
            y='grade_numeric',
            color='grade',
            title='Grade Quality Progression Over Time',
            labels={
                'grading_order': 'Grading Order',
                'grade_numeric': 'Grade Quality Score',
                'grade': 'Grade'
            },
            color_discrete_map={
                'A': '#2E8B57', 'B': '#1E90FF', 'C': '#FFD700',
                'D': '#FF8C00', 'F': '#DC143C'
            }
        )
        
        # Add trendline
        fig_progression.add_scatter(
            x=graded_df_sorted['grading_order'],
            y=graded_df_sorted['grade_numeric'].rolling(window=min(10, len(graded_df_sorted))).mean(),
            mode='lines',
            name='Trend (Moving Average)',
            line=dict(color='black', width=2, dash='dash')
        )
        
        fig_progression.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['F', 'D', 'C', 'B', 'A']
            )
        )
        st.plotly_chart(fig_progression, use_container_width=True)
    
    with col2:
        # Moving average of grades over time
        window_size = min(5, len(graded_df_sorted))
        if window_size > 1:
            graded_df_sorted['moving_avg'] = graded_df_sorted['grade_numeric'].rolling(
                window=window_size, min_periods=1
            ).mean()
            
            fig_avg = px.line(
                graded_df_sorted,
                x='grading_order',
                y='moving_avg',
                title=f'Grade Quality Trend (Moving Average, Window={window_size})',
                labels={
                    'grading_order': 'Grading Order',
                    'moving_avg': 'Average Grade Quality'
                }
            )
            fig_avg.update_layout(
                yaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2, 3, 4, 5],
                    ticktext=['F', 'D', 'C', 'B', 'A'],
                    range=[0.5, 5.5]
                )
            )
            st.plotly_chart(fig_avg, use_container_width=True)
    
    # Summary statistics
    if len(graded_df_sorted) >= 10:
        st.subheader("Active Learning Effectiveness")
        
        # Compare first 10 vs last 10 grades
        first_10 = graded_df_sorted.head(10)['grade_numeric'].mean()
        last_10 = graded_df_sorted.tail(10)['grade_numeric'].mean()
        improvement = last_10 - first_10
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "First 10 Avg", 
                f"{first_10:.2f}",
                help="Average grade quality of first 10 molecules graded"
            )
        with col2:
            st.metric(
                "Last 10 Avg", 
                f"{last_10:.2f}",
                delta=f"{improvement:+.2f}",
                help="Average grade quality of last 10 molecules graded"
            )
        with col3:
            effectiveness = "Improving" if improvement > 0 else "Declining" if improvement < 0 else "Stable"
            st.metric("AL Effectiveness", effectiveness)


def display_summary_statistics(df: pd.DataFrame):
    """Display summary statistics."""
    total_molecules = len(df)
    graded_count = df['grade'].notna().sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Molecules", total_molecules)
    
    with col2:
        st.metric("Graded", graded_count)
        if total_molecules > 0:
            progress = graded_count / total_molecules
            st.progress(progress)
    
    with col3:
        pred_count = df['prediction'].notna().sum()
        st.metric("With Predictions", pred_count)
    
    with col4:
        if 'clashes' in df.columns:
            clash_free = (df['clashes'] == 0).sum()
            st.metric("Clash-Free Poses", clash_free)


def export_interface(df: pd.DataFrame, session_dir: str, session_name: str):
    """Export interface for results."""
    st.subheader("📁 Export Results")
    
    # Validate data before export
    validation_issues = validate_export_data(df)
    
    if validation_issues['errors']:
        st.error("Data validation errors found:")
        for error in validation_issues['errors']:
            st.error(f"• {error}")
        return
    
    if validation_issues['warnings']:
        st.warning("Data validation warnings:")
        for warning in validation_issues['warnings']:
            st.warning(f"• {warning}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Individual Exports")
        
        # CSV export
        if st.button("📄 Export All Data (CSV)", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                export_to_csv(df, tmp.name, include_predictions=True, include_pose_metrics=True)
                
                with open(tmp.name, 'rb') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f.read(),
                        file_name=f"{session_name}_complete.csv",
                        mime="text/csv"
                    )
        
        # SDF export
        if st.button("🧪 Export Molecules (SDF)", use_container_width=True):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
                    export_to_sdf(df, tmp.name, include_predictions=True, include_pose_metrics=True)
                    
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download SDF",
                            data=f.read(),
                            file_name=f"{session_name}_complete.sdf",
                            mime="chemical/x-mdl-sdfile"
                        )
            except Exception as e:
                st.error(f"SDF export failed: {e}")
        
        # Graded molecules only
        graded_count = df['grade'].notna().sum()
        if graded_count > 0:
            if st.button(f"⭐ Export Graded Only ({graded_count} molecules)", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    export_graded_molecules_only(df, tmp.name, format="csv")
                    
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download Graded CSV",
                            data=f.read(),
                            file_name=f"{session_name}_graded.csv",
                            mime="text/csv"
                        )
    
    with col2:
        st.markdown("#### Analysis Reports")
        
        # Predictions summary
        pred_count = df['prediction'].notna().sum()
        if pred_count > 0:
            if st.button(f"🤖 Predictions Summary ({pred_count} predictions)", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    export_predictions_summary(df, tmp.name)
                    
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download Predictions Report",
                            data=f.read(),
                            file_name=f"{session_name}_predictions.csv",
                            mime="text/csv"
                        )
        
        # Pose quality report
        if 'clashes' in df.columns or 'strain_energy' in df.columns:
            if st.button("🏗️ Pose Quality Report", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    export_pose_quality_report(df, tmp.name)
                    
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download Pose Quality Report",
                            data=f.read(),
                            file_name=f"{session_name}_pose_quality.csv",
                            mime="text/csv"
                        )
        
        # Interaction analysis
        interaction_count = df['interactions'].notna().sum()
        if interaction_count > 0:
            if st.button(f"🔗 Interaction Analysis ({interaction_count} molecules)", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    export_interaction_analysis(df, tmp.name)
                    
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download Interaction Analysis",
                            data=f.read(),
                            file_name=f"{session_name}_interactions.csv",
                            mime="text/csv"
                        )
    
    st.divider()
    
    # Complete export package
    st.markdown("#### Complete Export Package")
    if st.button("📦 Create Complete Export Package", type="primary", use_container_width=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                create_export_package(df, tmp_dir, session_name)
                
                # Create ZIP archive
                import zipfile
                zip_path = f"{tmp_dir}/{session_name}_complete.zip"
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    tmp_path = Path(tmp_dir)
                    for file_path in tmp_path.glob("*"):
                        if file_path.is_file() and not file_path.name.endswith('.zip'):
                            zipf.write(file_path, file_path.name)
                
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="Download Complete Package (ZIP)",
                        data=f.read(),
                        file_name=f"{session_name}_complete.zip",
                        mime="application/zip"
                    )
                
                st.success("✅ Export package created successfully!")
                
            except Exception as e:
                st.error(f"Export package creation failed: {e}")


def main():
    """Main Results interface."""
    st.title("📊 Results & Analysis")
    
    # Check session
    if not st.session_state.get('session_id'):
        st.error("No session loaded. Please go to the main page and load a session.")
        if st.button("🏠 Go to Main Page"):
            st.switch_page("main.py")
        return
    
    session_dir = f"data/sessions/{st.session_state.session_id}"
    
    # Load data
    molecules_df = load_molecules_dataframe(session_dir)
    if molecules_df is None:
        st.error("No molecules loaded. Please upload data first.")
        return
    
    session_metadata = load_session_metadata(session_dir)
    session_name = session_metadata.get('protein_name', 'pickm8_session') if session_metadata else 'pickm8_session'
    
    # Remove file extension if present
    if session_name.endswith('.pdb'):
        session_name = session_name[:-4]
    
    # Navigation sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        
        if st.button("🎯 Active Learning", use_container_width=True):
            st.switch_page("pages/3_🎯_Active_Learning.py")
        
        if st.button("🏠 Main Page", use_container_width=True):
            st.switch_page("main.py")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Active Learning", "Grades", "Predictions", "Export"])
    
    with tab1:
        st.subheader("Active Learning Results")
        
        # Summary statistics
        display_summary_statistics(molecules_df)
        
        st.divider()
        
        # Active learning progression analysis
        create_active_learning_progression_plots(molecules_df)
        
        st.divider()
        
        # Session information
        if session_metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Session Information:**")
                st.write(f"**Protein:** {session_metadata.get('protein_name', 'Unknown')}")
                st.write(f"**Created:** {session_metadata.get('created_date', 'Unknown')[:19]}")
                st.write(f"**Score Property:** {session_metadata.get('score_label', 'score')}")
                st.write(f"**Score Direction:** {session_metadata.get('score_direction', 'Lower is better')}")
                st.write(f"**Interaction Type:** {session_metadata.get('interaction_type', 'Unknown').upper()}")
            
            with col2:
                # Training statistics
                training_stats = get_training_statistics(molecules_df)
                st.markdown("**Training Status:**")
                st.write(f"**Graded:** {training_stats['graded_molecules']}/{training_stats['total_molecules']}")
                st.write(f"**Progress:** {training_stats['grading_percentage']:.1f}%")
                if training_stats.get('grade_distribution'):
                    most_common = training_stats.get('most_common_grade', 'N/A')
                    st.write(f"**Most Common Grade:** {most_common}")
                
                # Model status
                if 'prediction' in molecules_df.columns and molecules_df['prediction'].notna().any():
                    pred_count = molecules_df['prediction'].notna().sum()
                    st.write(f"**Predictions:** {pred_count} molecules")
                    st.success("🤖 Model Trained")
    
    with tab2:
        st.subheader("Grade Analysis")
        
        # Grade distribution
        create_grade_distribution_plots(molecules_df)
        
        # Score vs grade analysis
        if molecules_df['grade'].notna().any():
            st.subheader("Score vs Grade Analysis")
            create_score_vs_grade_plot(molecules_df)
    
    with tab3:
        st.subheader("Prediction Analysis")
        create_prediction_analysis_plots(molecules_df)
        
        # Show prediction table if available
        pred_df = molecules_df[molecules_df['prediction'].notna()]
        if len(pred_df) > 0:
            st.subheader("Recent Predictions")
            
            display_columns = ['name', 'score', 'prediction', 'prediction_uncertainty', 'grade']
            display_columns = [col for col in display_columns if col in pred_df.columns]
            
            # Sort by uncertainty (highest first)
            if 'prediction_uncertainty' in pred_df.columns:
                pred_df_sorted = pred_df.sort_values('prediction_uncertainty', ascending=False)
            else:
                pred_df_sorted = pred_df.sort_values('prediction')
            
            st.dataframe(
                pred_df_sorted[display_columns].head(20),
                use_container_width=True,
                hide_index=True
            )
    
    with tab4:
        export_interface(molecules_df, session_dir, session_name)


if __name__ == "__main__":
    main()