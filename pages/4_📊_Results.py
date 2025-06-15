import streamlit as st
from utils.io_handlers import DataHandler
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import zipfile
from pathlib import Path
import json

st.set_page_config(page_title="Results - PickM8", page_icon="📊", layout="wide")

def create_results_archive(data_handler, session_state):
    archive_path = data_handler.session_path / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        # Add all parquet files
        for parquet_file in data_handler.session_path.glob("*.parquet"):
            zipf.write(parquet_file, parquet_file.name)
        
        # Add session state
        if session_state:
            zipf.writestr("session_info.json", json.dumps(session_state, indent=2))
        
        # Add model if exists
        model_path = data_handler.session_path / "model.pkl"
        if model_path.exists():
            zipf.write(model_path, "model.pkl")
    
    return archive_path

def main():
    st.title("📊 Results & Analysis")
    
    if 'session_id' not in st.session_state:
        st.error("No active session.")
        return
    
    data_handler = DataHandler(st.session_state.session_id)
    molecules_df = data_handler.load_molecules()
    grades_df = data_handler.load_grades()
    predictions_df = data_handler.load_predictions()
    session_state = data_handler.load_session_state()
    
    if molecules_df is None:
        st.error("No data available.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Grade Analysis", "Predictions", "Export"])
    
    with tab1:
        st.subheader("Session Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Molecules", len(molecules_df))
        with col2:
            st.metric("Graded", len(grades_df) if grades_df else 0)
        with col3:
            st.metric("Predictions", len(predictions_df) if predictions_df else 0)
        
        if grades_df and not grades_df.is_empty():
            st.subheader("Grading Progress")
            
            merged = molecules_df.join(
                grades_df.select(['mol_id', 'grade', 'timestamp']), 
                left_on='id', 
                right_on='mol_id',
                how='left'
            )
            
            progress_df = grades_df.sort('timestamp').with_row_count('index')
            fig = px.line(progress_df, x='timestamp', y='index', 
                         title='Grading Progress Over Time',
                         labels={'index': 'Cumulative Grades', 'timestamp': 'Time'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if grades_df is None or grades_df.is_empty():
            st.info("No grades available yet.")
        else:
            st.subheader("Grade Distribution")
            
            grade_counts = grades_df.group_by('grade').count().sort('grade')
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(grade_counts, x='grade', y='count',
                           title='Grade Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(grade_counts, values='count', names='grade',
                           title='Grade Proportions')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if predictions_df is None or predictions_df.is_empty():
            st.info("No predictions available. Train a model first.")
        else:
            st.subheader("Model Predictions")
            st.dataframe(predictions_df, use_container_width=True)
    
    with tab4:
        st.subheader("Export Results")
        
        if st.button("📁 Download Results Archive"):
            archive_path = create_results_archive(data_handler, session_state)
            
            with open(archive_path, 'rb') as f:
                st.download_button(
                    label="Download ZIP",
                    data=f.read(),
                    file_name=archive_path.name,
                    mime="application/zip"
                )
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    if st.sidebar.button("🎯 Active Learning"):
        st.switch_page("pages/3_🎯_Active_Learning.py")
    
    if st.sidebar.button("⚙️ Settings"):
        st.switch_page("pages/5_⚙️_Settings.py")
    
    if st.sidebar.button("🏠 Main Entry"):
        st.switch_page("main_entry.py")

if __name__ == "__main__":
    main()