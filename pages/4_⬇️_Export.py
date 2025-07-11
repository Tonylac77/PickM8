"""
Export Interface for PickM8
"""

import streamlit as st
import pandas as pd
import tempfile
import zipfile
import io
from pathlib import Path
from rdkit.Chem import PandasTools

# Import from new modular structure
from sessions import sessions
from analysis import grading

st.set_page_config(page_title="Export - PickM8", page_icon="media/pickm8_white_logoonly.png", layout="wide")


def export_interface(df: pd.DataFrame, session_id: str, session_name: str):
    """Export interface for results."""
    st.subheader("📁 Export Results")
    
    # Basic validation
    if df.empty:
        st.error("No data to export")
        return
    
    st.info(f"Ready to export {len(df)} molecules")
    
    st.markdown("#### Individual Exports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        if st.button("📄 Export All Data (CSV)", use_container_width=True):
            sorted_df = df.sort_values(by=['grade', 'score'], ascending=[True, False])
            sorted_df = sorted_df.reset_index(drop=True)
            csv_data = sorted_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{session_name}_complete.csv",
                mime="text/csv"
            )
    
    with col2:
        # SDF export
        if st.button("🧪 Export Molecules (SDF)", use_container_width=True):
            if 'mol_block' in df.columns and 'mol' in df.columns:
                export_df = df.copy()
                
                # Columns to export, similar to CSV
                export_cols = [
                    'id', 'name', 'smiles', 'score', 'grade', 'grade_timestamp',
                    'prediction', 'clashes', 'strain_energy',
                    'num_interactions'
                ]
                
                # Filter to columns that actually exist in the DataFrame
                available_cols = [col for col in export_cols if col in export_df.columns]
                
                # Ensure mol column is present
                if 'mol' not in export_df.columns:
                    st.error("'mol' column not found, cannot generate SDF.")
                    return
                
                export_df = export_df.sort_values(by=['grade', 'score'], ascending=[True, False])
                export_df = export_df.reset_index(drop=True)

                # Use a temporary file to write SDF data
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf", mode='w') as tmpfile:
                    PandasTools.WriteSDF(export_df, tmpfile.name, molColName='mol', idName='id', properties=available_cols)
                
                # Read the written SDF file
                with open(tmpfile.name, 'r') as f:
                    sdf_data = f.read()

                st.download_button(
                    label="Download SDF",
                    data=sdf_data,
                    file_name=f"{session_name}_complete.sdf",
                    mime="chemical/x-mdl-sdfile"
                )
            else:
                st.error("No molecular structure data available for SDF export.")
    
    # Graded molecules only
    graded_count = df['grade'].notna().sum()
    if graded_count > 0:
        st.markdown("#### Graded Data Only")
        if st.button(f"⭐ Export Graded Only ({graded_count} molecules)", use_container_width=True):
            graded_df = grading.filter_and_sort_molecules(df, mode='graded')
            graded_df = graded_df.sort_values(by=['grade', 'score'], ascending=[True, False])
            graded_df = graded_df.reset_index(drop=True)
            
            csv_data = graded_df.to_csv(index=False)
            st.download_button(
                label="Download Graded CSV",
                data=csv_data,
                file_name=f"{session_name}_graded.csv",
                mime="text/csv"
            )
    
    st.divider()
    
    # Complete export package
    st.markdown("#### Complete Export Package")
    st.write("Download all data formats in a single ZIP file:")
    
    if st.button("📦 Create Complete Export Package", type="primary", use_container_width=True):
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Sort data by grade and score
            sorted_df = df.sort_values(by=['grade', 'score'], ascending=[True, False])
            sorted_df = sorted_df.reset_index(drop=True)
            
            # Add main CSV
            zipf.writestr(f"{session_name}_complete.csv", sorted_df.to_csv(index=False))
            
            # Add SDF file if molecular data is available
            if 'mol_block' in df.columns and 'mol' in df.columns:
                export_cols = [
                    'id', 'name', 'smiles', 'score', 'grade', 'grade_timestamp',
                    'prediction', 'clashes', 'strain_energy', 'num_interactions'
                ]
                available_cols = [col for col in export_cols if col in sorted_df.columns]
                
                # Create SDF content using temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf", mode='w') as tmpfile:
                    PandasTools.WriteSDF(sorted_df, tmpfile.name, molColName='mol', idName='id', properties=available_cols)
                
                # Read SDF content and add to ZIP
                with open(tmpfile.name, 'r') as f:
                    sdf_data = f.read()
                zipf.writestr(f"{session_name}_complete.sdf", sdf_data)
            
            # Add graded molecules only CSV if available
            if df['grade'].notna().any():
                graded_df = grading.filter_and_sort_molecules(df, mode='graded')
                graded_df = graded_df.sort_values(by=['grade', 'score'], ascending=[True, False])
                graded_df = graded_df.reset_index(drop=True)
                zipf.writestr(f"{session_name}_graded.csv", graded_df.to_csv(index=False))
        
        st.download_button(
            label="Download Complete Package (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=f"{session_name}_complete.zip",
            mime="application/zip"
        )
        
        st.success("✅ Export package created successfully!")


def display_export_summary(df: pd.DataFrame):
    """Display summary of available data for export."""
    st.subheader("📊 Export Summary")
    
    total_molecules = len(df)
    graded_count = df['grade'].notna().sum()
    pred_count = df['prediction'].notna().sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Molecules", total_molecules)
    
    with col2:
        st.metric("Graded Molecules", graded_count)
        if total_molecules > 0:
            progress = graded_count / total_molecules
            st.progress(progress)
    
    with col3:
        st.metric("With Predictions", pred_count)
    
    # Additional data availability info
    st.markdown("#### Available Data Types")
    
    data_types = []
    if 'mol' in df.columns and 'mol_block' in df.columns:
        data_types.append("🧪 Molecular Structures (SDF)")
    if 'clashes' in df.columns:
        data_types.append("⚔️ Clash Data")
    if 'strain_energy' in df.columns:
        data_types.append("⚡ Strain Energy")
    if 'num_interactions' in df.columns:
        data_types.append("🔗 Interaction Data")
    if any(col.endswith('_fp') for col in df.columns):
        data_types.append("🔍 Molecular Fingerprints")
    
    if data_types:
        for data_type in data_types:
            st.write(f"✅ {data_type}")
    else:
        st.write("📄 Basic molecular data only")


def main():
    """Main Export interface."""
    # Add logo to app and sidebar
    st.logo(
        image="media/pickm8_white.png",
        size="large",
        icon_image="media/pickm8_white_logoonly.png"
    )
    
    st.title("⬇️ Export Data")
    
    # Check session
    if not st.session_state.get('session_id'):
        st.error("No session loaded. Please go to the main page and load a session.")
        if st.button("🏠 Go to Main Page"):
            st.switch_page("main.py")
        return
    
    session_id = st.session_state.session_id
    
    # Load data
    result = sessions.load_session(st.session_state.session_id)
    if not result:
        st.error("No molecules loaded. Please upload data first.")
        return
    
    molecules_df, session_metadata = result
    session_name = session_metadata.get('protein_name', 'pickm8_session') if session_metadata else 'pickm8_session'
    
    # Remove file extension if present
    if session_name.endswith('.pdb'):
        session_name = session_name[:-4]
    
    # Display export summary
    display_export_summary(molecules_df)
    
    st.divider()
    
    # Export interface
    export_interface(molecules_df, session_id, session_name)


if __name__ == "__main__":
    main()