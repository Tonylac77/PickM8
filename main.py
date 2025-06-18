"""
PickM8 Main Application
Active Learning for Molecular Screening using functional data processing approach.
"""
import json
import logging
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from core.fingerprints import (
    create_default_fingerprint_config,
    create_default_interaction_config,
)
from core.pose_analysis import (
    compute_pose_quality_batch,
    create_default_posecheck_config,
)

# Import new functional utilities
from utils.data_processing import (
    load_molecules_dataframe,
    load_pdb_file,
    load_sdf_file,
    load_session_metadata,
    save_molecules_dataframe,
    save_session_metadata,
)
from utils.processing import compute_fingerprints_batch, get_fingerprint_statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PickM8 - Active Learning for Molecular Screening",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_config():
    """Load application configuration."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file not found
        return {
            'data': {'sessions_dir': 'data/sessions'},
            'fingerprints': {'morgan_radius': 2, 'morgan_bits': 2048},
            'interactions': {'type': 'plip'}
        }


def get_existing_sessions():
    """Get list of existing sessions with metadata."""
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            try:
                # Load session metadata
                metadata = load_session_metadata(str(session_dir))
                if metadata is None:
                    continue
                
                # Load molecules to get counts
                molecules_df = load_molecules_dataframe(str(session_dir))
                
                num_molecules = len(molecules_df) if molecules_df is not None else 0
                num_grades = 0
                
                if molecules_df is not None and 'grade' in molecules_df.columns:
                    num_grades = molecules_df['grade'].notna().sum()
                
                # Get last modified time
                molecule_file = session_dir / "molecules.pkl"
                last_modified = datetime.fromtimestamp(
                    molecule_file.stat().st_mtime if molecule_file.exists() 
                    else session_dir.stat().st_mtime
                )
                
                sessions.append({
                    'session_id': session_dir.name,
                    'session_id_short': session_dir.name[:8],
                    'protein_name': metadata.get('protein_name', 'Unknown'),
                    'num_molecules': num_molecules,
                    'num_grades': num_grades,
                    'last_modified': last_modified,
                    'score_label': metadata.get('score_label', 'score'),
                    'created_date': metadata.get('created_date', 'Unknown')
                })
                
            except Exception as e:
                logger.warning(f"Could not read session {session_dir.name}: {str(e)}")
                continue
    
    # Sort by last modified (newest first)
    sessions.sort(key=lambda x: x['last_modified'], reverse=True)
    return sessions


def process_molecules_pipeline(df, protein_content, fp_config, interaction_config, pose_config):
    """Process molecules through the complete pipeline."""
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Step 1: Compute fingerprints
    status.text("Computing molecular and interaction fingerprints...")
    df = compute_fingerprints_batch(
        df, protein_content, fp_config, interaction_config
    )
    progress_bar.progress(0.4)
    
    # Step 2: Compute pose quality metrics
    status.text("Analyzing pose quality...")
    df = compute_pose_quality_batch(df, protein_content, pose_config)
    progress_bar.progress(0.8)
    
    # Step 3: Final cleanup
    status.text("Finalizing processing...")
    progress_bar.progress(1.0)
    
    # Show statistics
    fp_stats = get_fingerprint_statistics(df)
    status.text("Processing complete!")
    
    st.success(f"✅ Successfully processed {len(df)} molecules")
    
    # Display processing statistics
    with st.expander("📊 Processing Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Molecules Processed", fp_stats['total_molecules'])
            st.metric("Morgan FP", f"{fp_stats['morgan_fp_percentage']:.1f}%")
        
        with col2:
            st.metric("Interaction FP", f"{fp_stats['interaction_fp_percentage']:.1f}%")
            st.metric("Avg Interactions", f"{fp_stats['avg_interactions_per_molecule']:.1f}")
        
        with col3:
            st.metric("Molecules with Interactions", fp_stats['molecules_with_interactions'])
            st.metric("Max Interactions", fp_stats['max_interactions'])
    
    return df


def detect_sdf_properties(sdf_path):
    """Detect available properties in SDF file."""
    try:
        # Load just first few molecules to detect properties
        import pandas as pd
        from rdkit.Chem import PandasTools
        
        temp_df = PandasTools.LoadSDF(sdf_path, molColName='mol')
        if len(temp_df) == 0:
            return []
        
        # Get column names excluding RDKit columns
        properties = [col for col in temp_df.columns 
                     if col not in ['mol', 'ID'] and not col.startswith('_')]
        
        return properties
        
    except Exception as e:
        logger.error(f"Error detecting SDF properties: {e}")
        return []


def upload_new_session():
    """Handle uploading new PDB and SDF files for a new session."""
    st.subheader("🆕 Create New Session")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Upload Protein Structure")
        protein_file = st.file_uploader("Select PDB file", type=['pdb'], key="new_protein")
        
        protein_content = None
        if protein_file:
            protein_content = protein_file.getvalue().decode('utf-8')
            st.success(f"✅ Loaded protein: {protein_file.name}")
    
    with col2:
        st.markdown("#### 2. Upload Ligands")
        ligand_file = st.file_uploader("Select SDF file", type=['sdf', 'gz'], key="new_ligands")
        
        st.info("📋 **Score Requirements:**\n"
                "• All score values must be numeric\n" 
                "• No missing scores allowed\n"
                "• Choose direction preference below")
        
        ligand_path = None
        score_label = "score"
        available_properties = []
        
        if ligand_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
                tmp.write(ligand_file.getvalue())
                ligand_path = tmp.name
            
            # Detect available properties in the SDF file
            with st.spinner("Detecting SDF properties..."):
                available_properties = detect_sdf_properties(ligand_path)
            
            if available_properties:
                st.success(f"📋 Found {len(available_properties)} properties")
                
                # Create dropdown for score selection
                default_index = 0
                score_candidates = ["minimizedAffinity", "score", "Score", "docking_score", "binding_affinity"]
                
                for candidate in score_candidates:
                    if candidate in available_properties:
                        default_index = available_properties.index(candidate)
                        break
                
                score_label = st.selectbox(
                    "Select docking score property:",
                    options=available_properties,
                    index=default_index,
                    help="Choose which property to use as the docking score"
                )
                
                # Add score direction toggle
                col_score1, col_score2 = st.columns(2)
                with col_score1:
                    score_direction = st.selectbox(
                        "Score interpretation:",
                        options=["Lower is better", "Higher is better"],
                        index=0,
                        help="How should scores be interpreted for ranking?"
                    )
                with col_score2:
                    st.info(f"Selected: **{score_label}**")
                
            else:
                st.warning("⚠️ Could not detect properties. Manual input required.")
                col_manual1, col_manual2 = st.columns(2)
                with col_manual1:
                    score_label = st.text_input("Score property name", value="score")
                with col_manual2:
                    score_direction = st.selectbox(
                        "Score interpretation:",
                        options=["Lower is better", "Higher is better"],
                        index=0
                    )
    
    # Configuration Section
    st.markdown("#### 3. Processing Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        interaction_type = st.selectbox(
            "Interaction Analysis",
            options=["plip", "prolif"],
            index=0,
            help="PLIP: Fast protein-ligand interaction analysis\nProLIF: Comprehensive interaction fingerprints"
        )
    
    with col4:
        compute_pose_quality = st.checkbox(
            "Compute Pose Quality Metrics",
            value=True,
            help="Calculate clash detection and strain energy (requires PoseCheck)"
        )
    
    # Process button
    if st.button("🚀 Create Session & Process", 
                type="primary", 
                disabled=not (protein_file and ligand_file and ligand_path)):
        
        try:
            # Create new session ID
            new_session_id = str(uuid.uuid4())
            session_dir = f"data/sessions/{new_session_id}"
            
            st.session_state.session_id = new_session_id
            
            with st.spinner("Loading molecules from SDF..."):
                # Load molecules using new functional approach
                molecules_df = load_sdf_file(ligand_path)
                
                # Validate and set score column
                if score_label in molecules_df.columns:
                    score_values = molecules_df[score_label]
                    
                    # Check if all values are numeric
                    try:
                        numeric_scores = pd.to_numeric(score_values, errors='raise')
                        molecules_df['score'] = numeric_scores
                        
                        # Score direction is stored in metadata for selection logic
                        # Do not modify score values - selection logic will handle direction
                        
                        st.success(f"✅ Loaded {len(molecules_df)} molecules with valid scores")
                        st.info(f"Score range: {molecules_df['score'].min():.3f} to {molecules_df['score'].max():.3f}")
                        
                    except (ValueError, TypeError) as e:
                        st.error(f"❌ Score column '{score_label}' contains non-numeric values!")
                        st.error("All scores must be numeric. Please check your SDF file.")
                        return
                        
                else:
                    st.error(f"❌ Score column '{score_label}' not found in SDF file!")
                    st.error("Available columns: " + ", ".join(molecules_df.columns.tolist()))
                    return
            
            if len(molecules_df) > 0:
                # Create processing configurations
                fp_config = create_default_fingerprint_config()
                
                interaction_config = create_default_interaction_config()
                interaction_config['interaction_type'] = interaction_type
                
                pose_config = create_default_posecheck_config()
                pose_config['calculate_clashes'] = compute_pose_quality
                pose_config['calculate_strain'] = compute_pose_quality
                
                st.subheader("Processing Molecules")
                st.info(f"🧬 Using **{interaction_type.upper()}** interaction analysis")
                
                # Process molecules through complete pipeline
                processed_df = process_molecules_pipeline(
                    molecules_df, protein_content, fp_config, interaction_config, pose_config
                )
                
                # Save processed data
                save_molecules_dataframe(processed_df, session_dir)
                
                # Save session metadata
                session_metadata = {
                    'protein_name': protein_file.name,
                    'protein_content': protein_content,
                    'num_molecules': len(processed_df),
                    'score_label': score_label,
                    'score_direction': score_direction,
                    'created_date': datetime.now().isoformat(),
                    'interaction_type': interaction_type,
                    'compute_pose_quality': compute_pose_quality,
                    'available_properties': available_properties
                }
                save_session_metadata(session_dir, session_metadata)
                
                # Store in session state for immediate use
                st.session_state.molecules_df = processed_df
                st.session_state.protein_content = protein_content
                
                st.success(f"✅ Successfully created session and processed {len(processed_df)} molecules!")
                st.balloons()
                
                # Navigation
                col_nav1, col_nav2 = st.columns(2)
                with col_nav1:
                    if st.button("🎯 Start Active Learning", type="primary"):
                        st.switch_page("pages/3_🎯_Active_Learning.py")
                
                with col_nav2:
                    if st.button("📊 View Results", type="secondary"):
                        st.switch_page("pages/4_📊_Results.py")
            
        except Exception as e:
            st.error(f"❌ Error creating session: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            logger.error(f"Session creation error: {e}", exc_info=True)


def load_existing_session():
    """Handle loading an existing session."""
    st.subheader("📂 Load Existing Session")
    
    sessions = get_existing_sessions()
    
    if not sessions:
        st.info("No existing sessions found. Create a new session to get started.")
        return
    
    st.markdown(f"Found **{len(sessions)}** existing sessions:")
    
    # Display sessions in a nice format
    for i, session in enumerate(sessions):
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.markdown(f"""
                **{session['protein_name']}**  
                *Session: {session['session_id_short']}...*  
                *Created: {session['created_date'][:10] if session['created_date'] != 'Unknown' else 'Unknown'}*
                *Last modified: {session['last_modified'].strftime('%Y-%m-%d %H:%M')}*
                """)
            
            with col2:
                st.metric("Molecules", session['num_molecules'])
            
            with col3:
                st.metric("Graded", session['num_grades'])
                if session['num_molecules'] > 0:
                    progress = session['num_grades'] / session['num_molecules']
                    st.progress(progress)
            
            with col4:
                if st.button(f"Load Session", key=f"load_{session['session_id']}", type="primary"):
                    # Load this session
                    st.session_state.session_id = session['session_id']
                    session_dir = f"data/sessions/{session['session_id']}"
                    
                    # Load molecules DataFrame
                    molecules_df = load_molecules_dataframe(session_dir)
                    if molecules_df is not None:
                        st.session_state.molecules_df = molecules_df
                    
                    # Load session metadata
                    metadata = load_session_metadata(session_dir)
                    if metadata:
                        st.session_state.protein_content = metadata.get('protein_content', '')
                    
                    st.success(f"✅ Loaded session: {session['protein_name']}")
                    st.info("Navigate to Active Learning to continue grading molecules.")
                    
                    # Add navigation buttons
                    nav_col1, nav_col2 = st.columns(2)
                    with nav_col1:
                        if st.button("🎯 Active Learning", type="primary", key=f"nav_al_{session['session_id']}"):
                            st.switch_page("pages/3_🎯_Active_Learning.py")
                    with nav_col2:
                        if st.button("📊 View Results", type="secondary", key=f"nav_results_{session['session_id']}"):
                            st.switch_page("pages/4_📊_Results.py")
                    
                    st.stop()
            
            st.divider()


def main():
    """Main application entry point."""
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'molecules_df' not in st.session_state:
        st.session_state.molecules_df = None
    if 'protein_content' not in st.session_state:
        st.session_state.protein_content = None
    
    st.title("🧬 PickM8 - Active Learning for Molecular Screening")
    st.markdown("### Machine Learning-Guided Visual Inspection of Molecular Docking Results")
    
    # Main navigation tabs
    tab1, tab2 = st.tabs(["🆕 New Session", "📂 Load Session"])
    
    with tab1:
        upload_new_session()
    
    with tab2:
        load_existing_session()

if __name__ == "__main__":
    main()