"""PickM8 Setup Page - Session Creation and Loading"""
import streamlit as st
import logging
from pathlib import Path

# Import from new flat structure
from data import sessions, molecules, fingerprints, interactions
from analysis import grading, pose_quality
from ui_components import forms, progress_displays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Setup - PickM8", 
    page_icon="media/pickm8_white_logoonly.png",
    layout="wide"
)

def main():
    """Setup page entry point."""
    # Add logo to app and sidebar
    st.logo(
        image="media/pickm8_white.png",
        size="large", 
        icon_image="media/pickm8_white_logoonly.png"
    )
    
    st.title("🔧 Setup - PickM8")

    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None

    # Main tabs
    tab1, tab2 = st.tabs(["🆕 New Session", "📂 Load Session"])

    with tab1:
        create_new_session()

    with tab2:
        load_existing_session()

def create_new_session():
    """Handle new session creation."""
    st.subheader("Create New Session")

    # Two-column layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("#### Data & Processing")
        
        # File upload
        protein_file, ligand_file = forms.render_file_upload_section()

        if ligand_file:
            # Detect score properties
            with st.spinner("Detecting properties..."):
                properties = molecules.detect_sdf_properties(ligand_file)

            score_label = st.selectbox(
                "Select score column",
                options=properties,
                index=0 if properties else None
            )

            score_direction = st.selectbox(
                "Score interpretation",
                options=["Lower is better", "Higher is better"],
                index=0
            )
        else:
            score_label = None
            score_direction = None

        # Processing options
        options = forms.render_processing_options()
    
    with right_col:
        st.markdown("#### Machine Learning")
        
        # ML model selection and configuration
        model_type, model_config = forms.render_ml_model_options()

    # Validation (full width)
    validation = forms.validate_session_inputs(
        protein_file, ligand_file, score_label, options['fingerprint_types']
    )

    # Show validation messages
    if validation['errors']:
        for error in validation['errors']:
            st.error(f"❌ {error}")

    if validation['warnings']:
        for warning in validation['warnings']:
            st.warning(f"⚠️ {warning}")

    # Process button
    if st.button("🚀 Create Session", disabled=not validation['valid']):
        with st.spinner("Processing molecules..."):
            # Create processing config
            config = {
                'fingerprint_config': {
                    'compute_morgan': 'morgan' in options['fingerprint_types'],
                    'compute_rdkit': 'rdkit' in options['fingerprint_types'],
                    'compute_mapchiral': 'mapchiral' in options['fingerprint_types']
                },
                'interaction_config': {
                    'type': options['interaction_type']
                },
                'pose_config': {
                    'enabled': options['compute_pose_quality']
                },
                'compute_interactions': True,
                'compute_pose_quality': options['compute_pose_quality'],
                'model_config': {
                    'model_type': model_type,
                    'model_params': model_config['model_params'],
                    'use_calibration': model_config['use_calibration']
                }
            }

            # Create session
            session_id, df, metadata = sessions.create_session(
                protein_file, ligand_file, score_label, score_direction, config
            )

            # Save session
            if sessions.save_session(session_id, df, metadata):
                st.success(f"✅ Session created successfully!")
                st.session_state.session_id = session_id
                st.session_state.molecules_df = df
                st.session_state.metadata = metadata

                # Show summary
                st.info(f"Processed {len(df)} molecules")

                # Navigation buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🎯 Go to Active Learning"):
                        st.switch_page("pages/3_🎯_Active_Learning.py")
                with col2:
                    if st.button("📊 View Results"):
                        st.switch_page("pages/4_📊_Results.py")
            else:
                st.error("Failed to save session")

def load_existing_session():
    """Handle loading existing sessions."""
    st.subheader("Load Existing Session")

    # Get available sessions
    available_sessions = sessions.list_sessions()

    if not available_sessions:
        st.info("No existing sessions found. Create a new session to get started.")
        return

    # Display sessions
    for session in available_sessions:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

        with col1:
            st.text(f"{session['protein_name']} ({session['session_id'][:8]}...)")
        with col2:
            st.text(f"{session['num_molecules']} molecules")
        with col3:
            st.text(session['created_date'][:10])
        with col4:
            if st.button("Load", key=f"load_{session['session_id']}"):
                # Load session data
                result = sessions.load_session(session['session_id'])
                if result:
                    df, metadata = result
                    st.session_state.session_id = session['session_id']
                    st.session_state.molecules_df = df
                    st.session_state.metadata = metadata
                    st.success("Session loaded successfully!")
                    st.switch_page("pages/3_🎯_Active_Learning.py")
                else:
                    st.error("Failed to load session")

if __name__ == "__main__":
    main()