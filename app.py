import streamlit as st
import yaml
from pathlib import Path
import uuid

st.set_page_config(
    page_title="PickM8",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'molecules' not in st.session_state:
        st.session_state.molecules = None
    if 'current_mol_idx' not in st.session_state:
        st.session_state.current_mol_idx = 0
    if 'grades' not in st.session_state:
        st.session_state.grades = {}
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

def main():
    config = load_config()
    init_session_state()
    
    st.title("🧬 PickM8 - Active Learning for Molecular Screening")
    st.markdown("### Machine Learning-Guided Visual Inspection of Molecular Docking Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Session ID:** {st.session_state.session_id[:8]}...")
    
    with col2:
        if st.session_state.molecules:
            st.success(f"**Molecules Loaded:** {len(st.session_state.molecules)}")
        else:
            st.warning("**No molecules loaded**")
    
    with col3:
        st.metric("**Molecules Graded**", len(st.session_state.grades))
    
    st.markdown("---")
    
    st.markdown("""
    ## Getting Started
    
    1. **Upload Screen** - Upload your protein (PDB) and ligands (SDF) files
    2. **Active Learning** - Grade molecules and train models iteratively  
    3. **Results** - Export predictions and analysis
    
    Navigate using the sidebar menu.
    """)
    
    with st.expander("ℹ️ About PickM8"):
        st.markdown("""
        PickM8 is a streamlined tool for analyzing molecular docking results using active learning.
        It combines:
        - **LUNA** for interaction fingerprint calculation
        - **Machine Learning** for predicting molecule quality
        - **Active Learning** to prioritize molecules for review
        - **3D Visualization** for molecular interactions
        """)

if __name__ == "__main__":
    main()