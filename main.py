import streamlit as st
import yaml
from pathlib import Path
import uuid
import json
from datetime import datetime
from utils.io_handlers import DataHandler, MoleculeReader
from core.interaction_functions import create_interaction_context, calculate_interactions_batch
from core.fingerprints import FingerprintHandler
import tempfile
import traceback
import logging
from rdkit import Chem

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="PickM8 - Active Learning for Molecular Screening",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_data
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_existing_sessions():
    """Get list of existing sessions with metadata"""
    sessions_dir = Path("data/sessions")
    if not sessions_dir.exists():
        return []
    
    sessions = []
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            try:
                session_state_file = session_dir / "session_state.json"
                molecules_file = session_dir / "molecules.parquet"
                grades_file = session_dir / "grades.parquet"
                
                if session_state_file.exists():
                    with open(session_state_file, 'r') as f:
                        session_data = json.load(f)
                    
                    # Get file modification times
                    last_modified = max(
                        session_state_file.stat().st_mtime,
                        molecules_file.stat().st_mtime if molecules_file.exists() else 0,
                        grades_file.stat().st_mtime if grades_file.exists() else 0
                    )
                    
                    # Count molecules and grades
                    num_molecules = 0
                    num_grades = 0
                    
                    if molecules_file.exists():
                        import polars as pl
                        molecules_df = pl.read_parquet(molecules_file)
                        num_molecules = len(molecules_df)
                    
                    if grades_file.exists():
                        import polars as pl
                        grades_df = pl.read_parquet(grades_file)
                        num_grades = len(grades_df)
                    
                    sessions.append({
                        'session_id': session_dir.name,
                        'session_id_short': session_dir.name[:8],
                        'protein_name': session_data.get('protein_name', 'Unknown'),
                        'num_molecules': num_molecules,
                        'num_grades': num_grades,
                        'last_modified': datetime.fromtimestamp(last_modified),
                        'score_label': session_data.get('score_label', 'score')
                    })
            except Exception as e:
                logger.warning(f"Could not read session {session_dir.name}: {str(e)}")
                continue
    
    # Sort by last modified (newest first)
    sessions.sort(key=lambda x: x['last_modified'], reverse=True)
    return sessions

def process_molecules_parallel(protein_path, molecules, interaction_context, fp_handler):
    """Process molecules in parallel for fingerprint and interaction calculations"""
    progress_bar = st.progress(0)
    status = st.empty()
    
    status.text("Preparing molecules for parallel processing...")
    
    # Step 1: Prepare molecule objects
    mol_objects = []
    valid_molecules = []
    
    for i, mol in enumerate(molecules):
        try:
            mol_obj = Chem.MolFromMolBlock(mol['mol_block'])
            if mol_obj is None:
                logger.error(f"Failed to create molecule object for {mol['name']}")
                continue
            mol_objects.append(mol_obj)
            valid_molecules.append(mol)
        except Exception as e:
            logger.error(f"Error preparing molecule {mol['name']}: {str(e)}")
            continue
    
    if not valid_molecules:
        st.error("No valid molecules could be processed")
        return []
    
    logger.info(f"Successfully prepared {len(valid_molecules)} out of {len(molecules)} molecules")
    progress_bar.progress(0.1)
    
    # Step 2: Calculate fingerprints and interactions simultaneously
    status.text(f"Processing {len(valid_molecules)} molecules in parallel...")
    
    try:
        fingerprints, ifp_results, interaction_results, all_errors = fp_handler.process_molecules_batch(
            mol_objects, protein_path, interaction_context
        )
        logger.info(f"Parallel processing complete with {len(all_errors)} total errors")
    except Exception as e:
        st.error(f"Error during parallel processing: {str(e)}")
        return []
    
    progress_bar.progress(0.8)
    
    # Step 3: Combine all results
    status.text("Combining results...")
    processed_molecules = []
    
    for i, mol in enumerate(valid_molecules):
        try:
            # Store fingerprint data
            mol['morgan_fp'] = fingerprints[i].tolist()
            
            # Store interaction data
            ifp = ifp_results[i]
            interactions = interaction_results[i]
            
            if isinstance(ifp, dict):
                mol['ifp'] = json.dumps({str(k): int(v) for k, v in ifp.items()})
            else:
                # Handle numpy array
                mol['ifp'] = json.dumps({str(j): int(v) for j, v in enumerate(ifp) if v > 0})
            
            mol['interactions'] = json.dumps(interactions.get('interactions', []))
            mol['num_interactions'] = interactions.get('total_interactions', 0)
            
            processed_molecules.append(mol)
            
        except Exception as e:
            logger.error(f"Error combining results for molecule {mol['name']}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status.text("Processing complete!")
    
    # Report any errors
    if all_errors:
        st.warning(f"⚠️ {len(all_errors)} errors occurred during processing. Check logs for details.")
    
    logger.info(f"Parallel processing complete. Successfully processed {len(processed_molecules)}/{len(molecules)} molecules")
    return processed_molecules

def upload_new_session():
    """Handle uploading new PDB and SDF files for a new session"""
    st.subheader("🆕 Create New Session")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Upload Protein Structure")
        protein_file = st.file_uploader("Select PDB file", type=['pdb'], key="new_protein")
        
        if protein_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp:
                tmp.write(protein_file.getvalue())
                protein_path = tmp.name
                st.session_state.protein_path = protein_path
                st.success(f"✅ Loaded protein: {protein_file.name}")
    
    with col2:
        st.markdown("#### 2. Upload Ligands")
        ligand_file = st.file_uploader("Select SDF file", type=['sdf', 'gz'], key="new_ligands")
        
        ligand_path = None
        score_label = "minimizedAffinity"
        available_properties = []
        
        if ligand_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
                tmp.write(ligand_file.getvalue())
                ligand_path = tmp.name
            
            # Detect available properties in the SDF file
            with st.spinner("Detecting SDF properties..."):
                available_properties = MoleculeReader.get_sdf_properties(ligand_path)
            
            if available_properties:
                st.success(f"📋 Found {len(available_properties)} properties")
                
                # Create dropdown for score selection
                default_index = 0
                if "minimizedAffinity" in available_properties:
                    default_index = available_properties.index("minimizedAffinity")
                elif any("score" in prop.lower() for prop in available_properties):
                    for i, prop in enumerate(available_properties):
                        if "score" in prop.lower():
                            default_index = i
                            break
                
                score_label = st.selectbox(
                    "Select docking score property:",
                    options=available_properties,
                    index=default_index,
                    help="Choose which property to use as the docking score"
                )
            else:
                st.warning("⚠️ Could not detect properties. Using manual input.")
                score_label = st.text_input("Score label in SDF", value="minimizedAffinity")
    
    # Fingerprint Configuration
    st.markdown("#### 3. Fingerprint Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        interaction_fp_type = st.selectbox(
            "Interaction Fingerprint",
            options=["PLIP", "PROLIF"],
            index=0,
            help="PLIP: Fast analysis\nProLIF: Comprehensive analysis"
        )
    
    with col4:
        molecular_fp_type = st.selectbox(
            "Molecular Fingerprint", 
            options=["morgan", "rdkit"],
            index=0,
            help="Morgan: Circular fingerprints\nRDKit: Topological fingerprints"
        )
    
    # Process button
    if st.button("🚀 Create Session & Process", type="primary", disabled=not (protein_file and ligand_file and ligand_path)):
        try:
            # Create new session ID
            new_session_id = str(uuid.uuid4())
            st.session_state.session_id = new_session_id
            
            data_handler = DataHandler(new_session_id)
            
            with st.spinner("Loading molecules..."):
                molecules = MoleculeReader.read_sdf(ligand_path, score_label)
                st.success(f"Loaded {len(molecules)} molecules")
            
            if molecules:
                # Create fingerprint handler
                fp_handler = FingerprintHandler(
                    fp_type=molecular_fp_type,
                    interaction_fp_type=interaction_fp_type
                )
                interaction_context = create_interaction_context(ifp_type=interaction_fp_type)
                
                st.subheader("Processing Molecules")
                st.info(f"🧬 Using **{interaction_fp_type}** interaction fingerprints and **{molecular_fp_type}** molecular fingerprints")
                
                # Process molecules
                processed_molecules = process_molecules_parallel(
                    st.session_state.protein_path, 
                    molecules, 
                    interaction_context, 
                    fp_handler
                )
                
                # Save processed data
                data_handler.save_molecules(processed_molecules)
                
                protein_content = MoleculeReader.read_pdb(st.session_state.protein_path)
                
                session_data = {
                    'protein_name': protein_file.name,
                    'protein_content': protein_content,
                    'num_molecules': len(processed_molecules),
                    'score_label': score_label,
                    'created_date': datetime.now().isoformat(),
                    'interaction_fp_type': interaction_fp_type,
                    'molecular_fp_type': molecular_fp_type
                }
                data_handler.save_session_state(session_data)
                
                st.session_state.molecules = processed_molecules
                
                st.success(f"✅ Successfully created session and processed {len(processed_molecules)} molecules!")
                st.balloons()
                
                if st.button("Start Active Learning", type="primary"):
                    st.switch_page("pages/3_🎯_Active_Learning.py")
            
        except Exception as e:
            st.error(f"❌ Error creating session: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

def load_existing_session():
    """Handle loading an existing session"""
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
                *Last modified: {session['last_modified'].strftime('%Y-%m-%d %H:%M')}*
                """)
            
            with col2:
                st.metric("Molecules", session['num_molecules'])
            
            with col3:
                st.metric("Graded", session['num_grades'])
                progress = session['num_grades'] / session['num_molecules'] if session['num_molecules'] > 0 else 0
                st.progress(progress)
            
            with col4:
                if st.button(f"Load Session", key=f"load_{session['session_id']}", type="primary"):
                    # Load this session
                    st.session_state.session_id = session['session_id']
                    
                    # Load session data
                    data_handler = DataHandler(session['session_id'])
                    molecules_df = data_handler.load_molecules()
                    if molecules_df is not None:
                        st.session_state.molecules = molecules_df.to_dicts()
                    
                    st.success(f"✅ Loaded session: {session['protein_name']}")
                    st.info("Navigate to Active Learning to continue grading molecules.")
                    
                    # Add navigation buttons
                    nav_col1, nav_col2 = st.columns(2)
                    with nav_col1:
                        if st.button("🎯 Active Learning", type="primary"):
                            st.switch_page("pages/3_🎯_Active_Learning.py")
                    with nav_col2:
                        if st.button("📊 View Results", type="secondary"):
                            st.switch_page("pages/4_📊_Results.py")
                    
                    st.stop()
            
            st.divider()

def main():
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    
    st.title("🧬 PickM8 - Active Learning for Molecular Screening")
    st.markdown("### Machine Learning-Guided Visual Inspection of Molecular Docking Results")
    
    st.markdown("---")
    
    # Main navigation tabs
    tab1, tab2 = st.tabs(["🆕 New Session", "📂 Load Session"])
    
    with tab1:
        upload_new_session()
    
    with tab2:
        load_existing_session()
    
    # Footer with app info
    st.markdown("---")
    with st.expander("ℹ️ About PickM8"):
        st.markdown("""
        **PickM8** is a streamlined tool for analyzing molecular docking results using active learning.
        
        **Features:**
        - 🧪 **Interaction Analysis** - PLIP and ProLIF integration for protein-ligand interactions
        - 🤖 **Machine Learning** - Uncertainty-based molecular prioritization
        - 🎯 **Active Learning** - Iterative model training with human feedback
        - 📊 **3D Visualization** - Interactive molecular visualization with Molstar
        - 📈 **Results Export** - Comprehensive analysis and prediction export
        
        **Workflow:**
        1. Upload protein (PDB) and ligands (SDF) files
        2. Configure fingerprint settings (PLIP/ProLIF + Morgan/RDKit)
        3. Grade molecules using the active learning interface
        4. Train models and export results
        """)

if __name__ == "__main__":
    main()