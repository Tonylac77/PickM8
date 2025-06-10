import streamlit as st
from utils.io_handlers import DataHandler, MoleculeReader
from core.interaction_wrapper import InteractionWrapper
from core.fingerprints import FingerprintHandler
import tempfile
from pathlib import Path
import json
import traceback
import logging
from rdkit import Chem

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Upload Screen - PickM8", page_icon="📤", layout="wide")

def process_molecules(protein_path, molecules, interaction_wrapper, fp_handler):
    progress_bar = st.progress(0)
    status = st.empty()
    
    processed_molecules = []
    
    for i, mol in enumerate(molecules):
        status.text(f"Processing molecule {i+1}/{len(molecules)}: {mol['name']}")
        
        try:
            logger.info(f"Processing molecule {i+1}/{len(molecules)}: {mol['name']}")
            
            # Step 1: Create molecule object
            mol_obj = Chem.MolFromMolBlock(mol['mol_block'])
            if mol_obj is None:
                raise ValueError(f"Failed to create molecule object from mol_block for {mol['name']}")
            logger.debug(f"Successfully created molecule object for {mol['name']}")
            
            # Step 2: Calculate interactions
            logger.debug(f"Calculating interactions for {mol['name']}")
            ifp, interactions = interaction_wrapper.calculate_interactions(protein_path, mol_obj, mol['name'])
            logger.debug(f"Interactions calculated for {mol['name']}: {interactions.get('total_interactions', 0) if interactions else 0} interactions")
            
            # Step 3: Store interaction data
            if isinstance(ifp, dict):
                mol['ifp'] = json.dumps({str(k): int(v) for k, v in ifp.items()})
            else:
                # Handle numpy array from ProLIF or binary array from PLIP
                mol['ifp'] = json.dumps({str(i): int(v) for i, v in enumerate(ifp) if v > 0})
            
            # Store interaction summary
            mol['interactions'] = json.dumps(interactions.get('interactions', []))
            mol['num_interactions'] = interactions.get('total_interactions', 0)
            logger.debug(f"Interaction data stored for {mol['name']}")
            
            # Step 4: Calculate fingerprints
            logger.debug(f"Computing fingerprint for {mol['name']}")
            mol_fp = fp_handler.compute_fingerprint(mol_obj)
            mol['morgan_fp'] = mol_fp.tolist()
            logger.debug(f"Fingerprint computed for {mol['name']}")
            
            # Step 5: Store additional SDF properties
            # Extract any additional properties from the original molecule
            if 'original_mol' in mol:  # If we stored the original RDKit mol
                props = mol['original_mol'].GetPropsAsDict()
                for prop_name, prop_value in props.items():
                    if prop_name not in ['_Name', mol.get('score_label', 'score')]:
                        mol[f'prop_{prop_name}'] = prop_value
            
            processed_molecules.append(mol)
            logger.info(f"Successfully processed molecule {mol['name']}")
            
        except Exception as e:
            error_msg = f"Failed to process {mol['name']}: {str(e)}"
            full_traceback = traceback.format_exc()
            
            logger.error(f"Error processing molecule {mol['name']}")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback:\n{full_traceback}")
            
            # Display detailed error in Streamlit
            st.error(f"❌ {error_msg}")
            with st.expander(f"Error details for {mol['name']}"):
                st.code(full_traceback)
                st.write("**Molecule data:**")
                st.json({
                    'name': mol.get('name', 'Unknown'),
                    'mol_block_length': len(mol.get('mol_block', '')) if mol.get('mol_block') else 0,
                    'score': mol.get('score', 'N/A'),
                    'keys': list(mol.keys()) if mol else []
                })
        
        progress_bar.progress((i + 1) / len(molecules))
    
    status.text("Processing complete!")
    logger.info(f"Processing complete. Successfully processed {len(processed_molecules)}/{len(molecules)} molecules")
    return processed_molecules

def main():
    st.title("📤 Upload Molecular Screen")
    
    if 'session_id' not in st.session_state:
        st.error("No active session. Please start from the main page.")
        return
    
    data_handler = DataHandler(st.session_state.session_id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Protein Structure")
        protein_file = st.file_uploader("Select PDB file", type=['pdb'])
        
        if protein_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp:
                tmp.write(protein_file.getvalue())
                protein_path = tmp.name
                st.session_state.protein_path = protein_path
                st.success(f"Loaded protein: {protein_file.name}")
    
    with col2:
        st.subheader("2. Upload Ligands")
        ligand_file = st.file_uploader("Select SDF file", type=['sdf', 'gz'])
        score_label = st.text_input("Score label in SDF", value="minimizedAffinity")
        
        if ligand_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
                tmp.write(ligand_file.getvalue())
                ligand_path = tmp.name
    
    # Fingerprint Configuration Section
    st.subheader("3. Fingerprint Configuration")
    
    st.info("💡 **Fingerprint Selection Tips:**\n"
           "• **PLIP**: Faster processing, good for quick analysis with basic interaction types\n"
           "• **ProLIF**: More comprehensive analysis, slower but captures more interaction details\n"
           "• **Morgan**: Standard circular fingerprints, good general performance\n"
           "• **RDKit**: Topological fingerprints, captures different molecular features")
    
    col3, col4 = st.columns(2)
    
    with col3:
        interaction_fp_type = st.selectbox(
            "Interaction Fingerprint Type",
            options=["PLIP", "PROLIF"],
            index=0,
            help="PLIP: Fast protein-ligand interaction profiler\nProLIF: Comprehensive interaction fingerprints using MDAnalysis"
        )
        
        # Show available interaction types for selected method
        if interaction_fp_type == "PLIP":
            st.caption("📋 PLIP detects: Hydrogen bonds, Hydrophobic contacts, π-stacking, Salt bridges, Halogen bonds")
        else:
            st.caption("📋 ProLIF detects: HB Acceptor/Donor, Hydrophobic, π-stacking, Ionic interactions, Cation-π, and more")
    
    with col4:
        molecular_fp_type = st.selectbox(
            "Molecular Fingerprint Type", 
            options=["morgan", "rdkit"],
            index=0,
            help="Morgan: Circular fingerprints based on atom environments\nRDKit: Topological fingerprints"
        )
        
        # Show fingerprint details
        if molecular_fp_type == "morgan":
            st.caption("🔬 Morgan fingerprints: Radius-based circular patterns, 2048 bits")
        else:
            st.caption("🔬 RDKit fingerprints: Topological path-based patterns, 2048 bits")
    
    if st.button("Process Molecules", type="primary", disabled=not (protein_file and ligand_file)):
        try:
            logger.info("Starting molecule processing")
            
            with st.spinner("Loading molecules..."):
                logger.debug(f"Reading SDF file with score label: {score_label}")
                molecules = MoleculeReader.read_sdf(ligand_path, score_label)
                logger.info(f"Loaded {len(molecules)} molecules from SDF")
                st.success(f"Loaded {len(molecules)} molecules")
            
            if molecules:
                logger.debug("Initializing interaction wrapper and fingerprint handler")
                # Create fingerprint handler with user-selected options
                fp_handler = FingerprintHandler(
                    fp_type=molecular_fp_type,
                    interaction_fp_type=interaction_fp_type
                )
                interaction_wrapper = InteractionWrapper(ifp_type=interaction_fp_type)
                logger.debug(f"Wrappers initialized successfully with {interaction_fp_type} interaction fingerprints and {molecular_fp_type} molecular fingerprints")
                
                st.subheader("Processing Interactions")
                logger.info("Starting molecule processing with interactions")
                
                st.info(f"🧬 Using **{interaction_fp_type}** interaction fingerprints and **{molecular_fp_type}** molecular fingerprints")
                
                processed_molecules = process_molecules(
                    st.session_state.protein_path, 
                    molecules, 
                    interaction_wrapper, 
                    fp_handler
                )
                
                logger.info(f"Processed {len(processed_molecules)} molecules successfully")
                
                # Save processed data
                logger.debug("Saving processed molecules")
                data_handler.save_molecules(processed_molecules)
                
                logger.debug("Reading protein content")
                protein_content = MoleculeReader.read_pdb(st.session_state.protein_path)
                
                session_data = {
                    'protein_name': protein_file.name,
                    'protein_content': protein_content,
                    'num_molecules': len(processed_molecules),
                    'score_label': score_label
                }
                logger.debug("Saving session state")
                data_handler.save_session_state(session_data)
                
                st.session_state.molecules = processed_molecules
                logger.info("Processing completed successfully")
                
                st.success(f"✅ Successfully processed {len(processed_molecules)} molecules!")
                
            else:
                st.error("No molecules were loaded from the SDF file")
                logger.error("No molecules loaded from SDF file")
                
        except Exception as e:
            error_msg = f"Critical error during molecule processing: {str(e)}"
            full_traceback = traceback.format_exc()
            
            logger.error("Critical error in main processing")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Full traceback:\n{full_traceback}")
            
            st.error(f"❌ {error_msg}")
            with st.expander("Error Details"):
                st.code(full_traceback)
                st.write("**Debug Information:**")
                st.json({
                    'session_id': st.session_state.get('session_id', 'Not found'),
                    'protein_path': st.session_state.get('protein_path', 'Not found'),
                    'protein_file_name': protein_file.name if protein_file else 'No file',
                    'ligand_file_name': ligand_file.name if ligand_file else 'No file',
                    'score_label': score_label
                })
        
        if molecules and len(processed_molecules) > 0:
            st.balloons()
            
            if st.button("Proceed to Active Learning", type="primary"):
                st.switch_page("pages/3_🎯_Active_Learning.py")

if __name__ == "__main__":
    main()