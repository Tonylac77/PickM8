import streamlit as st
from utils.io_handlers import DataHandler, MoleculeReader
from core.interaction_functions import create_interaction_context, calculate_interactions_batch
from core.fingerprints import FingerprintHandler
import tempfile
import json
import traceback
import logging
from rdkit import Chem

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Upload Screen - PickM8", page_icon="📤", layout="wide")

def process_molecules_parallel(protein_path, molecules, interaction_context, fp_handler):
    """
    Process molecules in parallel for fingerprint and interaction calculations
    """
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
            
            # Store additional SDF properties
            if 'original_mol' in mol:
                props = mol['original_mol'].GetPropsAsDict()
                for prop_name, prop_value in props.items():
                    if prop_name not in ['_Name', mol.get('score_label', 'score')]:
                        mol[f'prop_{prop_name}'] = prop_value
            
            processed_molecules.append(mol)
            
        except Exception as e:
            logger.error(f"Error combining results for molecule {mol['name']}: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    status.text("Processing complete!")
    
    # Report any errors
    if all_errors:
        st.warning(f"⚠️ {len(all_errors)} errors occurred during processing. Check logs for details.")
        
        with st.expander(f"Processing errors ({len(all_errors)})"):
            for idx, error in all_errors.items():
                st.code(f"Molecule {idx}: {error}")
    
    logger.info(f"Parallel processing complete. Successfully processed {len(processed_molecules)}/{len(molecules)} molecules")
    return processed_molecules

def process_molecules_sequential(protein_path, molecules, interaction_context, fp_handler):
    """
    Fallback sequential processing for molecules when parallel processing fails
    """
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
            
            # Step 2: Calculate interactions
            from core.interaction_functions import calculate_with_context
            ifp, interactions = calculate_with_context(interaction_context, protein_path, mol_obj, mol['name'])
            
            # Step 3: Store interaction data
            if isinstance(ifp, dict):
                mol['ifp'] = json.dumps({str(k): int(v) for k, v in ifp.items()})
            else:
                mol['ifp'] = json.dumps({str(i): int(v) for i, v in enumerate(ifp) if v > 0})
            
            mol['interactions'] = json.dumps(interactions.get('interactions', []))
            mol['num_interactions'] = interactions.get('total_interactions', 0)
            
            # Step 4: Calculate fingerprints
            mol_fp = fp_handler.compute_fingerprint(mol_obj)
            mol['morgan_fp'] = mol_fp.tolist()
            
            # Step 5: Store additional SDF properties
            if 'original_mol' in mol:
                props = mol['original_mol'].GetPropsAsDict()
                for prop_name, prop_value in props.items():
                    if prop_name not in ['_Name', mol.get('score_label', 'score')]:
                        mol[f'prop_{prop_name}'] = prop_value
            
            processed_molecules.append(mol)
            
        except Exception as e:
            logger.error(f"Error processing molecule {mol['name']}: {str(e)}")
            st.error(f"❌ Failed to process {mol['name']}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(molecules))
    
    status.text("Processing complete!")
    logger.info(f"Sequential processing complete. Successfully processed {len(processed_molecules)}/{len(molecules)} molecules")
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
        
        # Initialize variables
        ligand_path = None
        score_label = "minimizedAffinity"  # Default value
        available_properties = []
        
        if ligand_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
                tmp.write(ligand_file.getvalue())
                ligand_path = tmp.name
            
            # Detect available properties in the SDF file
            with st.spinner("Detecting SDF properties..."):
                available_properties = MoleculeReader.get_sdf_properties(ligand_path)
            
            if available_properties:
                st.success(f"📋 Found {len(available_properties)} properties in SDF file")
                
                # Create dropdown for score selection
                default_index = 0
                if "minimizedAffinity" in available_properties:
                    default_index = available_properties.index("minimizedAffinity")
                elif any("score" in prop.lower() for prop in available_properties):
                    # Find first property containing "score"
                    for i, prop in enumerate(available_properties):
                        if "score" in prop.lower():
                            default_index = i
                            break
                elif any("affinity" in prop.lower() for prop in available_properties):
                    # Find first property containing "affinity"
                    for i, prop in enumerate(available_properties):
                        if "affinity" in prop.lower():
                            default_index = i
                            break
                
                score_label = st.selectbox(
                    "Select docking score property:",
                    options=available_properties,
                    index=default_index,
                    help="Choose which property to use as the docking score for active learning"
                )
                
                # Show preview of available properties
                with st.expander("📊 All Available Properties"):
                    for prop in available_properties:
                        st.text(f"• {prop}")
            else:
                st.warning("⚠️ Could not detect properties in SDF file. Using manual input.")
                score_label = st.text_input("Score label in SDF", value="minimizedAffinity")
    
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
    
    if st.button("Process Molecules", type="primary", disabled=not (protein_file and ligand_file and ligand_path)):
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
                interaction_context = create_interaction_context(ifp_type=interaction_fp_type)
                logger.debug(f"Wrappers initialized successfully with {interaction_fp_type} interaction fingerprints and {molecular_fp_type} molecular fingerprints")
                
                st.subheader("Processing Interactions")
                logger.info("Starting molecule processing with interactions")
                
                st.info(f"🧬 Using **{interaction_fp_type}** interaction fingerprints and **{molecular_fp_type}** molecular fingerprints")
                
                # Try parallel processing first
                try:
                    processed_molecules = process_molecules_parallel(
                        st.session_state.protein_path, 
                        molecules, 
                        interaction_context, 
                        fp_handler
                    )
                except Exception as e:
                    logger.error(f"Parallel processing failed: {str(e)}")
                    st.warning("⚠️ Parallel processing failed, falling back to sequential processing...")
                    processed_molecules = process_molecules_sequential(
                        st.session_state.protein_path, 
                        molecules, 
                        interaction_context, 
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