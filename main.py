"""
PickM8 Main Application
Active Learning for Molecular Screening using functional data processing approach.
"""
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml

# Import new session service layer
from core.session import (
    create_new_session,
    load_session_data,
    reprocess_session_data,
    get_session_list,
    detect_sdf_properties,
    validate_uploaded_files,
    process_score_column,
    create_processing_configs,
    execute_processing_pipeline,
    save_session_data,
    generate_session_id,
    prepare_file_for_processing,
    find_default_score_property,
    create_processing_statistics_summary,
    create_and_save_session,
    execute_reprocessing_pipeline,
    validate_and_prepare_session_creation
)

# Import UI service layer
from core.ui import (
    validate_session_inputs,
    prepare_session_state_data,
    update_session_state,
    clear_session_selections,
    handle_creation_success,
    handle_creation_error,
    handle_loading_success,
    handle_loading_error,
    handle_processing_success,
    handle_reprocessing_success
)

# Import functional utilities  
from utils.processing import get_fingerprint_statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
	page_title="Setup - PickM8",
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
	"""Get list of existing sessions with metadata. Wrapper for service function."""
	return get_session_list()


def render_processing_pipeline(
	molecules_df, 
	protein_content: str, 
	processing_configs: dict
) -> tuple:
	"""Render processing pipeline with progress tracking and statistics display."""
	
	progress_bar = st.progress(0)
	status = st.empty()
	
	# Execute processing using service layer
	status.text("Starting processing pipeline...")
	progress_bar.progress(0.1)
	
	status.text("Processing molecules...")
	progress_bar.progress(0.3)
	
	# Use service layer function
	processing_result = execute_processing_pipeline(
		molecules_df, protein_content, processing_configs
	)
	
	progress_bar.progress(0.9)
	status.text("Finalizing...")
	
	processed_df = processing_result['processed_df']
	processing_summary = processing_result['processing_summary']
	
	progress_bar.progress(1.0)
	status.text("Processing complete!")
	
	if processing_summary['success']:
		st.success(f"✅ Successfully processed {len(processed_df)} molecules")
		
		# Display processing statistics using service layer function
		fp_stats = processing_summary['fingerprint_stats']
		stats_summary = create_processing_statistics_summary(fp_stats)
		
		render_processing_statistics(stats_summary)
	else:
		st.error(f"❌ Processing failed: {processing_summary.get('error', 'Unknown error')}")
	
	return processed_df, processing_summary


def render_processing_statistics(stats_summary: dict):
	"""Render processing statistics in expandable section."""
	with st.expander("📊 Processing Statistics"):
		col1, col2, col3, col4 = st.columns(4)
		
		with col1:
			st.metric("Molecules Processed", stats_summary['total_molecules'])
			st.metric("Morgan FP", f"{stats_summary['fingerprint_percentages']['morgan']:.1f}%")
		
		with col2:
			st.metric("RDKit FP", f"{stats_summary['fingerprint_percentages']['rdkit']:.1f}%")
			st.metric("MapChiral FP", f"{stats_summary['fingerprint_percentages']['mapchiral']:.1f}%")
		
		with col3:
			st.metric("Interaction FP", f"{stats_summary['fingerprint_percentages']['interaction']:.1f}%")
			st.metric("Avg Interactions", f"{stats_summary['interaction_metrics']['avg_interactions']:.1f}")
		
		with col4:
			st.metric("Molecules with Interactions", stats_summary['interaction_metrics']['molecules_with_interactions'])
			st.metric("Max Interactions", stats_summary['interaction_metrics']['max_interactions'])




def render_file_upload_section():
	"""Render file upload UI section. Returns uploaded file objects."""
	col1, col2 = st.columns(2)
	
	with col1:
		st.markdown("#### 1. Upload Protein Structure")
		protein_file = st.file_uploader("Select PDB file", type=['pdb'], key="new_protein")
		
		if protein_file:
			st.success(f"✅ Loaded protein: {protein_file.name}")
	
	with col2:
		st.markdown("#### 2. Upload Ligands")
		ligand_file = st.file_uploader("Select SDF file", type=['sdf', 'gz'], key="new_ligands")
		
		st.info("📋 **Score Requirements:**\n"
				"• All score values must be numeric\n" 
				"• No missing scores allowed\n"
				"• Choose direction preference below")
	
	return protein_file, ligand_file


def render_score_selection(ligand_file):
	"""Render score property selection UI. Returns score configuration."""
	if not ligand_file:
		return None, None, []
	
	# Save uploaded file temporarily and detect properties
	with tempfile.NamedTemporaryFile(delete=False, suffix='.sdf') as tmp:
		tmp.write(ligand_file.getvalue())
		ligand_path = tmp.name
	
	with st.spinner("Detecting SDF properties..."):
		available_properties = detect_sdf_properties(ligand_path)
	
	if available_properties:
		st.success(f"📋 Found {len(available_properties)} properties")
		
		# Find default score property
		default_score, default_index = find_default_score_property(available_properties)
		
		score_label = st.selectbox(
			"Select docking score property:",
			options=available_properties,
			index=default_index,
			help="Choose which property to use as the docking score"
		)
		
		# Score direction selection
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
	
	return score_label, score_direction, available_properties, ligand_path


def render_processing_configuration():
	"""Render processing configuration UI. Returns configuration options."""
	st.markdown("#### 3. Processing Configuration")
	
	col3, col4, col5 = st.columns(3)
	
	with col3:
		interaction_type = st.selectbox(
			"Interaction Analysis",
			options=["plip", "prolif"],
			index=0,
			help="PLIP: Fast protein-ligand interaction analysis\nProLIF: Comprehensive interaction fingerprints"
		)
	
	with col4:
		molecular_fp_types = st.multiselect(
			"Molecular Fingerprints",
			options=["morgan", "rdkit", "mapchiral"],
			default=["morgan", "rdkit", "mapchiral"],
			help="Select molecular fingerprint types to compute:\n• Morgan: Circular fingerprints\n• RDKit: Path-based fingerprints\n• MapChiral: Chiral-aware fingerprints"
		)
	
	with col5:
		compute_pose_quality = st.checkbox(
			"Compute Pose Quality Metrics",
			value=True,
			help="Calculate clash detection and strain energy (requires PoseCheck)"
		)
	
	return interaction_type, molecular_fp_types, compute_pose_quality


def handle_session_creation(
	protein_file, ligand_path: str, score_label: str, score_direction: str,
	available_properties: list, molecular_fp_types: list, 
	interaction_type: str, compute_pose_quality: bool
):
	"""Handle the session creation and processing logic using service layer."""
	with st.spinner("Creating session and processing molecules..."):
		# Execute business logic using service layer
		result = create_and_save_session(
			protein_file, ligand_path, score_label, score_direction,
			available_properties, molecular_fp_types, 
			interaction_type, compute_pose_quality
		)
	
	if result['success']:
		# Handle success using UI service layer
		success_response = handle_creation_success(
			result['session_id'],
			result['molecules_count'],
			result['processing_summary'],
			protein_file.name
		)
		
		# Update session state
		st.session_state.session_id = result['session_id']
		st.session_state.molecules_df = result['molecules_df']
		st.session_state.protein_content = result['protein_content']
		
		# Display success UI
		st.success(success_response['message'])
		if success_response['show_balloons']:
			st.balloons()
		
		st.info(f"Score range: {result['score_range'][0]:.3f} to {result['score_range'][1]:.3f}")
		
		# Show next steps
		st.markdown("### Next Steps:")
		for step in success_response['next_steps']:
			st.write(f"• {step}")
		
		# Navigation buttons
		render_navigation_buttons()
	else:
		# Handle error using UI service layer
		error_details = result.get('exception_details')
		available_columns = result.get('available_columns')
		
		error_response = handle_creation_error(
			result['error'], error_details
		)
		
		# Display error UI
		st.error(error_response['message'])
		
		if available_columns:
			st.error("Available columns: " + ", ".join(available_columns))
		
		if error_response['show_expander'] and error_details:
			with st.expander("Error Details"):
				st.code(error_details)
		
		# Show suggestions
		st.markdown("### Suggestions:")
		for suggestion in error_response['suggestions']:
			st.write(f"• {suggestion}")


def render_navigation_buttons():
	"""Render navigation buttons for after session creation."""
	col_nav1, col_nav2 = st.columns(2)
	with col_nav1:
		if st.button("🎯 Start Active Learning", type="primary"):
			st.switch_page("pages/3_🎯_Active_Learning.py")
	
	with col_nav2:
		if st.button("📊 View Results", type="secondary"):
			st.switch_page("pages/4_📊_Results.py")


def upload_new_session():
	"""Handle uploading new PDB and SDF files for a new session."""
	st.subheader("🆕 Create New Session")
	
	# File upload section
	protein_file, ligand_file = render_file_upload_section()
	
	# Score selection section (only if ligand file uploaded)
	score_config = None
	if ligand_file:
		score_config = render_score_selection(ligand_file)
	
	# Processing configuration section
	interaction_type, molecular_fp_types, compute_pose_quality = render_processing_configuration()
	
	# Validate inputs using UI service layer
	score_label = score_config[0] if score_config else None
	validation_result = validate_session_inputs(
		protein_file, ligand_file, score_label, molecular_fp_types
	)
	
	# Display validation warnings
	if validation_result['warnings']:
		for warning in validation_result['warnings']:
			st.warning(f"⚠️ {warning}")
	
	# Display validation errors
	if validation_result['errors']:
		for error in validation_result['errors']:
			st.error(f"❌ {error}")
	
	# Enable process button only when validation passes
	can_process = validation_result['is_valid'] and score_config
	
	# Process button
	if st.button("🚀 Create Session & Process", type="primary", disabled=not can_process):
		score_label, score_direction, available_properties, ligand_path = score_config
		
		handle_session_creation(
			protein_file, ligand_path, score_label, score_direction,
			available_properties, molecular_fp_types, 
			interaction_type, compute_pose_quality
		)


def render_session_list(sessions: list):
	"""Render the list of existing sessions with action buttons."""
	for i, session in enumerate(sessions):
		with st.container():
			col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
			
			with col1:
				st.markdown(f"""
				**{session['protein_name']}**  
				*Session: {session['session_id_short']}...*  
				*Created: {session['created_date'][:10] if session['created_date'] != 'Unknown' else 'Unknown'}*  
				*Modified: {session['last_modified'].strftime('%Y-%m-%d %H:%M')}*
				""")
			
			with col2:
				st.metric("Molecules", session['num_molecules'])
			
			with col3:
				st.metric("Graded", session['num_grades'])
				if session['num_molecules'] > 0:
					progress = session['num_grades'] / session['num_molecules']
					st.progress(progress)
			
			with col4:
				load_btn_col, reprocess_btn_col = st.columns(2)
				
				with load_btn_col:
					if st.button("📂 Load", key=f"load_{session['session_id']}", type="primary", use_container_width=True):
						st.session_state.selected_session_for_load = session['session_id']
				
				with reprocess_btn_col:
					if st.button("🔄 Config", key=f"config_{session['session_id']}", type="secondary", use_container_width=True):
						st.session_state.selected_session_for_reprocess = session['session_id']
		
		st.divider()


def handle_session_loading(sessions: list):
	"""Handle the session loading logic using service layer."""
	session_id = st.session_state.selected_session_for_load
	session = next((s for s in sessions if s['session_id'] == session_id), None)
	
	if session:
		# Load session using service layer
		session_data = load_session_data(session_id)
		
		if session_data:
			# Prepare session state data using UI service layer
			state_data = prepare_session_state_data(
				session_id,
				session_data['molecules_df'],
				session_data['metadata'].get('protein_content', ''),
				session_data['metadata']
			)
			
			# Update session state
			st.session_state.session_id = state_data['session_id']
			st.session_state.molecules_df = state_data['molecules_df']
			st.session_state.protein_content = state_data['protein_content']
			
			# Handle success using UI service layer
			success_response = handle_loading_success(
				session_id,
				session['protein_name'],
				state_data['num_molecules'],
				state_data['num_graded'],
				session['created_date']
			)
			
			# Display success UI
			st.success(success_response['message'])
			
			# Show next steps
			st.markdown("### Next Steps:")
			for step in success_response['next_steps']:
				st.write(f"• {step}")
			
			# Navigation buttons
			st.markdown("### Navigate to:")
			nav_col1, nav_col2 = st.columns(2)
			with nav_col1:
				if st.button("🎯 Active Learning", type="primary", key="nav_al_loaded"):
					st.switch_page("pages/3_🎯_Active_Learning.py")
			with nav_col2:
				if st.button("📊 View Results", type="secondary", key="nav_results_loaded"):
					st.switch_page("pages/4_📊_Results.py")
		else:
			# Handle error using UI service layer
			error_response = handle_loading_error(
				session_id, "Session data could not be loaded", session['protein_name']
			)
			
			st.error(error_response['message'])
			st.write("**Suggestions:**")
			for suggestion in error_response['suggestions']:
				st.write(f"• {suggestion}")
		
		# Clear the selection flag
		del st.session_state.selected_session_for_load
		st.stop()


def render_current_configuration(metadata: dict, session: dict):
	"""Render current session configuration display."""
	with st.expander("📋 Current Configuration", expanded=False):
		current_col1, current_col2, current_col3 = st.columns(3)
		with current_col1:
			st.write(f"**Interaction Type:** {metadata.get('interaction_type', 'Unknown')}")
		with current_col2:
			st.write(f"**Pose Quality:** {'✅' if metadata.get('compute_pose_quality', False) else '❌'}")
		with current_col3:
			st.write(f"**Molecules:** {session['num_molecules']}")


def render_reprocessing_configuration(metadata: dict):
	"""Render reprocessing configuration UI. Returns configuration values."""
	st.markdown("#### New Processing Settings")
	reprocess_col1, reprocess_col2, reprocess_col3 = st.columns(3)
	
	with reprocess_col1:
		reprocess_interaction_type = st.selectbox(
			"Interaction Analysis",
			options=["plip", "prolif"],
			index=0 if metadata.get('interaction_type', 'plip') == 'plip' else 1,
			help="PLIP: Fast protein-ligand interaction analysis\nProLIF: Comprehensive interaction fingerprints",
			key="reprocess_interaction"
		)
	
	with reprocess_col2:
		reprocess_molecular_fp_types = st.multiselect(
			"Molecular Fingerprints",
			options=["morgan", "rdkit", "mapchiral"],
			default=["morgan", "rdkit", "mapchiral"],
			help="Select molecular fingerprint types to compute",
			key="reprocess_molecular_fps"
		)
	
	with reprocess_col3:
		reprocess_pose_quality = st.checkbox(
			"Pose Quality Metrics",
			value=metadata.get('compute_pose_quality', True),
			help="Calculate clash detection and strain energy",
			key="reprocess_pose_quality"
		)
	
	return reprocess_interaction_type, reprocess_molecular_fp_types, reprocess_pose_quality


def handle_reprocessing_execution(
	session: dict, session_data: dict,
	molecular_fp_types: list, interaction_type: str, compute_pose_quality: bool
):
	"""Handle the reprocessing execution using service layer."""
	st.subheader("🔄 Reprocessing Molecules")
	st.info(f"🧬 Using **{interaction_type.upper()}** interaction analysis")
	st.warning("⚠️ This will overwrite existing fingerprint and interaction data. User grades will be preserved.")
	
	with st.spinner("Reprocessing molecules..."):
		# Execute reprocessing using business logic service layer
		result = execute_reprocessing_pipeline(
			session_data, molecular_fp_types, interaction_type, compute_pose_quality
		)
	
	if result['success']:
		# Handle success using UI service layer
		success_response = handle_reprocessing_success(
			result['session_id'],
			result['molecules_count'],
			result['preserved_grades']
		)
		
		# Update session state
		st.session_state.molecules_df = result['molecules_df']
		st.session_state.protein_content = result['protein_content']
		st.session_state.session_id = result['session_id']
		
		# Display success UI
		st.success(success_response['message'])
		
		# Show debug information
		if result.get('debug_info'):
			with st.expander("Debug Information"):
				for debug_line in result['debug_info']:
					st.write(debug_line)
		
		# Show next steps
		st.markdown("### Next Steps:")
		for step in success_response['next_steps']:
			st.write(f"• {step}")
	else:
		# Handle error
		st.error(f"❌ Reprocessing failed: {result['error']}")
		
		if result.get('exception_details'):
			with st.expander("Error Details"):
				st.code(result['exception_details'])
	
	# Clear reprocess selection and refresh
	del st.session_state.selected_session_for_reprocess
	st.rerun()


def handle_reprocessing_interface(sessions: list):
	"""Handle the reprocessing interface using service layer."""
	session_id = st.session_state.selected_session_for_reprocess
	session = next((s for s in sessions if s['session_id'] == session_id), None)
	
	if session:
		# Load session data using service layer
		session_data = load_session_data(session_id)
		
		if not session_data:
			st.error(f"❌ Failed to load session data for: {session['protein_name']}")
			del st.session_state.selected_session_for_reprocess
			return
		
		st.markdown("---")
		st.subheader(f"🔄 Reprocess: {session['protein_name']}")
		st.info("Recalculate molecular fingerprints, interaction fingerprints, and pose quality metrics with new settings.")
		
		# Display current configuration
		render_current_configuration(session_data['metadata'], session)
		
		# Reprocessing configuration
		reprocess_interaction_type, reprocess_molecular_fp_types, reprocess_pose_quality = render_reprocessing_configuration(session_data['metadata'])
		
		# Action buttons
		st.markdown("#### Actions")
		action_col1, action_col2, action_col3 = st.columns(3)
		
		with action_col1:
			if st.button("🔄 Start Reprocessing", type="primary", 
						disabled=not reprocess_molecular_fp_types,
						key="start_reprocess", use_container_width=True):
				
				handle_reprocessing_execution(
					session, session_data,
					reprocess_molecular_fp_types, reprocess_interaction_type, reprocess_pose_quality
				)
		
		with action_col2:
			if st.button("❌ Cancel", type="secondary", key="cancel_reprocess", use_container_width=True):
				del st.session_state.selected_session_for_reprocess
				st.rerun()
		
		with action_col3:
			if st.button("📂 Load Session", type="secondary", key="load_from_reprocess", use_container_width=True):
				# Load the session normally using service layer
				st.session_state.session_id = session['session_id']
				st.session_state.molecules_df = session_data['molecules_df']
				st.session_state.protein_content = session_data['metadata'].get('protein_content', '')
				
				del st.session_state.selected_session_for_reprocess
				st.success(f"✅ Loaded session: {session['protein_name']}")
				st.rerun()


def load_existing_session():
	"""Handle loading an existing session using service layer."""
	st.subheader("📂 Load Existing Session")
	
	# Get sessions using service layer
	sessions = get_existing_sessions()
	
	if not sessions:
		st.info("No existing sessions found. Create a new session to get started.")
		return
	
	st.markdown(f"Found **{len(sessions)}** existing sessions:")
	
	# Render session list
	render_session_list(sessions)
	
	# Handle session loading
	if 'selected_session_for_load' in st.session_state:
		handle_session_loading(sessions)
	
	# Handle reprocessing interface
	if 'selected_session_for_reprocess' in st.session_state:
		handle_reprocessing_interface(sessions)


def main():
	"""Main application entry point."""
	# Initialize session state
	if 'session_id' not in st.session_state:
		st.session_state.session_id = None
	if 'molecules_df' not in st.session_state:
		st.session_state.molecules_df = None
	if 'protein_content' not in st.session_state:
		st.session_state.protein_content = None
	
	st.title("🧬 Setup - PickM8")
	st.markdown("### Machine Learning-Guided Visual Inspection of Molecular Docking Results")
	
	# Main navigation tabs
	tab1, tab2 = st.tabs(["🆕 New Session", "📂 Load Session"])
	
	with tab1:
		upload_new_session()
	
	with tab2:
		load_existing_session()

if __name__ == "__main__":
	main()