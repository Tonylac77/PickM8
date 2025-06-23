"""
Active Learning Interface for PickM8
Using functional data processing approach.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback
from typing import Dict, List, Optional

# Import new functional utilities
from utils.data_processing import (
    load_molecules_dataframe, save_molecules_dataframe, load_session_metadata
)
from core.grading import (
    add_grade_to_molecule, get_graded_molecules, get_ungraded_molecules,
    filter_and_sort_molecules
)
from core.pose_analysis import (
    compute_pose_quality_batch, get_pose_quality_statistics,
    create_default_posecheck_config
)
from core.active_learning import (
    prepare_features_from_dataframe, train_model_with_calibration,
    select_molecules_for_labeling, update_model_predictions,
    encode_grades_for_training, get_training_statistics,
    predict_with_uncertainty
)
from utils.visualization import MoleculeVisualizer

# Configure logging - INFO level to reduce noise
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('streamlit').setLevel(logging.WARNING)

# Set up our logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

st.set_page_config(page_title="Active Learning - PickM8", page_icon="🎯", layout="wide")


def init_page_state():
    """Initialize page session state variables."""
    if 'current_mol_idx' not in st.session_state:
        st.session_state.current_mol_idx = 0
    if 'suggestion_strategy' not in st.session_state:
        st.session_state.suggestion_strategy = 'random'
    if 'has_predictions' not in st.session_state:
        st.session_state.has_predictions = False


def save_grade_to_dataframe(df: pd.DataFrame, mol_id: int, grade: str, session_dir: str) -> pd.DataFrame:
    """Save grade for a molecule and update the DataFrame."""
    try:
        # Update DataFrame with grade
        updated_df = add_grade_to_molecule(df, mol_id, grade)
        
        # Save updated DataFrame
        save_molecules_dataframe(updated_df, session_dir)
        
        logger.info(f"Saved grade {grade} for molecule {mol_id}")
        return updated_df
        
    except Exception as e:
        logger.error(f"Error saving grade: {e}")
        st.error(f"Error saving grade: {e}")
        return df




def create_molecule_table_data(df: pd.DataFrame, suggested_molecules: list = None) -> List[Dict]:
    """Create table data for molecule display."""
    table_data = []
    
    for idx, row in df.iterrows():
        grade_status = "✅ Graded" if pd.notna(row['grade']) else "⏳ Pending"
        grade_display = row['grade'] if pd.notna(row['grade']) else "Not graded"
        
        # Add predicted grade and uncertainty if available
        pred_display = ""
        if pd.notna(row.get('prediction')):
            pred_display = f"{row['prediction']}"
        else:
            pred_display = "N/A"
        
        uncertainty_display = ""
        if pd.notna(row.get('prediction_uncertainty')):
            uncertainty_display = f"{row['prediction_uncertainty']:.3f}"
        else:
            uncertainty_display = "N/A"
        
        # Add suggestion indicator
        suggestion_indicator = ""
        if suggested_molecules and row['id'] in suggested_molecules:
            suggestion_indicator = "🎯 "
        
        table_data.append({
            'Name': suggestion_indicator + row['name'],
            'Score': f"{row['score']:.3f}",
            'Clashes': int(row.get('clashes', 0)),
            'Strain Energy': f"{row.get('strain_energy', 0.0):.2f}",
            'Interactions': int(row.get('num_interactions', 0)),
            'Grade': grade_display,
            'Predicted': pred_display,
            'Uncertainty': uncertainty_display,
            'Status': grade_status,
            'mol_idx': idx
        })
    
    return table_data


def display_molecule_2d(mol_block: str, mol_name: str):
    """Display 2D structure of molecule."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdDepictor
        
        mol = Chem.MolFromMolBlock(mol_block)
        if mol is not None:
            # Generate 2D coordinates if needed
            rdDepictor.Compute2DCoords(mol)
            
            # Create 2D image
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption=mol_name, use_container_width=True)
        else:
            st.error("Could not parse molecule structure")
            
    except Exception as e:
        st.error(f"Error generating 2D structure: {str(e)}")


def display_molecule_data(mol_data: Dict):
    """Display compact molecule data summary using standard Streamlit components."""
    st.markdown("**📊 Data**")
    
    # Core molecule metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Score
        try:
            score = float(mol_data.get('score', 0.0))
            if pd.isna(score):
                st.metric("Score", "N/A")
            else:
                st.metric("Score", f"{score:.3f}")
        except (ValueError, TypeError):
            st.metric("Score", "N/A")
        
        # Strain Energy
        try:
            strain = float(mol_data.get('strain_energy', 0.0))
            if pd.isna(strain):
                st.metric("Strain Energy", "N/A")
            else:
                st.metric("Strain Energy", f"{strain:.2f}")
        except (ValueError, TypeError):
            st.metric("Strain Energy", "N/A")
    
    with col2:
        # Clashes
        try:
            clashes = int(mol_data.get('clashes', 0))
            st.metric("Clashes", clashes)
        except (ValueError, TypeError):
            st.metric("Clashes", "N/A")
        
        # Interactions
        try:
            interactions = int(mol_data.get('num_interactions', 0))
            st.metric("Interactions", interactions)
        except (ValueError, TypeError):
            st.metric("Interactions", "N/A")
    
    # Prediction data if available
    if pd.notna(mol_data.get('prediction')) or pd.notna(mol_data.get('prediction_uncertainty')):
        st.markdown("**🤖 ML Predictions**")
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            if pd.notna(mol_data.get('prediction')):
                try:
                    pred = mol_data['prediction']
                    # Handle both numeric and string predictions
                    if isinstance(pred, str):
                        st.metric("Predicted Grade", pred)
                    else:
                        # Convert numeric to grade if needed
                        pred_val = float(pred)
                        if pd.isna(pred_val):
                            st.metric("Predicted Grade", "N/A")
                        else:
                            st.metric("Predicted Grade", f"{pred_val:.2f}")
                except (ValueError, TypeError):
                    st.metric("Predicted Grade", "N/A")
        
        with pred_col2:
            if pd.notna(mol_data.get('prediction_uncertainty')):
                try:
                    uncertainty = float(mol_data['prediction_uncertainty'])
                    if pd.isna(uncertainty):
                        st.metric("Uncertainty", "N/A")
                    else:
                        st.metric("Uncertainty", f"{uncertainty:.3f}")
                except (ValueError, TypeError):
                    st.metric("Uncertainty", "N/A")


def handle_grade_selection(mol_id: int, current_grade: Optional[str]) -> Optional[str]:
    """Handle grade selection interface."""
    grade_options = ['A', 'B', 'C', 'D', 'F']
    grade_colors = {
        'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🟠', 'F': '🔴'
    }
    
    selected_grade = None
    
    for grade in grade_options:
        color = grade_colors[grade]
        is_selected = current_grade == grade
        
        if st.button(
            f"{color} {grade}",
            key=f"grade_{grade}_{mol_id}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
            help=f"Grade {grade}"
        ):
            selected_grade = grade
    
    return selected_grade


def train_model_interface(df: pd.DataFrame, session_dir: str):
    """Interface for training ML model."""
    graded_df = get_graded_molecules(df)
    
    if len(graded_df) < 3:
        st.warning(f"Need at least 3 graded molecules to train model. Currently have {len(graded_df)}.")
        return df
    
    with st.spinner("Training machine learning model..."):
        try:
            logger.info("Starting model training process...")
            
            # Prepare features and labels
            logger.info("Preparing features from graded molecules...")
            features, mol_ids = prepare_features_from_dataframe(graded_df)
            logger.info(f"Prepared features for {len(mol_ids)} molecules, feature shape: {features.shape if len(features) > 0 else 'empty'}")
            
            if len(features) == 0:
                error_msg = "No valid features found for model training"
                logger.error(error_msg)
                
                # Store error in session state so it persists
                st.session_state.training_error = {
                    'message': error_msg,
                    'traceback': 'No molecules have computed fingerprints available for training',
                    'timestamp': datetime.now().isoformat()
                }
                
                st.error(f"❌ {error_msg}")
                st.warning("This usually means:")
                st.write("• Fingerprints haven't been computed for graded molecules")
                st.write("• The data processing step was incomplete")
                st.write("• Graded molecules are missing required molecular data")
                
                # Show diagnostic information
                with st.expander("🔍 Diagnostic Information", expanded=True):
                    st.write("**Graded Molecules Fingerprint Status:**")
                    
                    fingerprint_status = []
                    for idx, row in graded_df.iterrows():
                        # Safe checking for fingerprint data (handles arrays and None)
                        def check_fp_status(fp_data):
                            if fp_data is None:
                                return '❌'
                            if isinstance(fp_data, (list, np.ndarray)):
                                return '✅' if len(fp_data) > 0 else '❌'
                            return '✅' if pd.notna(fp_data) else '❌'
                        
                        status = {
                            'ID': row['id'],
                            'Name': row['name'],
                            'Grade': row['grade'],
                            'Morgan FP': check_fp_status(row.get('morgan_fp')),
                            'RDKit FP': check_fp_status(row.get('rdkit_fp')),
                            'MapChiral FP': check_fp_status(row.get('mapchiral_fp')),
                            'Interaction FP': check_fp_status(row.get('interaction_fp'))
                        }
                        fingerprint_status.append(status)
                    
                    if fingerprint_status:
                        st.dataframe(pd.DataFrame(fingerprint_status), use_container_width=True)
                    
                    st.write("**Solution:** Go to the main page and reprocess the molecules to compute missing fingerprints.")
                
                return df
            
            # Get grades for training molecules
            logger.info("Extracting grades for training molecules...")
            grades = [graded_df[graded_df['id'] == mid]['grade'].iloc[0] for mid in mol_ids]
            logger.info(f"Training grades: {grades}")
            
            # Encode grades
            logger.info("Encoding grades for training...")
            encoded_labels, label_to_int, int_to_label = encode_grades_for_training(grades)
            logger.info(f"Encoded labels: {encoded_labels}")
            logger.info(f"Label mappings - to_int: {label_to_int}, to_label: {int_to_label}")
            
            # Train model
            logger.info("Training model with calibration...")
            model, metrics = train_model_with_calibration(features, encoded_labels)
            logger.info(f"Model training completed. Metrics: {metrics}")
            
            # Make predictions on all molecules
            logger.info("Preparing features for all molecules for prediction...")
            all_features, all_mol_ids = prepare_features_from_dataframe(df)
            logger.info(f"Prepared features for {len(all_mol_ids)} total molecules")
            
            if len(all_features) > 0:
                logger.info("Making predictions...")
                predictions, probabilities, uncertainties = predict_with_uncertainty(model, all_features)
                logger.info(f"Generated {len(predictions)} predictions")
                
                # Convert predictions back to grades
                logger.info("Converting predictions to grades...")
                prediction_data = {}
                for i, mol_id in enumerate(all_mol_ids):
                    # Convert numeric prediction back to grade
                    numeric_pred = predictions[i]
                    try:
                        # Ensure we have a valid integer for lookup
                        pred_int = int(round(numeric_pred))
                        predicted_grade = int_to_label.get(pred_int, 'Unknown')
                        
                        # Debug logging
                        if predicted_grade == 'Unknown':
                            logger.warning(f"Unknown prediction value {numeric_pred} -> {pred_int} for molecule {mol_id}")
                            logger.warning(f"Available mappings: {int_to_label}")
                        
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error converting prediction {numeric_pred} for molecule {mol_id}: {e}")
                        predicted_grade = 'Unknown'
                    
                    prediction_data[mol_id] = {
                        'prediction': predicted_grade,  # Store as grade letter
                        'uncertainty': float(uncertainties[i])
                    }
                
                logger.info(f"Converted {len(prediction_data)} predictions to grades")
                
                # Update DataFrame with predictions  
                logger.info("Updating DataFrame with predictions...")
                updated_df = update_model_predictions(df, prediction_data)
                
                # Save updated DataFrame
                logger.info("Saving updated DataFrame...")
                save_molecules_dataframe(updated_df, session_dir)
                
                # Debug info
                pred_count = updated_df['prediction'].notna().sum()
                logger.info(f"Successfully updated {pred_count} molecules with predictions")
                
                st.success(f"✅ Model trained successfully! Accuracy: {metrics.get('train_accuracy', 'N/A'):.3f}")
                st.info(f"Generated predictions for {pred_count} molecules")
                
                return updated_df
            else:
                logger.warning("No features available for making predictions")
                st.warning("No features available for making predictions")
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            st.error(f"❌ Error training model: {str(e)}")
            
            # Display full error trace for debugging  
            import traceback
            full_trace = traceback.format_exc()
            logger.error(f"Full traceback: {full_trace}")
            
            with st.expander("🔍 Full Error Details (Click to expand)", expanded=False):
                st.code(full_trace, language="python")
                
            # Additional debugging info
            st.write("**Debug Information:**")
            st.write(f"- Graded molecules: {len(graded_df)}")
            st.write(f"- Total molecules: {len(df)}")
            st.write(f"- Session directory: {session_dir}")
            
            # Show column information
            st.write("**Available DataFrame columns:**")
            st.write(list(df.columns))
            
            # Show sample data
            if len(graded_df) > 0:
                st.write("**Sample graded molecule data:**")
                sample_cols = ['id', 'name', 'score', 'grade', 'morgan_fp', 'interaction_fp', 'clashes', 'strain_energy']
                display_cols = [col for col in sample_cols if col in graded_df.columns]
                st.dataframe(graded_df[display_cols].head(2))
    
    return df


def main():
    """Main Active Learning interface."""
    st.title("🎯 Active Learning Interface")
    init_page_state()
    
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
    protein_content = session_metadata.get('protein_content', '') if session_metadata else ''
    
    # Ensure pose quality metrics are computed
    if 'clashes' not in molecules_df.columns or molecules_df['clashes'].isna().all():
        st.info("Computing pose quality metrics...")
        from utils.pose_analysis import create_default_posecheck_config
        pose_config = create_default_posecheck_config()
        molecules_df = compute_pose_quality_batch(molecules_df, protein_content, pose_config)
        save_molecules_dataframe(molecules_df, session_dir)
    
    # Check for predictions once and store in session state
    has_predictions = 'prediction_uncertainty' in molecules_df.columns and molecules_df['prediction_uncertainty'].notna().any()
    st.session_state.has_predictions = has_predictions
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Controls")
        
        # Fixed mode - always annotate
        mode = "annotate"
        st.info("**Mode:** Annotate  \n*Grading ungraded molecules*")
        
        # Suggestion strategy selection
        if st.session_state.has_predictions:
            strategy_options = ["uncertainty", "predicted_grade"]
            strategy_labels = ["Highest Uncertainty", "Highest Predicted Grade"]
        else:
            strategy_options = ["random", "highest_score"]
            strategy_labels = ["Random", "Highest Score"]
        
        # Get current index safely
        try:
            current_strategy = st.session_state.suggestion_strategy
            current_idx = strategy_options.index(current_strategy) if current_strategy in strategy_options else 0
        except:
            current_idx = 0
        
        strategy_display = st.selectbox(
            "Selection Strategy",
            strategy_labels,
            index=current_idx,
            key="strategy_selector"
        )
        
        # Map display back to internal value and update session state
        new_strategy = strategy_options[strategy_labels.index(strategy_display)]
        
        # Check if strategy changed and force rerun if needed
        if st.session_state.suggestion_strategy != new_strategy:
            st.session_state.suggestion_strategy = new_strategy
            st.session_state.current_mol_idx = 0  # Reset to first molecule with new strategy
            st.rerun()
        
        st.divider()
        
        # Progress metrics
        graded_count = molecules_df['grade'].notna().sum()
        total_count = len(molecules_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Graded", graded_count)
        with col2:
            st.metric("📊 Total", total_count)
        
        if total_count > 0:
            progress = graded_count / total_count
            st.progress(progress)
            st.caption(f"{progress:.1%} Complete")
        
        st.divider()
        
        # Training interface
        if st.button("🤖 Train Model", type="primary", disabled=graded_count < 3):
            # Clear any previous training error
            if 'training_error' in st.session_state:
                del st.session_state.training_error
                
            # Train the model
            updated_molecules_df = train_model_interface(molecules_df, session_dir)
            
            # Only update and rerun if training was successful (no error stored)
            if 'training_error' not in st.session_state:
                st.session_state.molecules_df = updated_molecules_df
                st.session_state.current_mol_idx = 0  # Reset to first suggested molecule after training
                st.rerun()
            # If there's an error, don't rerun - this preserves the error display
        
        # Display persistent training error if it exists
        if 'training_error' in st.session_state:
            st.error(f"❌ Training Error: {st.session_state.training_error['message']}")
            
            with st.expander("🔍 Full Error Details", expanded=False):
                st.text(f"Error occurred at: {st.session_state.training_error['timestamp']}")
                st.code(st.session_state.training_error['traceback'], language="python")
                
            if st.button("Clear Error", key="clear_training_error"):
                del st.session_state.training_error
                st.rerun()
        
        # Show model training status
        if st.session_state.has_predictions:
            st.success("🤖 Model Trained!")
            pred_count = molecules_df['prediction'].notna().sum()
            st.metric("Predictions Made", pred_count)
        
        st.divider()
        
        if st.button("🏠 Main Page", type="secondary"):
            st.switch_page("main.py")
    
    # Get suggested molecules based on strategy using session state
    suggested_molecules = None
    current_strategy = st.session_state.suggestion_strategy
    
    # Always get suggested molecules based on current strategy
    suggested_molecules = select_molecules_for_labeling(molecules_df, n_molecules=20, strategy=current_strategy)
    
    
    # Filter and sort molecules (with suggestions prioritized if available)
    filtered_df = filter_and_sort_molecules(molecules_df, mode, current_strategy, suggested_molecules)
    
    if len(filtered_df) == 0:
        st.info(f"No molecules to show in {mode} mode.")
        return
    
    # Ensure current index is valid
    if st.session_state.current_mol_idx >= len(filtered_df):
        st.session_state.current_mol_idx = 0
    
    current_mol = filtered_df.iloc[st.session_state.current_mol_idx]
    
    # Main three-column layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # 2D Structure
        st.subheader("2D Structure")
        display_molecule_2d(current_mol['mol_block'], current_mol['name'])
        
        # Molecule data
        display_molecule_data(current_mol)
    
    with col2:
        # 3D Visualization with suggestion indicator
        mol_title = f"🧬 {current_mol['name']}"
        if suggested_molecules and current_mol['id'] in suggested_molecules:
            mol_title = f"🎯 {mol_title}"
            st.subheader(mol_title)
            st.caption(f"**Suggested by {current_strategy.replace('_', ' ').title()} Strategy**")
        else:
            st.subheader(mol_title)
        
        if protein_content:
            visualizer = MoleculeVisualizer()
            try:
                visualizer.show_complex(
                    protein_content,
                    current_mol['mol_block'],
                    current_mol.get('interactions', '[]'),
                    key=f"mol_{current_mol['id']}"
                )
                
                # Interaction summary
                interaction_summary = visualizer.get_interaction_summary(current_mol.get('interactions', '[]'))
                with st.expander("🔬 Interaction Summary", expanded=True):
                    visualizer.show_interaction_legend(interaction_summary)
                    
            except Exception as e:
                st.error(f"Error displaying 3D structure: {e}")
        else:
            st.warning("No protein structure available for visualization")
    
    with col3:
        # Grading interface
        st.subheader("⭐ Grade")
        
        current_grade = current_mol['grade'] if pd.notna(current_mol['grade']) else None
        selected_grade = handle_grade_selection(current_mol['id'], current_grade)
        
        if selected_grade:
            # Save grade and update DataFrame
            molecules_df = save_grade_to_dataframe(molecules_df, current_mol['id'], selected_grade, session_dir)
            st.session_state.molecules_df = molecules_df
            st.success(f"✅ Saved grade: {selected_grade}")
            
            # Auto-advance to next molecule (will be recalculated with new suggestions)
            if st.session_state.current_mol_idx < len(filtered_df) - 1:
                st.session_state.current_mol_idx += 1
            else:
                # If at end, reset to beginning to check for new suggestions
                st.session_state.current_mol_idx = 0
                
            st.rerun()
        
        st.divider()
        
        # Navigation
        if suggested_molecules and len(suggested_molecules) > 1:
            # Show suggestion navigation when multiple suggestions available
            col_prev, col_next, col_suggestion = st.columns([1, 1, 1])
        else:
            col_prev, col_next = st.columns(2)
            col_suggestion = None
        
        with col_prev:
            if st.button("⬅️", disabled=st.session_state.current_mol_idx == 0, 
                        use_container_width=True, help="Previous molecule"):
                st.session_state.current_mol_idx -= 1
                st.rerun()
        
        with col_next:
            if st.button("➡️", disabled=st.session_state.current_mol_idx >= len(filtered_df) - 1,
                        use_container_width=True, help="Next molecule"):
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        # Add jump to next suggestion button
        if col_suggestion and suggested_molecules:
            # Find next ungraded suggestion
            current_pos = st.session_state.current_mol_idx
            next_suggestion_idx = None
            
            for i in range(current_pos + 1, len(filtered_df)):
                if filtered_df.iloc[i]['id'] in suggested_molecules:
                    next_suggestion_idx = i
                    break
            
            # If no suggestions after current position, wrap to beginning
            if next_suggestion_idx is None:
                for i in range(0, current_pos):
                    if filtered_df.iloc[i]['id'] in suggested_molecules:
                        next_suggestion_idx = i
                        break
            
            if next_suggestion_idx is not None:
                if st.button("🎯", use_container_width=True, help="Jump to next suggestion"):
                    st.session_state.current_mol_idx = next_suggestion_idx
                    st.rerun()
        
        # Enhanced position indicator
        position_info = f"{st.session_state.current_mol_idx + 1}/{len(filtered_df)}"
        
        # Add suggestion position if applicable
        if suggested_molecules and current_mol['id'] in suggested_molecules:
            suggestion_position = suggested_molecules.index(current_mol['id']) + 1
            position_info += f" • 🎯 Suggestion #{suggestion_position}"
        
        st.caption(position_info)
    
    # Molecule table
    st.divider()
    st.subheader("📋 Molecule Table")
    
    # Add explanation of symbols
    if suggested_molecules:
        st.caption("🎯 = Suggested by ML model")
    
    table_data = create_molecule_table_data(filtered_df, suggested_molecules)
    
    if table_data:
        # Convert to DataFrame for display
        df_display = pd.DataFrame(table_data)
        
        # Pagination
        rows_per_page = 20
        total_rows = len(df_display)
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        if 'table_page' not in st.session_state:
            st.session_state.table_page = 0
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️ Page", disabled=st.session_state.table_page == 0):
                st.session_state.table_page -= 1
                st.rerun()
        with col2:
            st.caption(f"Page {st.session_state.table_page + 1}/{total_pages}")
        with col3:
            if st.button("Page ➡️", disabled=st.session_state.table_page >= total_pages - 1):
                st.session_state.table_page += 1
                st.rerun()
        
        # Display current page
        start_idx = st.session_state.table_page * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        page_data = df_display.iloc[start_idx:end_idx]
        
        # Display table
        st.dataframe(
            page_data.drop('mol_idx', axis=1),
            use_container_width=True,
            hide_index=True
        )
        


if __name__ == "__main__":
    main()