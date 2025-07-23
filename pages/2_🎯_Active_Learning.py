"""Active Learning Interface - Simplified"""
import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

# Import from new modular structure
from sessions import sessions
from machine_learning import ml_models
from analysis import grading, similarity
from ui.components import molecule_viewer, forms

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Active Learning - PickM8", page_icon="media/pickm8_white_logoonly.png", layout="wide")

def main():
    """Main Active Learning interface."""
    # Add logo to app and sidebar
    st.logo(
        image="media/pickm8_white.png",
        size="large",
        icon_image="media/pickm8_white_logoonly.png"
    )
    
    st.title("🎯 Active Learning")

    # Check session
    if 'session_id' not in st.session_state or 'molecules_df' not in st.session_state:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🏠 Go to Main Page", use_container_width=True, type="primary"):
                st.switch_page("main.py")

            st.info("💡 Create a new session or load an existing one from the main page to access active learning features.")
        return

    # Load current data
    df = st.session_state.molecules_df
    metadata = st.session_state.get('metadata', {})

    # Add controls to sidebar
    with st.sidebar:
        render_sidebar_controls()

    # Main content
    render_grading_interface(df, metadata)

def render_sidebar_controls():
    """Render sidebar controls."""
    df = st.session_state.molecules_df
    review_mode = grading.is_review_mode(df)

    st.subheader("🎛️ Controls")

    # Progress metrics
    stats = grading.get_grading_statistics(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Graded", stats['graded_count'])
    with col2:
        st.metric("Total", stats['total_molecules'])

    st.progress(stats['grading_percentage'] / 100)

    if review_mode:
        # Review mode info
        st.success("✅ All molecules graded!")
        
        # Grade distribution in review mode
        if 'grade_distribution' in stats:
            st.subheader("📊 Grade Distribution")
            for grade, count in sorted(stats['grade_distribution'].items()):
                st.write(f"**Grade {grade}**: {count} molecules")
    else:
        # Normal mode: Selection Strategy
        st.subheader("🎯 Selection Strategy")
        
        has_model = grading.has_trained_model(df)
        
        if has_model:
            strategies = ["Best Predictions"]
            default_strategy = "Best Predictions"
        else:
            strategies = ["Random", "Best Score"]
            default_strategy = "Best Score"
        
        if 'selection_strategy' not in st.session_state:
            st.session_state.selection_strategy = default_strategy
        
        if st.session_state.selection_strategy not in strategies:
            st.session_state.selection_strategy = default_strategy
        
        st.session_state.selection_strategy = st.selectbox(
            "Selection Strategy",
            strategies,
            index=strategies.index(st.session_state.selection_strategy) if st.session_state.selection_strategy in strategies else 0,
            help="Choose how molecules are selected for grading"
        )
        
        # ML Configuration
        st.subheader("🤖 Machine Learning")
        
        # Model configuration and training buttons
        metadata = st.session_state.get('metadata', {})
        config = metadata.get('config', {})
        current_model_config = config.get('model_config', {})
        
        if not current_model_config:
            current_model_config = {
                'model_type': 'RandomForest',
                'use_calibration': True,
                'model_params': {}
            }
        
        with st.expander("🔄 Change Model", expanded=False):
            new_model_type, new_model_config, config_changed = forms.render_model_switcher(current_model_config)
            
            if config_changed:
                if st.button("📝 Update Model Configuration", use_container_width=True):
                    if 'metadata' not in st.session_state:
                        st.session_state.metadata = {}
                    if 'config' not in st.session_state.metadata:
                        st.session_state.metadata['config'] = {}
                    
                    st.session_state.metadata['config']['model_config'] = new_model_config
                    
                    sessions.save_session(
                        st.session_state.session_id,
                        st.session_state.molecules_df,
                        st.session_state.metadata
                    )
                    
                    st.success(f"✅ Model configuration updated to {new_model_type}")
                    st.rerun()
        
        display_model_type = current_model_config.get('model_type', 'RandomForest')
        
        st.info(f"**Current Model**: {display_model_type}")
        
        # Training buttons
        if st.button("🚀 Train Model", 
                     disabled=stats['graded_count'] < 3, 
                     use_container_width=True):
            train_model(df)
        
        has_model = grading.has_trained_model(df)
        if has_model and st.button("🔄 Retrain with New Config", 
                                   disabled=stats['graded_count'] < 3, 
                                   use_container_width=True):
            train_model_with_config_update(df)

def render_grading_interface(df: pd.DataFrame, metadata: Dict[str, Any]):
    """Render the main grading interface."""
    
    # Check if we're in review mode
    review_mode = grading.is_review_mode(df)
    
    # Initialize navigation index in session state if not present
    if 'current_review_index' not in st.session_state:
        st.session_state.current_review_index = 0
    
    # Get current molecule
    if review_mode:
        # Review mode: navigate through all graded molecules
        current_mol = grading.get_molecule_for_review(
            df, 
            st.session_state.current_review_index, 
            metadata
        )
        total_molecules = len(df)
    else:
        # Normal mode: get best ungraded molecule
        strategy = st.session_state.get('selection_strategy', 'Best Score')
        if strategy == 'Best Predictions' and grading.has_trained_model(df):
            current_mol = grading.get_best_predicted_ungraded_molecule(df, metadata)
        else:
            current_mol = grading.get_best_ungraded_molecule(df, strategy, metadata)
        
        # For progress calculation in normal mode
        ungraded = grading.get_ungraded_molecules(df)
        total_molecules = len(ungraded)

    if current_mol is None:
        st.info("No molecules to display!")
        return

    # Display review mode info
    if review_mode:
        st.info("✅ **All molecules have been graded!** You are now in review mode. ML training is no longer available.")

    # Grade editing controls
    st.markdown("""
    <style>
    .stButton > button {
        height: 45px !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Define grade colors and descriptions
    grade_info = {
        'A': {'color': '#28a745', 'desc': 'Excellent', 'emoji': '🟢'},
        'B': {'color': '#6f42c1', 'desc': 'Good', 'emoji': '🟡'},
        'C': {'color': '#fd7e14', 'desc': 'Fair', 'emoji': '🟠'},
        'D': {'color': '#dc3545', 'desc': 'Poor', 'emoji': '🔴'}
    }

    # Navigation and grade buttons row
    if review_mode:
        # Review mode: Previous/Next buttons + Grade buttons + Progress
        nav_col1, nav_col2, *grade_cols, prog_col = st.columns([1, 1, 1, 1, 1, 1, 2])
        
        # Previous button
        with nav_col1:
            if st.button("◀ Previous", use_container_width=True, 
                        disabled=st.session_state.current_review_index == 0):
                st.session_state.current_review_index -= 1
                st.rerun()
        
        # Next button
        with nav_col2:
            if st.button("Next ▶", use_container_width=True,
                        disabled=st.session_state.current_review_index >= len(df) - 1):
                st.session_state.current_review_index += 1
                st.rerun()
    else:
        # Normal mode: Just grade buttons + progress
        grade_cols = st.columns(4)
        prog_col = st.columns([1, 1, 1, 1, 1])[4]
    
    # Grade buttons
    for i, (grade, info) in enumerate(grade_info.items()):
        with grade_cols[i]:
            # Highlight current grade in review mode
            is_current_grade = review_mode and current_mol.get('grade') == grade
            
            # Individual grade button styling
            st.markdown(f"""
            <style>
            div[data-testid="column"]:nth-child({i+3 if review_mode else i+1}) .stButton > button {{
                background-color: {info['color']} !important;
                color: white !important;
                {f"box-shadow: 0 0 20px {info['color']}80 !important;" if is_current_grade else ""}
            }}
            </style>
            """, unsafe_allow_html=True)
            
            button_label = f"{info['emoji']} {grade}"
            if is_current_grade:
                button_label += " ✓"
            
            if st.button(button_label, 
                        key=f"grade_{grade}", 
                        use_container_width=True,
                        help=f"{info['desc']} - Click to {'change' if review_mode else 'set'} grade"):
                # Update grade
                df_updated = grading.add_grade(df, current_mol['id'], grade)
                st.session_state.molecules_df = df_updated

                # Save session
                sessions.save_session(
                    st.session_state.session_id,
                    df_updated,
                    st.session_state.metadata
                )

                if not review_mode:
                    st.rerun()
                else:
                    # In review mode, just update the display
                    st.success(f"Grade updated to {grade}")
                    st.rerun()
    
    # Progress indicator
    with prog_col:
        if review_mode:
            current = st.session_state.current_review_index + 1
            total = len(df)
            st.progress(current / total)
            st.caption(f"{current}/{total} • Review Mode")
        else:
            # In normal mode, show remaining ungraded
            current = 1  # Always showing the "next" molecule to grade
            remaining = total_molecules - 1
            st.progress(current / total_molecules if total_molecules > 0 else 1)
            st.caption(f"{current}/{total_molecules} • {remaining} left")

    # Rest of the interface remains the same (3D viewer, 2D structure, properties)
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.markdown("### 🧬 3D Structure")
        if metadata.get('protein_content'):
            viewer = molecule_viewer.MoleculeVisualizer()
            viewer.show_complex(
                metadata['protein_content'],
                current_mol['mol_block'],
                current_mol.get('interactions', '[]'),
                key=f"mol_{current_mol['id']}"
            )
        else:
            st.warning("No protein structure available for 3D visualization")

    with col2:
        st.markdown("### ⚛️ 2D Structure")
        viewer = molecule_viewer.MoleculeVisualizer()
        viewer.show_2d_structure(current_mol['mol_block'], size=(300, 300))

    with col3:
        st.markdown("### 📊 Properties")
        viewer.show_compact_molecule_info(current_mol.to_dict())

    # Add similarity analysis for both active learning and review modes
    render_similarity_analysis(df, current_mol, metadata, search_only_graded=not review_mode)

def render_similarity_analysis(df: pd.DataFrame, current_mol: pd.Series, metadata: Dict[str, Any], search_only_graded: bool = False):
    """Render similarity analysis for both active learning and review modes."""
    st.markdown("### 🔍 Most Similar Molecule")
    
    # Filter dataset based on mode
    if search_only_graded:
        # Active learning mode: only search in graded molecules
        search_df = df[df['grade'].notna()].copy()
        if len(search_df) == 0:
            st.info("No graded molecules available for similarity comparison yet.")
            return
    else:
        # Review mode: search in all molecules
        search_df = df.copy()
    
    # Try to use molecule objects for efficient similarity calculation
    if 'mol' in df.columns and current_mol.get('mol') is not None:
        # Use RDKit molecule objects directly (more efficient)
        similar_result = similarity.find_most_similar_molecule_by_mol(
            current_mol['mol'], 
            search_df,  # Use filtered dataset
            fp_type='morgan',  # Use Morgan fingerprints (ECFP equivalent)
            exclude_self=True,
            target_id=current_mol.get('id')
        )
    else:
        # Fallback to pre-computed fingerprints
        config = metadata.get('config', {})
        fingerprint_columns = similarity.get_enabled_fingerprint_columns(config)
        
        if not fingerprint_columns:
            st.warning("No fingerprint data available for similarity analysis.")
            return
        
        similar_result = similarity.find_most_similar_molecule(
            current_mol, search_df, fingerprint_columns, exclude_self=True  # Use filtered dataset
        )
    
    if similar_result is None:
        st.info("No similar molecules found.")
        return
    
    similar_mol, similarity_score = similar_result

    # Display similar molecule in same layout as current molecule
    st.markdown(f"**Similar Molecule:** {similar_mol.get('name', 'Unknown')}")
    
    col1, col2, col3 = st.columns([6, 2.5, 2])
    
    with col1:
        st.markdown("#### 🧬 3D Structure")
        if metadata.get('protein_content'):
            viewer = molecule_viewer.MoleculeVisualizer()
            viewer.show_complex(
                metadata['protein_content'],
                similar_mol['mol_block'],
                similar_mol.get('interactions', '[]'),
                key=f"similar_mol_{similar_mol['id']}"
            )
        else:
            st.warning("No protein structure available for 3D visualization")

    with col2:
        st.markdown("#### ⚛️ 2D Structure")
        viewer = molecule_viewer.MoleculeVisualizer()
        viewer.show_2d_structure(similar_mol['mol_block'], size=(300, 300))

        st.markdown(
            f"##### 🟰 Tanimoto Similarity: {similarity_score:.3f}",
            help="Similarity score based on Morgan fingerprints (0 = completely different, 1 = identical)"
        )
        
        # Show grade comparison
        current_grade = current_mol.get('grade', 'N/A')
        similar_grade = similar_mol.get('grade', 'N/A')
        
        if current_grade != 'N/A' and similar_grade != 'N/A':
            grade_match = '✅ Same' if current_grade == similar_grade else '❌ Different'
            st.markdown(f"##### 🆚 Grade Comparison: {current_grade} vs {similar_grade} \n ##### {grade_match}")
        else:
            st.markdown(f"**Similar Mol Grade:** {similar_grade}")

        # Show score comparison
        current_score = current_mol.get('score', 0)
        similar_score = similar_mol.get('score', 0)
        score_diff = similar_score - current_score

        st.markdown(f"##### 💯Score Difference: {score_diff:+.3f}", help=f"Current: {current_score:.3f}, Similar: {similar_score:.3f}")

    with col3:
        st.markdown("#### 📊 Properties")
        viewer.show_compact_molecule_info(similar_mol.to_dict())

def train_model(df: pd.DataFrame):
    """Train ML model on graded molecules."""
    with st.spinner("Training model..."):
        try:
            # Get model configuration from session metadata
            metadata = st.session_state.get('metadata', {})
            config = metadata.get('config', {})
            model_config = config.get('model_config')
            
            # Get fingerprint configuration from session config
            fingerprint_config = config.get('fingerprint_config')
            
            # Train model with automatic fingerprint selection based on available data
            model, metrics = ml_models.train_model(df, model_config, fingerprint_config)

            # Update predictions with trained model
            df_updated = ml_models.update_predictions(df, model, metrics)

            # Store label mapping in metadata for prediction display
            if 'metadata' not in st.session_state:
                st.session_state.metadata = {}
            st.session_state.metadata['label_mapping'] = metrics.get('label_mapping', {})

            # Save updated data
            st.session_state.molecules_df = df_updated
            sessions.save_session(
                st.session_state.session_id,
                df_updated,
                st.session_state.metadata
            )

            model_type = metrics.get('model_type', 'Unknown')
            accuracy = metrics.get('accuracy', 0)
            n_samples = metrics.get('n_samples', 0)
            use_calibration = metrics.get('use_calibration', False)
            
            success_msg = f"✅ {model_type} model trained! Accuracy: {accuracy:.2f} ({n_samples} samples)"
            if use_calibration:
                success_msg += " with calibration"
            
            st.success(success_msg)
            st.rerun()

        except Exception as e:
            st.error(f"Training failed: {str(e)}")

def train_model_with_config_update(df: pd.DataFrame):
    """Train model and update predictions, replacing any existing predictions."""
    with st.spinner("Retraining model with new configuration..."):
        try:
            # Get updated model configuration from session metadata
            metadata = st.session_state.get('metadata', {})
            config = metadata.get('config', {})
            model_config = config.get('model_config')
            
            # Clear existing predictions before retraining
            df_clear = df.copy()
            df_clear['prediction'] = None
            df_clear['prediction_timestamp'] = None
            
            # Get fingerprint configuration from session config
            fingerprint_config = config.get('fingerprint_config')
            
            # Train model with new configuration and automatic fingerprint selection
            model, metrics = ml_models.train_model(df_clear, model_config, fingerprint_config)

            # Update predictions with new model
            df_updated = ml_models.update_predictions(df_clear, model, metrics)

            # Store label mapping in metadata for prediction display
            if 'metadata' not in st.session_state:
                st.session_state.metadata = {}
            st.session_state.metadata['label_mapping'] = metrics.get('label_mapping', {})

            # Save updated data
            st.session_state.molecules_df = df_updated
            sessions.save_session(
                st.session_state.session_id,
                df_updated,
                st.session_state.metadata
            )

            model_type = metrics.get('model_type', 'Unknown')
            accuracy = metrics.get('accuracy', 0)
            n_samples = metrics.get('n_samples', 0)
            use_calibration = metrics.get('use_calibration', False)
            
            success_msg = f"🔄 Model switched to {model_type}! Accuracy: {accuracy:.2f} ({n_samples} samples)"
            if use_calibration:
                success_msg += " with calibration"
            
            st.success(success_msg)
            st.info("All predictions have been updated with the new model.")
            st.rerun()

        except Exception as e:
            st.error(f"Model retraining failed: {str(e)}")

def reset_grades(df: pd.DataFrame):
    """Reset all grades with comprehensive session state cleanup and confirmation dialog."""
    try:
        
        # Reset grades in the dataframe
        df_reset = grading.reset_all_grades(df)
        
        # Update session state
        st.session_state.molecules_df = df_reset
        
        # Comprehensive session state cleanup
        _cleanup_session_state_after_reset()
        
        # Clean up model-related metadata
        if 'metadata' in st.session_state:
            st.session_state.metadata = grading.cleanup_model_metadata(st.session_state.metadata)
        
        # Save the reset data
        save_success = sessions.save_session(
            st.session_state.session_id,
            df_reset,
            st.session_state.metadata
        )
        
        # DEBUG: Verify saved data
        if save_success:
            result = sessions.load_session(st.session_state.session_id)
            if result:
                loaded_df, _ = result

        # Reset confirmation state
        st.session_state.confirm_reset = False
        
        # Success message
        st.success("✅ Reset complete!")
        
        # Wait a bit to see debug output
        import time
        time.sleep(2)
        
        st.rerun()
        
    except Exception as e:
        st.session_state.confirm_reset = False
        st.error(f"❌ Reset failed: {str(e)}")
        logger.error(f"Reset operation failed: {e}", exc_info=True)

def _cleanup_session_state_after_reset():
    """Clean up session state variables after grade reset."""
    # Reset molecule navigation
    if 'current_idx' in st.session_state:
        st.session_state.current_idx = 0
    
    # Force strategy re-evaluation by removing it
    if 'selection_strategy' in st.session_state:
        del st.session_state.selection_strategy
    
    # Clear any cached model state
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith('model_') or k.startswith('prediction_')]
    for key in keys_to_clear:
        del st.session_state[key]

if __name__ == "__main__":
    main()