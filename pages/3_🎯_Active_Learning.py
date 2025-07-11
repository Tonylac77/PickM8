"""Active Learning Interface - Simplified"""
import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any

# Import from new modular structure
from sessions import sessions
from data_io import molecules
from machine_learning import ml_models
from analysis import grading, statistics
from ui.components import molecule_viewer, progress_displays, forms

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

    st.subheader("🎛️ Controls")

    # Progress metrics
    stats = grading.get_grading_statistics(df)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Graded", stats['graded_count'])
    with col2:
        st.metric("Total", stats['total_molecules'])

    st.progress(stats['grading_percentage'] / 100)

    # Selection Strategy
    st.subheader("🎯 Selection Strategy")
    
    # Determine available strategies based on model status
    has_model = grading.has_trained_model(df)
    
    if has_model:
        strategies = ["Best Predictions"]
        default_strategy = "Best Predictions"
    else:
        strategies = ["Random", "Best Score"]
        default_strategy = "Best Score"
    
    # Initialize selection strategy in session state
    if 'selection_strategy' not in st.session_state:
        st.session_state.selection_strategy = default_strategy
    
    if st.session_state.selection_strategy not in strategies:
        st.session_state.selection_strategy = default_strategy
    
    # Update strategy if model status changed
    strategy_was_reset = False
    if st.session_state.selection_strategy not in strategies:
        old_strategy = st.session_state.selection_strategy
        st.session_state.selection_strategy = default_strategy
        strategy_was_reset = True
    
    # Strategy dropdown
    st.session_state.selection_strategy = st.selectbox(
        "Selection Strategy",
        strategies,
        index=strategies.index(st.session_state.selection_strategy) if st.session_state.selection_strategy in strategies else 0,
        help="Choose how molecules are selected for grading"
    )
    
    # ML Configuration
    st.subheader("🤖 Machine Learning")
    
    
    # Get current model configuration
    metadata = st.session_state.get('metadata', {})
    config = metadata.get('config', {})
    current_model_config = config.get('model_config', {})
    
    # Default model config if none exists
    if not current_model_config:
        current_model_config = {
            'model_type': 'RandomForest',
            'use_calibration': True,
            'model_params': {}
        }
    
    # Model switcher in expandable section
    with st.expander("🔄 Change Model", expanded=False):
        new_model_type, new_model_config, config_changed = forms.render_model_switcher(current_model_config)
        
        if config_changed:
            if st.button("📝 Update Model Configuration", use_container_width=True):
                # Update session metadata with new model config
                if 'metadata' not in st.session_state:
                    st.session_state.metadata = {}
                if 'config' not in st.session_state.metadata:
                    st.session_state.metadata['config'] = {}
                
                st.session_state.metadata['config']['model_config'] = new_model_config
                
                # Save updated metadata
                sessions.save_session(
                    st.session_state.session_id,
                    st.session_state.molecules_df,
                    st.session_state.metadata
                )
                
                st.success(f"✅ Model configuration updated to {new_model_type}")
                st.rerun()
    
    # Display current model configuration
    display_model_type = current_model_config.get('model_type', 'RandomForest')
    display_use_calibration = current_model_config.get('use_calibration', True)
    
    st.info(f"**Current Model**: {display_model_type}")
    if display_use_calibration:
        st.caption("✅ Probability calibration enabled")
    else:
        st.caption("❌ Probability calibration disabled")
    
    # Training buttons
    if st.button("🚀 Train Model", 
                 disabled=stats['graded_count'] < 3, 
                 use_container_width=True):
        train_model(df)
    
        # Show retrain with new config button if config changed but not yet applied
    has_model = grading.has_trained_model(df)
    if has_model and st.button("🔄 Retrain with New Config", 
                               disabled=stats['graded_count'] < 3, 
                               use_container_width=True):
        train_model_with_config_update(df)

    # Reset grades section
    st.divider()
    st.subheader("🔄 Reset")
    
    if st.button("🗑️ Reset All Grades", 
                 disabled=stats['graded_count'] == 0, 
                 use_container_width=True,
                 help="Clear all grades and predictions to start fresh"):
        reset_grades(df)

    # Navigation
    st.divider()
    if st.button("🏠 Main Page"):
        st.switch_page("main.py")

def render_grading_interface(df: pd.DataFrame, metadata: Dict[str, Any]):
    """Render the main grading interface."""
    # Get best ungraded molecule using selected strategy
    strategy = st.session_state.get('selection_strategy', 'Best Score')
    current_mol = grading.get_best_ungraded_molecule(df, strategy, metadata)

    if current_mol is None:
        st.info("All molecules have been graded!")
        return

    # Sleek unified control bar
    st.markdown("""
    <style>
    /* Unified button styling for perfect alignment */
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
        'A': {'color': '#28a745', 'desc': 'Excellent', 'emoji': '⭐'},
        'B': {'color': '#6f42c1', 'desc': 'Good', 'emoji': '👍'},
        'C': {'color': '#fd7e14', 'desc': 'Fair', 'emoji': '👌'},
        'D': {'color': '#dc3545', 'desc': 'Poor', 'emoji': '👎'}
    }

    # Single row with 5 columns for grade buttons and progress
    ctrl_cols = st.columns(5, gap="small")
    
    # Grade buttons (columns 0-3)
    for i, (grade, info) in enumerate(grade_info.items()):
        with ctrl_cols[i]:
            # Individual grade button styling
            st.markdown(f"""
            <style>
            div[data-testid="column"]:nth-child({i+1}) .stButton > button {{
                background-color: {info['color']} !important;
                color: white !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            if st.button(f"{info['emoji']} {grade}", 
                        key=f"grade_{grade}", 
                        use_container_width=True,
                        help=info['desc']):
                # Update grade
                df_updated = grading.add_grade(df, current_mol['id'], grade)
                st.session_state.molecules_df = df_updated

                # Save session
                sessions.save_session(
                    st.session_state.session_id,
                    df_updated,
                    st.session_state.metadata
                )

                st.rerun()
    
    # Progress indicator
    with ctrl_cols[4]:
        # Get current ungraded count for progress display
        filtered_df = grading.get_molecules_by_strategy(df, strategy, metadata)
        progress_pct = 1 / len(filtered_df) if len(filtered_df) > 0 else 1
        st.progress(progress_pct)
        remaining = len(filtered_df) - 1
        st.caption(f"1/{len(filtered_df)} • {remaining} left")

    # Main content row - 3D | 2D | Properties
    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        # 3D Visualization
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
        # 2D Visualization
        st.markdown("### ⚛️ 2D Structure")
        viewer = molecule_viewer.MoleculeVisualizer()
        viewer.show_2d_structure(current_mol['mol_block'], size=(300, 300))

    with col3:
        # Compact properties panel
        st.markdown("### 📊 Properties")
        viewer.show_compact_molecule_info(current_mol.to_dict())

def train_model(df: pd.DataFrame):
    """Train ML model on graded molecules."""
    with st.spinner("Training model..."):
        try:
            # Get model configuration from session metadata
            metadata = st.session_state.get('metadata', {})
            config = metadata.get('config', {})
            model_config = config.get('model_config')
            
            # Use default fingerprint configuration (ECFP, FunctionalGroups, MACCS, Interaction)
            # Train model with default fingerprint selection
            model, metrics = ml_models.train_model(df, model_config)

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
            
            # Train model with new configuration using default fingerprint selection
            model, metrics = ml_models.train_model(df_clear, model_config)

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
    stats = grading.get_grading_statistics(df)
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