"""UI form components and validation."""
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import logging
from utils.config import load_ml_config, ConfigurationError

logger = logging.getLogger(__name__)

def validate_session_inputs(
    protein_file: Any,
    ligand_file: Any,
    score_label: Optional[str],
    fingerprint_types: List[str]
) -> Dict[str, Any]:
    """
    Validate inputs for session creation.

    Chain-of-Thought:
    - Pure validation function
    - Returns structured validation result
    - No side effects or UI rendering
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check required files
    if not protein_file:
        result['valid'] = False
        result['errors'].append("Protein file is required")

    if not ligand_file:
        result['valid'] = False
        result['errors'].append("Ligand file is required")

    # Check file extensions
    if protein_file and not protein_file.name.lower().endswith('.pdb'):
        result['warnings'].append("Protein file should have .pdb extension")

    if ligand_file and not ligand_file.name.lower().endswith('.sdf'):
        result['warnings'].append("Ligand file should have .sdf extension")

    # Check other inputs
    if not score_label:
        result['valid'] = False
        result['errors'].append("Score column must be selected")

    if not fingerprint_types:
        result['valid'] = False
        result['errors'].append("At least one fingerprint type must be selected")

    return result

def render_file_upload_section() -> Tuple[Any, Any]:
    """Render file upload UI section."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1. Upload Protein Structure")
        protein_file = st.file_uploader("Select PDB file", type=['pdb'], key="protein")

        if protein_file:
            st.success(f"✅ Loaded: {protein_file.name}")

    with col2:
        st.markdown("#### 2. Upload Ligands")
        ligand_file = st.file_uploader("Select SDF file", type=['sdf'], key="ligands")

        if ligand_file:
            st.success(f"✅ Loaded: {ligand_file.name}")

    return protein_file, ligand_file

def render_processing_options() -> Dict[str, Any]:
    """Render processing configuration options."""
    st.markdown("#### 3. Processing Options")

    col1, col2 = st.columns(2)

    with col1:
        fingerprint_types = st.multiselect(
            "Molecular Fingerprints",
            options=[
                'mapchiral',  # Legacy (kept for backward compatibility)
                'e3fp', 'ecfp', 'electroshape', 'functional_groups', 
                'maccs', 'pattern', 'pharmacophore'  # scikit-fingerprints
            ],
            default=['ecfp', 'functional_groups', 'maccs'],
            help="Select fingerprint types to compute. Legacy: MapChiral. scikit-fingerprints: E3FP, ECFP, ElectroShape, FunctionalGroups, MACCS, Pattern, Pharmacophore"
        )
    with col2:
        interaction_type = st.selectbox(
            "Interaction Analysis",
            options=['plip', 'prolif'],
            index=0,
            help="Method for protein-ligand interaction analysis"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        compute_pose_quality = st.checkbox(
            "Compute Pose Quality",
            value=True,
            help="Calculate clash detection and strain energy"
        )
    with col4:
        # Check if GRADE is available
        try:
            from features.grade_descriptors import is_grade_available
            grade_available = is_grade_available()
        except ImportError:
            grade_available = False
        
        compute_grade = st.checkbox(
            "Compute GRADE Descriptors",
            value=False,
            disabled=not grade_available,
            help="Calculate GRAIL affinity prediction descriptors (requires CDPL/GRAIL). Parameters configured in config.yaml"
        )

    return {
        'fingerprint_types': fingerprint_types,
        'interaction_type': interaction_type,
        'compute_pose_quality': compute_pose_quality,
        'compute_grade': compute_grade
    }


def render_random_forest_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render RandomForest parameter form."""
    with st.expander("RandomForest Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.number_input(
                "Number of trees", 
                min_value=10, max_value=1000, 
                value=defaults.get('n_estimators', 100),
                help="Number of trees in the forest"
            )
            min_samples_split = st.number_input(
                "Min samples split", 
                min_value=2, max_value=20, 
                value=defaults.get('min_samples_split', 2),
                help="Minimum samples required to split an internal node"
            )
        
        with col2:
            max_depth = st.number_input(
                "Max depth", 
                min_value=1, max_value=50, 
                value=defaults.get('max_depth') or 10,
                help="Maximum depth of the tree (None = unlimited)"
            )
            min_samples_leaf = st.number_input(
                "Min samples leaf", 
                min_value=1, max_value=20, 
                value=defaults.get('min_samples_leaf', 1),
                help="Minimum samples required to be at a leaf node"
            )
    
    return {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth) if max_depth else None,
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'max_features': defaults.get('max_features', 'sqrt'),
        'random_state': 42,
        'n_jobs': -1
    }

def render_gaussian_process_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render GaussianProcess parameter form."""
    with st.expander("Gaussian Process Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            kernel = st.selectbox(
                "Kernel", 
                options=['RBF', 'Matern'],
                index=['RBF', 'Matern'].index(defaults.get('kernel', 'RBF')),
                help="Kernel type for Gaussian Process"
            )
        
        with col2:
            n_restarts = st.number_input(
                "Optimizer restarts", 
                min_value=0, max_value=10, 
                value=defaults.get('n_restarts_optimizer', 0),
                help="Number of restarts of the optimizer"
            )
    
    st.info("💡 Gaussian Process model configuration")
    
    return {
        'kernel': kernel,
        'n_restarts_optimizer': int(n_restarts),
        'random_state': 42
    }

def render_logistic_at_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render LogisticAT ordinal regression parameter form."""
    with st.expander("Ordinal Logistic Regression (LogisticAT) Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            alpha = st.number_input(
                "Regularization Strength (alpha)",
                min_value=0.001,
                max_value=100.0,
                value=float(defaults.get('alpha', 1.0)),
                step=0.1,
                help="Regularization parameter to prevent overfitting"
            )
            
        with col2:
            max_iter = st.number_input(
                "Maximum Iterations",
                min_value=100,
                max_value=5000,
                value=int(defaults.get('max_iter', 1000)),
                step=100,
                help="Maximum number of iterations for convergence"
            )
    
    return {
        'alpha': alpha,
        'max_iter': int(max_iter),
        'random_state': 42
    }

def render_ml_model_options() -> Tuple[str, Dict[str, Any]]:
    """
    Render ML model selection and parameter configuration.
    
    Returns:
        Tuple of (model_type, model_params)
    """
    st.markdown("#### 4. Machine Learning Model")
    
    # Load configuration with error handling
    try:
        config = load_ml_config()
    except ConfigurationError as e:
        st.error(f"❌ Configuration Error: {e}")
        st.info("💡 Please check your config.yaml file and ensure the ml_models section is properly configured.")
        # Use minimal fallback defaults
        config = {
            'default_type': 'RandomForest',
            'RandomForest': {'n_estimators': 100, 'max_depth': None},
            'GaussianProcess': {'kernel': 'RBF'},
            'LogisticAT': {'alpha': 1.0, 'max_iter': 1000}
        }
    
    # Model type selection
    model_type = st.selectbox(
        "Model Type",
        options=['RandomForest', 'GaussianProcess', 'LogisticAT'],
        index=['RandomForest', 'GaussianProcess', 'LogisticAT'].index(
            config.get('default_type', 'RandomForest')
        ) if config.get('default_type', 'RandomForest') in ['RandomForest', 'GaussianProcess', 'LogisticAT'] else 0,
        help="Select the machine learning algorithm to use for predictions"
    )
    
    # Get default parameters for selected model
    model_defaults = config.get(model_type, {})
    
    # Render model-specific parameters
    if model_type == 'RandomForest':
        model_params = render_random_forest_params(model_defaults)
    elif model_type == 'GaussianProcess':
        model_params = render_gaussian_process_params(model_defaults)
    elif model_type == 'LogisticAT':
        model_params = render_logistic_at_params(model_defaults)
    else:
        model_params = model_defaults
    
    # Calibration disabled - no UI options needed
    use_calibration = False
    
    return model_type, {
        'model_params': model_params,
        'use_calibration': use_calibration
    }

def render_model_switcher(current_config: Dict[str, Any]) -> Tuple[str, Dict[str, Any], bool]:
    """
    Render compact model switcher for active learning sessions.
    
    Args:
        current_config: Current model configuration from session metadata
        
    Returns:
        Tuple of (model_type, model_config, config_changed)
    """
    # Load ML configuration defaults with error handling
    try:
        config = load_ml_config()
    except ConfigurationError as e:
        st.error(f"❌ Configuration Error: {e}")
        # Use minimal fallback defaults
        config = {
            'default_type': 'RandomForest',
            'RandomForest': {'n_estimators': 100, 'max_depth': None},
            'GaussianProcess': {'kernel': 'RBF'},
            'LogisticAT': {'alpha': 1.0, 'max_iter': 1000}
        }
    
    # Get current values or defaults
    current_model_type = current_config.get('model_type', 'RandomForest')
    current_use_calibration = False
    current_model_params = current_config.get('model_params', {})
    
    # Model type selection
    new_model_type = st.selectbox(
        "Model Type",
        options=['RandomForest', 'GaussianProcess', 'LogisticAT'],
        index=['RandomForest', 'GaussianProcess', 'LogisticAT'].index(current_model_type) if current_model_type in ['RandomForest', 'GaussianProcess', 'LogisticAT'] else 0,
        help="Change the ML algorithm for predictions",
        key="model_switcher_type"
    )
    
    # Get model defaults for the selected type
    model_defaults = config.get(new_model_type, {})
    
    # Use current params if same model type, otherwise use defaults
    if new_model_type == current_model_type:
        param_defaults = current_model_params
    else:
        param_defaults = model_defaults
    
    # Render model-specific parameters in compact form
    if new_model_type == 'RandomForest':
        new_model_params = render_compact_random_forest_params(param_defaults)
    elif new_model_type == 'GaussianProcess':
        new_model_params = render_compact_gaussian_process_params(param_defaults)
    elif new_model_type == 'LogisticAT':
        new_model_params = render_compact_logistic_at_params(param_defaults)
    else:
        new_model_params = param_defaults
    
    # Calibration disabled - no UI options needed
    new_use_calibration = False

    
    # Check if configuration changed
    config_changed = (
        new_model_type != current_model_type or
        new_use_calibration != current_use_calibration or
        new_model_params != current_model_params
    )
    
    new_config = {
        'model_type': new_model_type,
        'model_params': new_model_params,
        'use_calibration': new_use_calibration
    }
    
    return new_model_type, new_config, config_changed

def render_compact_random_forest_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact RandomForest parameter form for model switching."""
    n_estimators = st.slider(
        "Trees", 
        min_value=10, max_value=300, 
        value=defaults.get('n_estimators', 100),
        step=10,
        help="Number of trees in the forest"
    )
    max_depth = st.slider(
        "Max Depth", 
        min_value=3, max_value=20, 
        value=defaults.get('max_depth') or 10,
        help="Maximum depth of trees"
    )
    
    return {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': defaults.get('min_samples_split', 2),
        'min_samples_leaf': defaults.get('min_samples_leaf', 1),
        'max_features': defaults.get('max_features', 'sqrt'),
        'random_state': 42,
        'n_jobs': -1
    }

def render_compact_gaussian_process_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact GaussianProcess parameter form for model switching."""
    kernel = st.selectbox(
        "Kernel", 
        options=['RBF', 'Matern'],
        index=['RBF', 'Matern'].index(defaults.get('kernel', 'RBF')),
        help="Kernel type"
    )
    
    return {
        'kernel': kernel,
        'n_restarts_optimizer': defaults.get('n_restarts_optimizer', 0),
        'random_state': 42
    }

def render_compact_logistic_at_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact LogisticAT parameter form for model switching."""
    alpha = st.slider(
        "Regularization (alpha)", 
        min_value=0.1, max_value=10.0, 
        value=float(defaults.get('alpha', 1.0)),
        step=0.1,
        help="Regularization strength"
    )
    
    return {
        'alpha': alpha,
        'max_iter': defaults.get('max_iter', 1000),
        'random_state': 42
    }

