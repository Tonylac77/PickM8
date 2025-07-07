"""UI form components and validation."""
import streamlit as st
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

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
            options=['morgan', 'rdkit', 'mapchiral'],
            default=['morgan', 'rdkit'],
            help="Select fingerprint types to compute"
        )
    with col2:
        interaction_type = st.selectbox(
            "Interaction Analysis",
            options=['plip', 'prolif'],
            index=0,
            help="Method for protein-ligand interaction analysis"
        )
    compute_pose_quality = st.checkbox(
        "Compute Pose Quality",
        value=True,
        help="Calculate clash detection and strain energy"
    )

    return {
        'fingerprint_types': fingerprint_types,
        'interaction_type': interaction_type,
        'compute_pose_quality': compute_pose_quality
    }

def load_ml_config() -> Dict[str, Any]:
    """Load ML model configuration from config.yaml with fallback defaults."""
    config_path = Path("config.yaml")
    
    default_config = {
        'default_type': 'RandomForest',
        'calibration_enabled': True,
        'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
        'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
        'SVM': {'C': 1.0, 'gamma': 'scale', 'kernel': 'rbf'},
        'GaussianProcess': {'kernel': 'RBF'},
        'MLP': {'hidden_layer_sizes': [100], 'learning_rate': 'constant', 'alpha': 0.0001}
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('ml_models', default_config)
        return default_config
    except Exception as e:
        logger.error(f"Error loading ML config: {e}")
        return default_config

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

def render_gradient_boosting_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render GradientBoosting parameter form."""
    with st.expander("Gradient Boosting Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.number_input(
                "Number of boosting stages", 
                min_value=10, max_value=500, 
                value=defaults.get('n_estimators', 100),
                help="Number of boosting stages"
            )
            max_depth = st.number_input(
                "Max depth", 
                min_value=1, max_value=20, 
                value=defaults.get('max_depth', 3),
                help="Maximum depth of individual regression estimators"
            )
        
        with col2:
            learning_rate = st.number_input(
                "Learning rate", 
                min_value=0.01, max_value=1.0, 
                value=defaults.get('learning_rate', 0.1),
                step=0.01,
                help="Learning rate shrinks the contribution of each tree"
            )
            subsample = st.number_input(
                "Subsample", 
                min_value=0.1, max_value=1.0, 
                value=defaults.get('subsample', 1.0),
                step=0.1,
                help="Fraction of samples used for fitting trees"
            )
    
    return {
        'n_estimators': int(n_estimators),
        'learning_rate': float(learning_rate),
        'max_depth': int(max_depth),
        'subsample': float(subsample),
        'min_samples_split': defaults.get('min_samples_split', 2),
        'min_samples_leaf': defaults.get('min_samples_leaf', 1),
        'random_state': 42
    }

def render_svm_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render SVM parameter form."""
    with st.expander("SVM Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            C = st.number_input(
                "Regularization parameter (C)", 
                min_value=0.01, max_value=100.0, 
                value=defaults.get('C', 1.0),
                step=0.1,
                help="Regularization parameter (smaller = more regularization)"
            )
            kernel = st.selectbox(
                "Kernel", 
                options=['rbf', 'linear', 'poly', 'sigmoid'],
                index=['rbf', 'linear', 'poly', 'sigmoid'].index(defaults.get('kernel', 'rbf')),
                help="Kernel type for SVM"
            )
        
        with col2:
            gamma = st.selectbox(
                "Gamma", 
                options=['scale', 'auto'],
                index=['scale', 'auto'].index(defaults.get('gamma', 'scale')),
                help="Kernel coefficient for rbf, poly and sigmoid"
            )
    
    return {
        'C': float(C),
        'gamma': gamma,
        'kernel': kernel,
        'probability': True,  # Always enable for uncertainty
        'random_state': 42
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
    
    st.info("💡 Gaussian Process provides natural uncertainty estimates without requiring calibration")
    
    return {
        'kernel': kernel,
        'n_restarts_optimizer': int(n_restarts),
        'random_state': 42
    }

def render_mlp_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render MLP parameter form."""
    with st.expander("Neural Network (MLP) Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            hidden_size = st.number_input(
                "Hidden layer size", 
                min_value=10, max_value=500, 
                value=defaults.get('hidden_layer_sizes', [100])[0],
                help="Number of neurons in hidden layer"
            )
            alpha = st.number_input(
                "L2 penalty (alpha)", 
                min_value=1e-6, max_value=1e-2, 
                value=defaults.get('alpha', 0.0001),
                format="%.6f",
                help="L2 penalty (regularization) parameter"
            )
        
        with col2:
            learning_rate = st.selectbox(
                "Learning rate", 
                options=['constant', 'invscaling', 'adaptive'],
                index=['constant', 'invscaling', 'adaptive'].index(defaults.get('learning_rate', 'constant')),
                help="Learning rate schedule for weight updates"
            )
            max_iter = st.number_input(
                "Max iterations", 
                min_value=50, max_value=1000, 
                value=defaults.get('max_iter', 200),
                help="Maximum number of iterations"
            )
    
    return {
        'hidden_layer_sizes': [int(hidden_size)],
        'learning_rate': learning_rate,
        'alpha': float(alpha),
        'max_iter': int(max_iter),
        'activation': defaults.get('activation', 'relu'),
        'solver': defaults.get('solver', 'adam'),
        'random_state': 42
    }

def render_autoparty_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render AutoParty ensemble parameter form."""
    with st.expander("AutoParty Ensemble Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            committee_size = st.number_input(
                "Committee size", 
                min_value=2, max_value=10, 
                value=defaults.get('committee_size', 3),
                help="Number of models in the ensemble"
            )
            n_neurons = st.number_input(
                "Neurons per layer", 
                min_value=128, max_value=4096, 
                value=defaults.get('n_neurons', 1024),
                step=128,
                help="Number of neurons in each hidden layer"
            )
            hidden_layers = st.number_input(
                "Hidden layers", 
                min_value=1, max_value=5, 
                value=defaults.get('hidden_layers', 2),
                help="Number of hidden layers"
            )
        
        with col2:
            dropout = st.number_input(
                "Dropout rate", 
                min_value=0.0, max_value=0.5, 
                value=defaults.get('dropout', 0.2),
                step=0.05,
                help="Dropout probability for regularization"
            )
            data_split = st.selectbox(
                "Data split method", 
                options=['bootstrap', 'full-split'],
                index=['bootstrap', 'full-split'].index(defaults.get('data_split', 'bootstrap')),
                help="How to split data among ensemble members"
            )
    
    st.info("🤖 AutoParty uses an ensemble of neural networks with built-in uncertainty estimation")
    
    return {
        'committee_size': int(committee_size),
        'n_neurons': int(n_neurons),
        'hidden_layers': int(hidden_layers),
        'dropout': float(dropout),
        'data_split': data_split
    }

def render_ml_model_options() -> Tuple[str, Dict[str, Any]]:
    """
    Render ML model selection and parameter configuration.
    
    Returns:
        Tuple of (model_type, model_params)
    """
    st.markdown("#### 4. Machine Learning Model")
    
    # Load configuration
    config = load_ml_config()
    
    # Model type selection - Add AutoPartyEnsemble
    model_type = st.selectbox(
        "Model Type",
        options=['RandomForest', 'GradientBoosting', 'SVM', 'GaussianProcess', 'MLP', 'AutoPartyEnsemble'],
        index=['RandomForest', 'GradientBoosting', 'SVM', 'GaussianProcess', 'MLP', 'AutoPartyEnsemble'].index(
            config.get('default_type', 'RandomForest')
        ),
        help="Select the machine learning algorithm to use for predictions"
    )
    
    # Get default parameters for selected model
    model_defaults = config.get(model_type, {})
    
    # Render model-specific parameters
    if model_type == 'RandomForest':
        model_params = render_random_forest_params(model_defaults)
    elif model_type == 'GradientBoosting':
        model_params = render_gradient_boosting_params(model_defaults)
    elif model_type == 'SVM':
        model_params = render_svm_params(model_defaults)
    elif model_type == 'GaussianProcess':
        model_params = render_gaussian_process_params(model_defaults)
    elif model_type == 'MLP':
        model_params = render_mlp_params(model_defaults)
    elif model_type == 'AutoPartyEnsemble':
        model_params = render_autoparty_params(model_defaults)
    else:
        model_params = model_defaults
    
    # Calibration options
    use_calibration = st.checkbox(
        "Enable Probability Calibration",
        value=config.get('calibration_enabled', True),
        help="Improve probability estimates for better uncertainty quantification",
        disabled=(model_type == 'AutoPartyEnsemble')  # AutoParty has built-in uncertainty
    )
    
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
    # Load ML configuration defaults
    config = load_ml_config()
    
    # Get current values or defaults
    current_model_type = current_config.get('model_type', 'RandomForest')
    current_use_calibration = current_config.get('use_calibration', True)
    current_model_params = current_config.get('model_params', {})
    
    # Model type selection
    new_model_type = st.selectbox(
        "Model Type",
        options=['RandomForest', 'GradientBoosting', 'SVM', 'GaussianProcess', 'MLP', 'AutoPartyEnsemble'],
        index=['RandomForest', 'GradientBoosting', 'SVM', 'GaussianProcess', 'MLP', 'AutoPartyEnsemble'].index(current_model_type),
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
    elif new_model_type == 'GradientBoosting':
        new_model_params = render_compact_gradient_boosting_params(param_defaults)
    elif new_model_type == 'SVM':
        new_model_params = render_compact_svm_params(param_defaults)
    elif new_model_type == 'GaussianProcess':
        new_model_params = render_compact_gaussian_process_params(param_defaults)
    elif new_model_type == 'MLP':
        new_model_params = render_compact_mlp_params(param_defaults)
    elif new_model_type == 'AutoPartyEnsemble':
        new_model_params = render_compact_autoparty_params(param_defaults)
    else:
        new_model_params = param_defaults
    
    # Calibration checkbox
    new_use_calibration = st.checkbox(
        "Enable Probability Calibration",
        value=current_use_calibration,
        help="Improve probability estimates for uncertainty",
        key="model_switcher_calibration",
        disabled=(new_model_type == 'AutoPartyEnsemble')  # AutoParty has built-in uncertainty
    )

    
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

def render_compact_gradient_boosting_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact GradientBoosting parameter form for model switching."""
    n_estimators = st.slider(
        "Boosting Stages", 
        min_value=10, max_value=200, 
        value=defaults.get('n_estimators', 100),
        step=10,
        help="Number of boosting stages"
    )
    learning_rate = st.slider(
        "Learning Rate", 
        min_value=0.01, max_value=0.5, 
        value=defaults.get('learning_rate', 0.1),
        step=0.01,
        help="Learning rate"
    )
    
    return {
        'n_estimators': int(n_estimators),
        'learning_rate': float(learning_rate),
        'max_depth': defaults.get('max_depth', 3),
        'subsample': defaults.get('subsample', 1.0),
        'min_samples_split': defaults.get('min_samples_split', 2),
        'min_samples_leaf': defaults.get('min_samples_leaf', 1),
        'random_state': 42
    }

def render_compact_svm_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact SVM parameter form for model switching."""
    C = st.slider(
        "Regularization (C)", 
        min_value=0.1, max_value=10.0, 
        value=defaults.get('C', 1.0),
        step=0.1,
        help="Regularization parameter"
    )
    kernel = st.selectbox(
        "Kernel", 
        options=['rbf', 'linear', 'poly'],
        index=['rbf', 'linear', 'poly'].index(defaults.get('kernel', 'rbf')),
        help="Kernel type"
    )
    
    return {
        'C': float(C),
        'gamma': defaults.get('gamma', 'scale'),
        'kernel': kernel,
        'probability': True,
        'random_state': 42
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

def render_compact_mlp_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact MLP parameter form for model switching."""
    hidden_size = st.slider(
        "Hidden Neurons", 
        min_value=50, max_value=200, 
        value=defaults.get('hidden_layer_sizes', [100])[0],
        step=10,
        help="Number of neurons in hidden layer"
    )
    alpha = st.select_slider(
        "Regularization", 
        options=[1e-5, 1e-4, 1e-3, 1e-2],
        value=defaults.get('alpha', 0.0001),
        format_func=lambda x: f"{x:.0e}",
        help="L2 penalty parameter"
    )
    
    return {
        'hidden_layer_sizes': [int(hidden_size)],
        'learning_rate': defaults.get('learning_rate', 'constant'),
        'alpha': float(alpha),
        'max_iter': defaults.get('max_iter', 200),
        'activation': defaults.get('activation', 'relu'),
        'solver': defaults.get('solver', 'adam'),
        'random_state': 42
    }

def render_compact_autoparty_params(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render compact AutoParty parameter form for model switching."""
    committee_size = st.slider(
        "Committee Size", 
        min_value=2, max_value=5, 
        value=defaults.get('committee_size', 3),
        help="Number of models in the ensemble"
    )
    n_neurons = st.select_slider(
        "Neurons per Layer", 
        options=[256, 512, 1024, 2048],
        value=defaults.get('n_neurons', 1024),
        help="Number of neurons in each hidden layer"
    )
    dropout = st.slider(
        "Dropout Rate", 
        min_value=0.1, max_value=0.5, 
        value=defaults.get('dropout', 0.2),
        step=0.05,
        help="Dropout probability for regularization"
    )
    
    return {
        'committee_size': int(committee_size),
        'n_neurons': int(n_neurons),
        'hidden_layers': defaults.get('hidden_layers', 2),
        'dropout': float(dropout),
        'data_split': defaults.get('data_split', 'bootstrap')
    }