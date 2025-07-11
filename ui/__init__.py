"""
UI Package for PickM8

This package provides the user interface components and pages for the PickM8 molecular screening application.
The UI is organized into two main modules:
- components: Reusable UI components (forms, molecule viewer, progress displays)
- pages: Streamlit page definitions
"""

# Import all key UI components to provide a clean public API
from .components import (
    validate_session_inputs,
    render_file_upload_section,
    render_processing_options,
    render_ml_model_options,
    render_model_switcher,
    load_ml_config,
    MoleculeVisualizer,
    display_progress_metrics,
    display_processing_status
)

# Import submodules for direct access
from . import components

__all__ = [
    # Core components
    'components',
    'pages',
    
    # Form components
    'validate_session_inputs',
    'render_file_upload_section',
    'render_processing_options',
    'render_ml_model_options',
    'render_model_switcher',
    'load_ml_config',
    
    # Visualization components
    'MoleculeVisualizer',
    
    # Progress components
    'display_progress_metrics',
    'display_processing_status'
]