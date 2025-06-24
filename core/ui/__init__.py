"""
UI service layer using functional programming approach.
Handles validation, state management, and response formatting.
"""

from .validation import (
    validate_session_inputs,
    validate_file_uploads,
    validate_processing_configuration
)

from .state_management import (
    prepare_session_state_data,
    update_session_state,
    clear_session_selections
)

from .response_handlers import (
    handle_creation_success,
    handle_creation_error,
    handle_loading_success,
    handle_loading_error,
    handle_processing_success,
    handle_processing_error,
    handle_reprocessing_success
)

__all__ = [
    'validate_session_inputs',
    'validate_file_uploads', 
    'validate_processing_configuration',
    'prepare_session_state_data',
    'update_session_state',
    'clear_session_selections',
    'handle_creation_success',
    'handle_creation_error',
    'handle_loading_success',
    'handle_loading_error',
    'handle_processing_success',
    'handle_processing_error',
    'handle_reprocessing_success'
]