"""
Sessions module for PickM8 molecular screening application.

This module provides functionality for session management including:
- Session creation with molecular processing
- Session loading and saving
- Session listing and metadata management

The session system uses a functional programming approach with pure functions
that don't mutate input data. Sessions are persisted to disk using pickle
for DataFrames and JSON for metadata.
"""

from .sessions import (
    # Core session functions
    create_session,
    save_session,
    load_session,
    list_sessions,
    
    # Utility functions
    generate_session_id,
)

# Define public API
__all__ = [
    'create_session',
    'save_session', 
    'load_session',
    'list_sessions',
    'generate_session_id',
]