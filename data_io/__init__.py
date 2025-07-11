"""
PickM8 Data I/O Module - Clean public API for molecular data operations.

This module provides a unified interface for:
- Loading and processing molecular data
- DataFrame schema management
- Molecule operations

The module is organized into submodules:
- molecules: Core molecular data operations
- exporters: Data export utilities (placeholder)
"""

# Core molecule operations
from .molecules import (
    create_empty_dataframe,
    load_sdf,
    process_score_column,
    detect_sdf_properties,
    add_grade_columns,
    save_molecules_dataframe,
    load_molecules_dataframe,
    add_grade_to_molecule
)

# Public API - most commonly used functions
__all__ = [
    # Core molecule operations
    'create_empty_dataframe',
    'load_sdf',
    'process_score_column',
    'detect_sdf_properties',
    'add_grade_columns',
    'save_molecules_dataframe',
    'load_molecules_dataframe',
    'add_grade_to_molecule'
]