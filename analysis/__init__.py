"""
Analysis module for PickM8 molecular screening application.

This module    # Similarity analysis
    'get_enabled_fingerprint_columns',
    'concatenate_fingerprints',
    'calculate_tanimoto_similarity',
    'calculate_mol_similarity',
    'find_most_similar_molecule',
    'find_most_similar_molecule_by_mol',
    'get_similarity_statistics',
    'find_similar_molecules',
    'validate_fingerprint_data',s functionality for analyzing molecular data including:
- Manual grading workflows and statistics
- Pose quality analysis using PoseCheck
- Dataset statistics and correlation analysis
- Molecule filtering and sorting strategies

All functions follow functional programming principles with pure functions
that return new DataFrames without mutating input data.
"""

from .grading import (
    # Grading functions
    add_grade,
    get_graded_molecules,
    get_ungraded_molecules,
    get_grading_statistics,
    filter_and_sort_molecules,
    get_molecules_by_strategy,
    reset_all_grades,
    cleanup_model_metadata,
    
    # Utility functions
    has_trained_model,
)

from .pose_quality import (
    # Pose quality analysis
    analyze_single_pose,
    analyze_all_poses,
    get_pose_quality_statistics,
)

from .statistics import (
    # Statistical analysis
    calculate_dataset_statistics,
    calculate_correlation_matrix,
)

from .similarity import (
    # Similarity analysis
    get_enabled_fingerprint_columns,
    concatenate_fingerprints,
    calculate_tanimoto_similarity,
    calculate_mol_similarity,
    find_most_similar_molecule,
    find_most_similar_molecule_by_mol,
    get_similarity_statistics,
    find_similar_molecules,
    validate_fingerprint_data,
)

# Define public API
__all__ = [
    # Grading functions
    'add_grade',
    'get_graded_molecules',
    'get_ungraded_molecules', 
    'get_grading_statistics',
    'filter_and_sort_molecules',
    'get_molecules_by_strategy',
    'reset_all_grades',
    'cleanup_model_metadata',
    'has_trained_model',
    
    # Pose quality functions
    'analyze_single_pose',
    'analyze_all_poses',
    'get_pose_quality_statistics',
    
    # Statistical functions
    'calculate_dataset_statistics',
    'calculate_correlation_matrix',

    # Similarity functions
    'get_enabled_fingerprint_columns',
    'concatenate_fingerprints',
    'calculate_tanimoto_similarity',
    'find_most_similar_molecule',
    'get_similarity_statistics',
    'find_similar_molecules',
    'validate_fingerprint_data',
]