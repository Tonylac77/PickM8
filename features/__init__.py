"""
Features module for molecular feature extraction and computation.

This module provides a clean API for computing various molecular features
including fingerprints, protein-ligand interactions, and GRADE descriptors.
"""

# Import key functions from each submodule
from .fingerprints import (
    compute_mapchiral_fingerprint,
    compute_all_fingerprints,
    get_fingerprint_statistics,
    is_mapchiral_available
)

from .interactions import (
    compute_interaction_fingerprint,
    compute_all_interactions,
    create_default_interaction_config,
    calculate_plip_interactions,
    calculate_prolif_interactions,
    is_plip_available,
    is_prolif_available,
    get_plip_interaction_types,
    get_prolif_interaction_types,
    create_complex_with_biopython
)

from .grade_descriptors import (
    compute_grade_descriptors,
    compute_all_grade_descriptors,
    create_default_grade_config,
    get_grade_descriptor_names,
    is_grade_available,
    load_protein_from_pdb_content,
    rdkit_mol_to_cdpl_mol
)

# Export all public functions
__all__ = [
    # Fingerprints
    'compute_morgan_fingerprint',
    'compute_rdkit_fingerprint',
    'compute_mapchiral_fingerprint',
    'compute_all_fingerprints',
    'get_fingerprint_statistics',
    'is_mapchiral_available',
    
    # Interactions  
    'compute_interaction_fingerprint',
    'compute_all_interactions',
    'create_default_interaction_config',
    'calculate_plip_interactions',
    'calculate_prolif_interactions',
    'is_plip_available',
    'is_prolif_available',
    'get_plip_interaction_types',
    'get_prolif_interaction_types',
    'create_complex_with_biopython',
    
    # GRADE descriptors
    'compute_grade_descriptors',
    'compute_all_grade_descriptors',
    'create_default_grade_config',
    'get_grade_descriptor_names',
    'is_grade_available',
    'load_protein_from_pdb_content',
    'rdkit_mol_to_cdpl_mol'
]