"""
Fingerprint calculation modules for PickM8.
"""

from .molecular import compute_morgan_fingerprint, compute_rdkit_fingerprint, compute_mapchiral_fingerprint, is_mapchiral_available
from .interaction_wrapper import compute_interaction_fingerprint, create_default_interaction_config
from .config import create_default_fingerprint_config

# Import core interaction functions for backward compatibility
try:
    from .interactions.functions import calculate_interactions
    CORE_INTERACTIONS_AVAILABLE = True
except ImportError:
    CORE_INTERACTIONS_AVAILABLE = False

__all__ = [
    'compute_morgan_fingerprint',
    'compute_rdkit_fingerprint',
    'compute_mapchiral_fingerprint',
    'is_mapchiral_available',
    'compute_interaction_fingerprint',
    'create_default_fingerprint_config',
    'create_default_interaction_config'
]

# Add core interaction functions if available
if CORE_INTERACTIONS_AVAILABLE:
    __all__.append('calculate_interactions')