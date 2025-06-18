#!/usr/bin/env python3
"""
Functional interface for protein-ligand interaction fingerprint calculations.
Replaces the class-based InteractionWrapper with pure functions.
"""

import numpy as np
import yaml
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Dict, Optional, Union

from .plip import (
    calculate_plip_interactions,
    get_plip_interaction_types,
    is_plip_available
)
from .prolif import (
    calculate_prolif_interactions,
    get_prolif_interaction_types,
    is_prolif_available
)

logger = logging.getLogger(__name__)


def determine_ifp_type(ifp_type: Optional[str] = None) -> str:
    """
    Determine the IFP type to use based on parameter or config.
    
    Args:
        ifp_type: Explicitly specified IFP type
        
    Returns:
        IFP type string ('PLIP' or 'PROLIF')
    """
    if ifp_type is not None:
        return ifp_type
    
    # Try to load from main config file
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        fp_config = config.get('fingerprinting', {})
        return fp_config.get('default_type', 'PLIP')
    
    return 'PLIP'  # default fallback


def load_interaction_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration for interaction calculations.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def validate_ifp_type(ifp_type: str) -> None:
    """
    Validate and check availability of the requested IFP type.
    
    Args:
        ifp_type: IFP type to validate
        
    Raises:
        ValueError: If invalid IFP type
        ImportError: If required library is not available
    """
    # Normalize to uppercase for case-insensitive matching
    ifp_type = ifp_type.upper()
    valid_types = ['PLIP', 'PROLIF']
    if ifp_type not in valid_types:
        raise ValueError(f"Invalid IFP type: {ifp_type}. Must be one of: {valid_types}")
    
    if ifp_type == "PLIP" and not is_plip_available():
        raise ImportError("PLIP is not available. Install with: pip install plip")
    elif ifp_type == "PROLIF" and not is_prolif_available():
        raise ImportError("ProLIF is not available. Install with: pip install prolif")


def calculate_interactions(
    protein_path: str,
    ligand_mol: 'Chem.Mol',
    ligand_name: str = "LIG",
    ifp_type: Optional[str] = None,
    config: Optional[dict] = None
) -> Tuple[np.ndarray, dict]:
    """
    Calculate interaction fingerprints using either PLIP or ProLIF.
    
    Args:
        protein_path: Path to protein PDB file
        ligand_mol: RDKit molecule object
        ligand_name: Name for the ligand
        ifp_type: Type of interaction fingerprint ('PLIP' or 'PROLIF')
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (interaction_fingerprint, interaction_summary)
        
    Raises:
        ValueError: If unsupported IFP type
        ImportError: If required library is not available
    """
    if ifp_type is None:
        ifp_type = determine_ifp_type()
    
    # Normalize to uppercase for case-insensitive matching
    ifp_type = ifp_type.upper()
    validate_ifp_type(ifp_type)
    
    if ifp_type == "PLIP":
        return calculate_plip_interactions(protein_path, ligand_mol, ligand_name)
    elif ifp_type == "PROLIF":
        return calculate_prolif_interactions(protein_path, ligand_mol, ligand_name)
    else:
        raise ValueError(f"Unsupported IFP type: {ifp_type}")


def get_supported_interaction_types(ifp_type: Optional[str] = None) -> List[str]:
    """
    Get list of interaction types supported by the specified method.
    Alias for get_available_interaction_types for consistency.
    
    Args:
        ifp_type: IFP type to query
        
    Returns:
        List of supported interaction type names
    """
    return get_available_interaction_types(ifp_type)


def get_available_interaction_types(ifp_type: Optional[str] = None) -> List[str]:
    """
    Get list of interaction types supported by the specified method.
    
    Args:
        ifp_type: IFP type to query
        
    Returns:
        List of supported interaction type names
    """
    if ifp_type is None:
        ifp_type = determine_ifp_type()
    
    # Normalize to uppercase for case-insensitive matching
    ifp_type = ifp_type.upper()
    
    if ifp_type == "PLIP":
        return get_plip_interaction_types()
    elif ifp_type == "PROLIF":
        return get_prolif_interaction_types()
    else:
        return []


def calculate_interactions_batch(
    protein_path: str,
    molecules: List['Chem.Mol'],
    ifp_type: Optional[str] = None,
    max_workers: Optional[int] = None,
    config: Optional[dict] = None
) -> Tuple[List[np.ndarray], List[dict], Dict[int, str]]:
    """
    Calculate interaction fingerprints for multiple molecules in parallel.
    
    Args:
        protein_path: Path to protein PDB file
        molecules: List of RDKit molecule objects
        ifp_type: Type of interaction fingerprint
        max_workers: Maximum number of parallel workers
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (ifp_list, interactions_list, errors)
    """
    if ifp_type is None:
        ifp_type = determine_ifp_type()
    
    # Normalize to uppercase for case-insensitive matching
    ifp_type = ifp_type.upper()
    validate_ifp_type(ifp_type)
    
    if max_workers is None:
        import multiprocessing as mp
        # Use fewer workers for IFP calculations due to I/O overhead
        max_workers = min(mp.cpu_count() // 2, max(1, min(4, len(molecules) // 5)))
    
    logger.info(f"Calculating interactions for {len(molecules)} molecules using {max_workers} workers")
    
    def calculate_single_interaction(mol_data):
        idx, mol, name = mol_data
        try:
            ifp, interactions = calculate_interactions(
                protein_path, mol, name, ifp_type, config
            )
            return idx, ifp, interactions, None
        except Exception as e:
            logger.error(f"Error calculating interactions for molecule {idx} ({name}): {str(e)}")
            # Return default values on error
            default_ifp = np.zeros(1024, dtype=int)
            default_interactions = {
                'total_interactions': 0,
                'interaction_types': {},
                'interactions': []
            }
            return idx, default_ifp, default_interactions, str(e)
    
    # Create indexed molecule data
    indexed_molecules = [
        (i, mol, f"mol_{i}") for i, mol in enumerate(molecules)
    ]
    
    ifp_results = [None] * len(molecules)
    interaction_results = [None] * len(molecules)
    errors = {}
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(calculate_single_interaction, mol_data): mol_data[0]
            for mol_data in indexed_molecules
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, ifp, interactions, error = future.result()
            ifp_results[idx] = ifp
            interaction_results[idx] = interactions
            if error:
                errors[idx] = error
    
    if errors:
        logger.warning(f"Errors occurred for {len(errors)} molecules during IFP calculation")
    
    return ifp_results, interaction_results, errors


def create_interaction_context(
    ifp_type: Optional[str] = None,
    config_path: Optional[str] = None
) -> dict:
    """
    Create a context dictionary for interaction calculations.
    This replaces the need for class instantiation.
    
    Args:
        ifp_type: Type of interaction fingerprint
        config_path: Path to configuration file
        
    Returns:
        Context dictionary containing configuration
    """
    if ifp_type is None:
        ifp_type = determine_ifp_type()
    
    # Normalize to uppercase for case-insensitive matching
    ifp_type = ifp_type.upper()
    validate_ifp_type(ifp_type)
    
    config = load_interaction_config(config_path)
    
    return {
        'ifp_type': ifp_type,
        'config': config,
        'available_types': get_available_interaction_types(ifp_type)
    }


def set_interaction_type(context: dict, ifp_type: str) -> dict:
    """
    Update interaction context with new IFP type.
    
    Args:
        context: Existing context dictionary
        ifp_type: New IFP type
        
    Returns:
        Updated context dictionary
    """
    # Normalize to uppercase for case-insensitive matching
    ifp_type = ifp_type.upper()
    validate_ifp_type(ifp_type)
    
    context = context.copy()
    context['ifp_type'] = ifp_type
    context['available_types'] = get_available_interaction_types(ifp_type)
    
    return context


# Convenience functions for backward compatibility
def get_interaction_context(**kwargs) -> dict:
    """Get interaction context - alias for create_interaction_context."""
    return create_interaction_context(**kwargs)


def calculate_with_context(
    context: dict,
    protein_path: str,
    ligand_mol: 'Chem.Mol',
    ligand_name: str = "LIG"
) -> Tuple[np.ndarray, dict]:
    """
    Calculate interactions using a context dictionary.
    
    Args:
        context: Context dictionary from create_interaction_context
        protein_path: Path to protein PDB file
        ligand_mol: RDKit molecule object
        ligand_name: Name for the ligand
        
    Returns:
        Tuple of (interaction_fingerprint, interaction_summary)
    """
    return calculate_interactions(
        protein_path,
        ligand_mol,
        ligand_name,
        context['ifp_type'],
        context['config']
    )


def calculate_batch_with_context(
    context: dict,
    protein_path: str,
    molecules: List['Chem.Mol'],
    max_workers: Optional[int] = None
) -> Tuple[List[np.ndarray], List[dict], Dict[int, str]]:
    """
    Calculate batch interactions using a context dictionary.
    
    Args:
        context: Context dictionary from create_interaction_context
        protein_path: Path to protein PDB file
        molecules: List of RDKit molecule objects
        max_workers: Maximum number of parallel workers
        
    Returns:
        Tuple of (ifp_list, interactions_list, errors)
    """
    return calculate_interactions_batch(
        protein_path,
        molecules,
        context['ifp_type'],
        max_workers,
        context['config']
    )