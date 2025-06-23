"""
Molecular fingerprint calculations.
Handles Morgan (ECFP), RDKit, and MapChiral fingerprints for molecular structures.
"""

import logging
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)

# MapChiral imports
try:
    from mapchiral.mapchiral import encode as mapchiral_encode
    MAPCHIRAL_AVAILABLE = True
except ImportError:
    MAPCHIRAL_AVAILABLE = False
    logger.warning("MapChiral not available. Install with: pip install mapchiral")


def compute_morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> Optional[List[int]]:
    """
    Compute Morgan (ECFP) fingerprint for a molecule.
    
    Args:
        mol: RDKit molecule object
        radius: Fingerprint radius
        n_bits: Number of bits in fingerprint
        
    Returns:
        List of integers representing fingerprint bits, or None if error
    """
    try:
        if mol is None:
            return None
            
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return list(fp.ToBitString())
        
    except Exception as e:
        logger.error(f"Error computing Morgan fingerprint: {e}")
        return None


def compute_rdkit_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> Optional[List[int]]:
    """
    Compute RDKit fingerprint for a molecule.
    
    Args:
        mol: RDKit molecule object
        n_bits: Number of bits in fingerprint
        
    Returns:
        List of integers representing fingerprint bits, or None if error
    """
    try:
        if mol is None:
            return None
            
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        return [int(bit) for bit in fp.ToBitString()]
        
    except Exception as e:
        logger.error(f"Error computing RDKit fingerprint: {e}")
        return None


def compute_mapchiral_fingerprint(mol: Chem.Mol, max_radius: int = 2, n_permutations: int = 2048, 
                                 mapping: bool = False) -> Optional[List[int]]:
    """
    Compute MapChiral fingerprint for a molecule.
    
    Args:
        mol: RDKit molecule object
        max_radius: Maximum radius for fingerprint calculation
        n_permutations: Number of permutations (fingerprint size)
        mapping: Whether to return mapping information
        
    Returns:
        List of integers representing fingerprint bits, or None if error
    """
    try:
        if mol is None:
            return None
            
        if not MAPCHIRAL_AVAILABLE:
            logger.warning("MapChiral not available, returning None")
            return None
            
        fp = mapchiral_encode(mol, max_radius=max_radius, n_permutations=n_permutations, mapping=mapping)
        return fp.tolist() if hasattr(fp, 'tolist') else list(fp)
        
    except Exception as e:
        logger.error(f"Error computing MapChiral fingerprint: {e}")
        return None


def is_mapchiral_available() -> bool:
    """
    Check if MapChiral is available for use.
    
    Returns:
        True if MapChiral is available, False otherwise
    """
    return MAPCHIRAL_AVAILABLE