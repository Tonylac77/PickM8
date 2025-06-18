"""
Molecular fingerprint calculations.
Handles Morgan (ECFP) and RDKit fingerprints for molecular structures.
"""

import logging
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

logger = logging.getLogger(__name__)


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