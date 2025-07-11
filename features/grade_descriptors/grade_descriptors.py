"""GRADE descriptor computation functions."""
import pandas as pd
import numpy as np
import logging
import tempfile
import os
from typing import List, Optional, Dict, Any, Tuple
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)

# Try to import CDPL/GRAIL
try:
    import CDPL.Chem as CDPLChem
    import CDPL.Biomol as CDPLBiomol
    import CDPL.Math as CDPLMath
    import CDPL.GRAIL as CDPLGRAIL
    GRADE_AVAILABLE = True
except ImportError:
    GRADE_AVAILABLE = False
    logger.warning("CDPL/GRAIL not available. Install CDPL to use GRADE descriptors.")

# Constants from the original script
LIG_ENV_MAX_RADIUS = 21.0
REMOVE_NON_STD_RESIDUES = True

def is_grade_available() -> bool:
    """Check if GRADE/CDPL is available for use."""
    return GRADE_AVAILABLE

def get_grade_descriptor_names(extended: bool = False) -> List[str]:
    """
    Get list of GRADE descriptor names.
    
    Args:
        extended: Whether to use extended descriptors
        
    Returns:
        List of descriptor column names
    """
    if not GRADE_AVAILABLE:
        return []
    
    try:
        if extended:
            descr_calc = CDPLGRAIL.GRAILXDescriptorCalculator()
        else:
            descr_calc = CDPLGRAIL.GRAILDescriptorCalculator()
        
        return list(descr_calc.ElementIndex.names.keys())
    except Exception as e:
        logger.error(f"Error getting GRADE descriptor names: {e}")
        return []

def remove_non_std_residues(protein):
    """
    Remove non-standard residues from protein structure.
    
    Args:
        protein: CDPL protein molecule
    """
    residues = CDPLBiomol.ResidueList(protein)
    
    for res in residues:
        is_std_res = CDPLBiomol.ResidueDictionary.isStdResidue(CDPLBiomol.getResidueCode(res))
        
        if is_std_res and res.numAtoms < 5:
            logger.warning(f"Isolated standard residue fragment of size {res.numAtoms} found")
            protein -= res
            
        elif REMOVE_NON_STD_RESIDUES and not is_std_res:
            if res.numAtoms == 1 and CDPLChem.AtomDictionary.isMetal(CDPLChem.getType(res.atoms[0])):
                continue
            protein -= res

    CDPLChem.clearSSSR(protein)

def check_protein(protein):
    """
    Check protein structure for issues.
    
    Args:
        protein: CDPL protein molecule
    """
    for atom in protein.atoms:
        if CDPLChem.getType(atom) == CDPLChem.AtomType.H and atom.numAtoms == 0:
            logger.warning("Isolated hydrogen atom encountered")
             
        elif CDPLChem.getType(atom) == CDPLChem.AtomType.UNKNOWN:
            logger.warning("Atom of unknown element encountered")

def load_protein_from_pdb_content(pdb_content: str, normalize_charges: bool = False):
    """
    Load protein from PDB content string.
    
    Args:
        pdb_content: PDB file content as string
        normalize_charges: Whether to normalize charges for pH 7
        
    Returns:
        CDPL protein molecule
    """
    # Create temporary PDB file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_file:
        tmp_file.write(pdb_content)
        tmp_file_path = tmp_file.name
    
    try:
        pdb_reader = CDPLBiomol.FilePDBMoleculeReader(tmp_file_path)
        protein = CDPLChem.BasicMolecule()

        if not pdb_reader.read(protein):
            raise RuntimeError("Failed to read PDB content")
        
        check_protein(protein)
        remove_non_std_residues(protein)
        
        CDPLGRAIL.prepareForGRAILDescriptorCalculation(protein, normalize_charges)
        
        return protein
        
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def rdkit_mol_to_cdpl_mol(rdkit_mol: Chem.Mol):
    """
    Convert RDKit molecule to CDPL molecule.
    
    Args:
        rdkit_mol: RDKit molecule object
        
    Returns:
        CDPL molecule object
    """
    # Convert RDKit mol to MOL block
    mol_block = Chem.MolToMolBlock(rdkit_mol)
    
    # Create temporary MOL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mol', delete=False) as tmp_file:
        tmp_file.write(mol_block)
        tmp_file_path = tmp_file.name
    
    try:
        # Read with CDPL
        mol_reader = CDPLChem.FileMOLMoleculeReader(tmp_file_path)
        cdpl_mol = CDPLChem.BasicMolecule()
        
        if not mol_reader.read(cdpl_mol):
            raise RuntimeError("Failed to convert RDKit molecule to CDPL")
        
        return cdpl_mol
        
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def compute_grade_descriptors(
    ligand_mol: Chem.Mol,
    protein_content: str,
    extended: bool = False,
    normalize_charges: bool = False
) -> Optional[Dict[str, float]]:
    """
    Compute GRADE descriptors for a ligand-protein complex.
    
    Args:
        ligand_mol: RDKit molecule object for the ligand
        protein_content: PDB content as string
        extended: Whether to use extended descriptors
        normalize_charges: Whether to normalize charges for pH 7
        
    Returns:
        Dictionary of descriptor name -> value, or None if failed
    """
    if not GRADE_AVAILABLE or ligand_mol is None:
        return None
    
    try:
        # Load protein
        protein = load_protein_from_pdb_content(protein_content, normalize_charges)
        
        # Convert ligand to CDPL
        ligand = rdkit_mol_to_cdpl_mol(ligand_mol)
        
        # Prepare ligand for GRAIL calculation
        CDPLGRAIL.prepareForGRAILDescriptorCalculation(ligand, normalize_charges)
        
        # Extract ligand environment
        lig_env = CDPLChem.Fragment()
        CDPLBiomol.extractEnvironmentResidues(
            ligand, protein, lig_env, 
            CDPLChem.Atom3DCoordinatesFunctor(), 
            LIG_ENV_MAX_RADIUS, 
            False
        )
        CDPLChem.extractSSSRSubset(protein, lig_env, True)
        
        # Create descriptor calculator
        if extended:
            descr_calc = CDPLGRAIL.GRAILXDescriptorCalculator()
        else:
            descr_calc = CDPLGRAIL.GRAILDescriptorCalculator()
        
        # Calculate descriptors
        descr = CDPLMath.DVector()
        lig_atom_coords = CDPLMath.Vector3DArray()
        
        CDPLChem.get3DCoordinates(ligand, lig_atom_coords)
        
        descr_calc.initTargetData(lig_env, CDPLChem.Atom3DCoordinatesFunctor())
        descr_calc.initLigandData(ligand)
        descr_calc.calculate(lig_atom_coords, descr)
        
        # Convert to dictionary
        descriptor_names = list(descr_calc.ElementIndex.names.keys())
        descriptors = {}
        
        for i, name in enumerate(descriptor_names):
            if i < descr_calc.TOTAL_DESCRIPTOR_SIZE:
                descriptors[f"grade_{name}"] = float(descr(i))
        
        return descriptors
        
    except Exception as e:
        logger.error(f"Error computing GRADE descriptors: {e}")
        return None

def _compute_grade_worker(args):
    """Worker function for parallel GRADE computation."""
    idx, mol, protein_content, config = args
    
    if mol is None:
        return idx, None
    
    # Compute GRADE descriptors
    descriptors = compute_grade_descriptors(
        mol,
        protein_content,
        extended=config.get('extended', False),
        normalize_charges=config.get('normalize_charges', False)
    )
    
    return idx, descriptors

def compute_all_grade_descriptors(
    df: pd.DataFrame,
    protein_content: str,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Compute GRADE descriptors for all molecules in DataFrame.
    
    Args:
        df: DataFrame with molecules
        protein_content: PDB content as string
        config: GRADE configuration
        
    Returns:
        DataFrame with GRADE descriptor columns added
    """
    if not GRADE_AVAILABLE:
        logger.warning("GRADE not available, skipping descriptor computation")
        return df
    
    df = df.copy()
    
    # Get descriptor names for column initialization
    descriptor_names = get_grade_descriptor_names(config.get('extended', False))
    if not descriptor_names:
        logger.warning("Could not get GRADE descriptor names")
        return df
    
    # Initialize descriptor columns
    for name in descriptor_names:
        col_name = f"grade_{name}"
        df[col_name] = np.nan
    
    # Prepare data for parallel processing
    valid_molecules = []
    for idx, row in df.iterrows():
        if row['mol'] is not None:
            valid_molecules.append((idx, row['mol'], protein_content, config))
    
    if not valid_molecules:
        return df
    
    # Determine number of workers
    max_workers = config.get('max_workers', min(mp.cpu_count(), len(valid_molecules)))
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_compute_grade_worker, mol_data): mol_data[0] 
            for mol_data in valid_molecules
        }
        
        for future in as_completed(future_to_idx):
            try:
                idx, descriptors = future.result()
                if descriptors:
                    for desc_name, value in descriptors.items():
                        df.at[idx, desc_name] = value
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Error computing GRADE descriptors for molecule {idx}: {e}")
    
    logger.info(f"Computed GRADE descriptors for {len(valid_molecules)} molecules using {max_workers} workers")
    return df

def create_default_grade_config() -> Dict[str, Any]:
    """
    Create default configuration for GRADE descriptor calculation.
    
    Returns:
        Default GRADE configuration
    """
    return {
        "enabled": False,  # Disabled by default
        "extended": False,  # Use standard descriptors
        "normalize_charges": False,  # Don't normalize charges by default
        "max_workers": min(mp.cpu_count(), 8)  # Conservative default
    }