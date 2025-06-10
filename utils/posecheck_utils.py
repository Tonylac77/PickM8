import tempfile
from pathlib import Path
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)

class PoseCheckAnalyzer:
    """Wrapper for PoseCheck functionality"""
    
    def __init__(self):
        self.pc = None
        self.protein_loaded = False
        
    def load_protein_from_content(self, protein_content):
        """Load protein from PDB content string"""
        try:
            from posecheck import PoseCheck
            
            self.pc = PoseCheck()
            
            # Create temporary PDB file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                tmp.write(protein_content)
                tmp_path = tmp.name
            
            # Load protein
            self.pc.load_protein_from_pdb(tmp_path)
            self.protein_loaded = True
            
            # Clean up
            Path(tmp_path).unlink()
            
            logger.info("PoseCheck: Protein loaded successfully")
            
        except ImportError:
            logger.warning("PoseCheck not available - install with: pip install posecheck")
            self.pc = None
            self.protein_loaded = False
        except Exception as e:
            logger.error(f"Failed to load protein in PoseCheck: {str(e)}")
            self.pc = None
            self.protein_loaded = False
    
    def analyze_molecule(self, mol_block):
        """Analyze a single molecule and return clashes and strain energy"""
        if not self.protein_loaded or self.pc is None:
            return 0, 0.0
        
        try:
            # Convert mol_block to RDKit molecule
            mol = Chem.MolFromMolBlock(mol_block)
            if mol is None:
                return 0, 0.0
            
            # Load molecule into PoseCheck
            self.pc.load_ligands_from_mols([mol])
            
            # Calculate clashes
            clashes = self.pc.calculate_clashes()
            clash_count = clashes[0] if clashes else 0
            
            # Calculate strain energy
            strain = self.pc.calculate_strain_energy()
            strain_energy = strain[0] if strain else 0.0
            
            return clash_count, strain_energy
            
        except Exception as e:
            logger.warning(f"PoseCheck analysis failed: {str(e)}")
            return 0, 0.0
    
    def analyze_multiple_molecules(self, mol_blocks):
        """Analyze multiple molecules and return lists of clashes and strain energies"""
        if not self.protein_loaded or self.pc is None:
            return [0] * len(mol_blocks), [0.0] * len(mol_blocks)
        
        try:
            # Convert mol_blocks to RDKit molecules
            mols = []
            for mol_block in mol_blocks:
                mol = Chem.MolFromMolBlock(mol_block)
                if mol is not None:
                    mols.append(mol)
                else:
                    mols.append(None)
            
            # Filter out None molecules
            valid_mols = [mol for mol in mols if mol is not None]
            
            if not valid_mols:
                return [0] * len(mol_blocks), [0.0] * len(mol_blocks)
            
            # Load molecules into PoseCheck
            self.pc.load_ligands_from_mols(valid_mols)
            
            # Calculate clashes
            clashes = self.pc.calculate_clashes()
            
            # Calculate strain energy
            strain = self.pc.calculate_strain_energy()
            
            # Map results back to original list (handling None molecules)
            clash_results = []
            strain_results = []
            valid_idx = 0
            
            for mol in mols:
                if mol is not None:
                    clash_results.append(clashes[valid_idx] if valid_idx < len(clashes) else 0)
                    strain_results.append(strain[valid_idx] if valid_idx < len(strain) else 0.0)
                    valid_idx += 1
                else:
                    clash_results.append(0)
                    strain_results.append(0.0)
            
            return clash_results, strain_results
            
        except Exception as e:
            logger.warning(f"PoseCheck batch analysis failed: {str(e)}")
            return [0] * len(mol_blocks), [0.0] * len(mol_blocks)