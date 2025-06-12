import tempfile
from pathlib import Path
from rdkit import Chem
import logging
import multiprocessing as mp
from functools import partial

logger = logging.getLogger(__name__)

def _analyze_molecule_chunk(protein_content, mol_blocks_chunk):
    """Worker function to analyze a chunk of molecules in a separate process"""
    try:
        from posecheck import PoseCheck
        
        # Create PoseCheck instance
        pc = PoseCheck()
        
        # Load protein
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            tmp.write(protein_content)
            tmp_path = tmp.name
        
        pc.load_protein_from_pdb(tmp_path)
        Path(tmp_path).unlink()
        
        results = []
        
        for mol_block in mol_blocks_chunk:
            try:
                if mol_block is None:
                    results.append((0, 0.0))
                    continue
                    
                mol = Chem.MolFromMolBlock(mol_block)
                if mol is None:
                    results.append((0, 0.0))
                    continue
                
                # Load molecule
                pc.load_ligands_from_mols([mol])
                
                # Calculate metrics
                clashes = pc.calculate_clashes()
                strain = pc.calculate_strain_energy()
                
                clash_count = clashes[0] if clashes else 0
                strain_energy = strain[0] if strain else 0.0
                
                results.append((clash_count, strain_energy))
                
            except Exception as e:
                logger.warning(f"PoseCheck analysis failed for molecule: {str(e)}")
                results.append((0, 0.0))
        
        return results
        
    except Exception as e:
        logger.error(f"Worker process failed: {str(e)}")
        return [(0, 0.0)] * len(mol_blocks_chunk)

class PoseCheckAnalyzer:
    """Wrapper for PoseCheck functionality"""
    
    def __init__(self):
        self.pc = None
        self.protein_loaded = False
        self.protein_content = None
        
    def load_protein_from_content(self, protein_content):
        """Load protein from PDB content string"""
        try:
            from posecheck import PoseCheck
            
            self.pc = PoseCheck()
            self.protein_content = protein_content  # Store for parallel processing
            
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
    
    def analyze_multiple_molecules_parallel(self, mol_blocks, n_workers=None, chunk_size=None):
        """Analyze multiple molecules in parallel and return lists of clashes and strain energies"""
        if not self.protein_loaded or self.pc is None or self.protein_content is None:
            return [0] * len(mol_blocks), [0.0] * len(mol_blocks)
        
        if len(mol_blocks) == 0:
            return [], []
        
        # Determine number of workers
        if n_workers is None:
            n_workers = min(mp.cpu_count(), max(1, len(mol_blocks) // 10))
        
        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(mol_blocks) // n_workers)
        
        # For small datasets, use sequential processing
        if len(mol_blocks) < 10 or n_workers == 1:
            return self.analyze_multiple_molecules(mol_blocks)
        
        try:
            # Split mol_blocks into chunks
            chunks = []
            for i in range(0, len(mol_blocks), chunk_size):
                chunks.append(mol_blocks[i:i + chunk_size])
            
            # Process chunks in parallel using partial function
            from functools import partial
            worker_func = partial(_analyze_molecule_chunk, self.protein_content)
            
            with mp.Pool(processes=n_workers) as pool:
                chunk_results = pool.map(worker_func, chunks)
            
            # Combine results
            clashes = []
            strain_energies = []
            
            for chunk_result in chunk_results:
                for clash, strain in chunk_result:
                    clashes.append(clash)
                    strain_energies.append(strain)
            
            return clashes, strain_energies
            
        except Exception as e:
            logger.warning(f"Parallel processing failed, falling back to sequential: {str(e)}")
            return self.analyze_multiple_molecules(mol_blocks)
    
    def analyze_multiple_molecules_smart(self, mol_blocks, parallel_threshold=50):
        """Automatically choose between parallel and sequential processing based on dataset size"""
        if len(mol_blocks) >= parallel_threshold:
            logger.info(f"Using parallel processing for {len(mol_blocks)} molecules")
            return self.analyze_multiple_molecules_parallel(mol_blocks)
        else:
            logger.info(f"Using sequential processing for {len(mol_blocks)} molecules")
            return self.analyze_multiple_molecules(mol_blocks)