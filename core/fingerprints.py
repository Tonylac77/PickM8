import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import DataStructs
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

class FingerprintHandler:
    def __init__(self, fp_type='morgan', fp_size=2048, radius=2, interaction_fp_type='PLIP'):
        self.fp_type = fp_type
        self.fp_size = fp_size
        self.radius = radius
        self.interaction_fp_type = interaction_fp_type
        
    @classmethod
    def from_config(cls, config_path="config.yaml"):
        """Create FingerprintHandler from configuration file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            fp_config = config.get('fingerprinting', {})
            return cls(
                fp_type=fp_config.get('molecule_fp_type', 'morgan'),
                fp_size=fp_config.get('molecule_fp_size', 2048),
                radius=fp_config.get('molecule_fp_radius', 2),
                interaction_fp_type=fp_config.get('default_type', 'PLIP')
            )
        else:
            return cls()
    
    def compute_fingerprint(self, mol):
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        
        if self.fp_type == 'morgan':
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fp_size)
            fp = mfpgen.GetFingerprint(mol)
        elif self.fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, fpSize=self.fp_size)
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fp_type}")
        
        arr = np.zeros((self.fp_size,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    
    def ifp_to_array(self, ifp_dict, fp_length=4096):
        fp = np.zeros(fp_length)
        if isinstance(ifp_dict, dict):
            for key, value in ifp_dict.items():
                if int(key) < fp_length:
                    fp[int(key)] = value
        elif hasattr(ifp_dict, 'counts'):
            for key, value in ifp_dict.counts.items():
                if int(key) < fp_length:
                    fp[int(key)] = value
        return fp
    
    def combine_fingerprints(self, mol_fp, ifp_array):
        return np.concatenate([mol_fp, ifp_array])
    
    def get_interaction_fingerprint_type(self):
        """Get the current interaction fingerprint type"""
        return self.interaction_fp_type
    
    def set_interaction_fingerprint_type(self, fp_type):
        """Set the interaction fingerprint type"""
        if fp_type not in ['PLIP', 'PROLIF']:
            raise ValueError(f"Invalid fingerprint type: {fp_type}. Must be one of: PLIP, PROLIF")
        self.interaction_fp_type = fp_type
    
    def compute_fingerprints_batch(self, molecules, max_workers=None):
        """
        Compute fingerprints for multiple molecules in parallel
        
        Args:
            molecules: List of RDKit molecule objects or SMILES strings
            max_workers: Maximum number of parallel workers (None for auto-detection)
            
        Returns:
            Tuple of (fingerprint_list, errors_dict)
        """
        if max_workers is None:
            import multiprocessing as mp
            # Auto-detect optimal worker count for CPU-bound tasks
            max_workers = min(mp.cpu_count(), max(2, len(molecules) // 10))
        
        logger.info(f"Computing fingerprints for {len(molecules)} molecules using {max_workers} workers")
        
        def compute_single_fp(mol_data):
            idx, mol = mol_data
            try:
                fp = self.compute_fingerprint(mol)
                return idx, fp, None
            except Exception as e:
                logger.error(f"Error computing fingerprint for molecule {idx}: {str(e)}")
                return idx, None, str(e)
        
        # Create indexed molecule data for parallel processing
        indexed_molecules = [(i, mol) for i, mol in enumerate(molecules)]
        
        results = [None] * len(molecules)
        errors = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(compute_single_fp, mol_data): mol_data[0] 
                for mol_data in indexed_molecules
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx, fp, error = future.result()
                if error is None:
                    results[idx] = fp
                else:
                    errors[idx] = error
                    # Use zero array as fallback
                    results[idx] = np.zeros(self.fp_size, dtype=np.int8)
        
        if errors:
            logger.warning(f"Errors occurred for {len(errors)} molecules: {errors}")
        
        return results, errors
    
    def process_molecules_batch(self, molecules, protein_path, interaction_context, max_fp_workers=None, max_ifp_workers=None):
        """
        Process molecules for both fingerprints and interactions in parallel
        
        Args:
            molecules: List of RDKit molecule objects
            protein_path: Path to protein PDB file
            interaction_context: Interaction context dictionary
            max_fp_workers: Maximum workers for fingerprint calculation
            max_ifp_workers: Maximum workers for interaction calculation
            
        Returns:
            Tuple of (fingerprints, ifp_results, interaction_results, all_errors)
        """
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor
        
        num_molecules = len(molecules)
        
        # Auto-detect optimal worker counts if not specified
        if max_fp_workers is None:
            max_fp_workers = min(mp.cpu_count(), max(2, num_molecules // 10))
        
        if max_ifp_workers is None:
            max_ifp_workers = min(mp.cpu_count() // 2, max(1, min(4, num_molecules // 5)))
        
        logger.info(f"Processing {num_molecules} molecules with {max_fp_workers} FP workers and {max_ifp_workers} IFP workers")
        
        # Start both fingerprint and interaction calculations in parallel
        with ThreadPoolExecutor(max_workers=max_fp_workers + max_ifp_workers) as executor:
            # Submit fingerprint calculation
            fp_future = executor.submit(
                self.compute_fingerprints_batch, 
                molecules, 
                max_fp_workers
            )
            
            # Submit interaction calculation
            from core.interaction_functions import calculate_batch_with_context
            ifp_future = executor.submit(
                calculate_batch_with_context,
                interaction_context,
                protein_path,
                molecules,
                max_ifp_workers
            )
            
            # Collect results
            fingerprints, fp_errors = fp_future.result()
            ifp_results, interaction_results, ifp_errors = ifp_future.result()
        
        # Combine all errors
        all_errors = {**fp_errors, **ifp_errors}
        
        logger.info(f"Batch processing complete: {len(fp_errors)} FP errors, {len(ifp_errors)} IFP errors")
        
        return fingerprints, ifp_results, interaction_results, all_errors

    # ...existing code...