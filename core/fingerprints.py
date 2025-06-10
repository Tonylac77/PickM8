import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import json
import yaml
from pathlib import Path

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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.fp_size)
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