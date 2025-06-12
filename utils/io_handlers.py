import polars as pl
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from rdkit import Chem
import gzip

class DataHandler:
    def __init__(self, session_id):
        self.session_id = session_id
        self.session_path = Path(f"data/sessions/{session_id}")
        self.session_path.mkdir(parents=True, exist_ok=True)
    
    def save_molecules(self, molecules_data):
        df = pl.DataFrame(molecules_data)
        df.write_parquet(self.session_path / "molecules.parquet")
    
    def save_grades(self, grades_data):
        df = pl.DataFrame(grades_data)
        df.write_parquet(self.session_path / "grades.parquet")
    
    def save_predictions(self, predictions_data):
        df = pl.DataFrame(predictions_data)
        df.write_parquet(self.session_path / "predictions.parquet")
    
    def load_molecules(self):
        path = self.session_path / "molecules.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None
    
    def load_grades(self):
        path = self.session_path / "grades.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None
    
    def load_predictions(self):
        path = self.session_path / "predictions.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None
    
    def save_session_state(self, state_dict):
        with open(self.session_path / "session_state.json", 'w') as f:
            json.dump(state_dict, f, default=str)
    
    def load_session_state(self):
        path = self.session_path / "session_state.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

class MoleculeReader:
    @staticmethod
    def get_sdf_properties(filepath, max_molecules=5):
        """
        Parse SDF file to extract available properties from the first few molecules.
        Returns a set of all property names found.
        """
        properties = set()
        
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rt') as f:
                    supplier = Chem.ForwardSDMolSupplier(f)
            else:
                supplier = Chem.ForwardSDMolSupplier(filepath)
            
            mol_count = 0
            for mol in supplier:
                if mol is None:
                    continue
                
                # Get all properties from this molecule
                all_props = mol.GetPropsAsDict()
                properties.update(all_props.keys())
                
                mol_count += 1
                if mol_count >= max_molecules:
                    break
                    
        except Exception as e:
            print(f"Error reading SDF file: {e}")
            return []
        
        # Filter out the _Name property as it's handled separately
        properties.discard('_Name')
        return sorted(list(properties))
    
    @staticmethod
    def read_sdf(filepath, score_label='score'):
        molecules = []
        
        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rt') as f:
                supplier = Chem.ForwardSDMolSupplier(f)
        else:
            supplier = Chem.ForwardSDMolSupplier(filepath)
        
        for i, mol in enumerate(supplier):
            if mol is None:
                continue
            
            try:
                # Get all properties from the molecule
                all_props = mol.GetPropsAsDict()
                
                mol_data = {
                    'id': i,
                    'name': mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{i}",
                    'mol_block': Chem.MolToMolBlock(mol),
                    'smiles': Chem.MolToSmiles(mol),
                    'score': float(mol.GetProp(score_label)) if mol.HasProp(score_label) else 0.0
                }
                
                # Add all other properties as additional columns
                for prop_name, prop_value in all_props.items():
                    if prop_name not in ['_Name', score_label]:
                        # Try to convert to appropriate type
                        try:
                            # Try float first
                            prop_value = float(prop_value)
                        except (ValueError, TypeError):
                            try:
                                # Try int
                                prop_value = int(prop_value)
                            except (ValueError, TypeError):
                                # Keep as string
                                pass
                        
                        # Store with prop_ prefix to avoid conflicts
                        mol_data[f'prop_{prop_name}'] = prop_value
                
                molecules.append(mol_data)
            except Exception as e:
                continue
        
        return molecules
    
    @staticmethod
    def read_pdb(filepath):
        with open(filepath, 'r') as f:
            return f.read()