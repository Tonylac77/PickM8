import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
import tempfile
import os
import yaml

# PLIP imports
try:
    from plip.structure.preparation import PDBComplex
    from plip.exchange.report import BindingSiteReport
    PLIP_AVAILABLE = True
except ImportError:
    PLIP_AVAILABLE = False

# ProLIF imports  
try:
    import prolif as plf
    import MDAnalysis as mda
    PROLIF_AVAILABLE = True
except ImportError:
    PROLIF_AVAILABLE = False

class InteractionWrapper:
    """Unified wrapper for PLIP and ProLIF interaction fingerprint calculations"""
    
    def __init__(self, config_path=None, ifp_type=None):
        self.ifp_type = self._determine_ifp_type(ifp_type)
        self.config = self._load_config(config_path)
        
        # Validate availability of requested tool
        if self.ifp_type == "PLIP" and not PLIP_AVAILABLE:
            raise ImportError("PLIP is not available. Install with: pip install plip")
        elif self.ifp_type == "PROLIF" and not PROLIF_AVAILABLE:
            raise ImportError("ProLIF is not available. Install with: pip install prolif")
    
    def _determine_ifp_type(self, ifp_type):
        """Determine the IFP type to use based on parameter or config"""
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
    
    def _load_config(self, config_path):
        """Load configuration for interaction calculations"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def set_ifp_type(self, ifp_type):
        """Set the interaction fingerprint type"""
        valid_types = ['PLIP', 'PROLIF']
        if ifp_type not in valid_types:
            raise ValueError(f"Invalid IFP type: {ifp_type}. Must be one of: {valid_types}")
        
        # Validate availability
        if ifp_type == "PLIP" and not PLIP_AVAILABLE:
            raise ImportError("PLIP is not available")
        elif ifp_type == "PROLIF" and not PROLIF_AVAILABLE:
            raise ImportError("ProLIF is not available")
            
        self.ifp_type = ifp_type
    
    def calculate_interactions(self, protein_path, ligand_mol, ligand_name="ligand"):
        """Calculate interaction fingerprints using either PLIP or ProLIF"""
        if self.ifp_type == "PLIP":
            return self._calculate_plip_interactions(protein_path, ligand_mol, ligand_name)
        elif self.ifp_type == "PROLIF":
            return self._calculate_prolif_interactions(protein_path, ligand_mol, ligand_name)
        else:
            raise ValueError(f"Unsupported IFP type: {self.ifp_type}")
    
    def _calculate_plip_interactions(self, protein_path, ligand_mol, ligand_name):
        """Calculate interactions using PLIP"""
        if not PLIP_AVAILABLE:
            raise ImportError("PLIP is not available")
        
        # Create temporary PDB file with ligand
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb:
            # Read protein and clean residue numbers
            cleaned_protein = self._clean_pdb_for_plip(protein_path)
            
            # Convert ligand to PDB format
            ligand_pdb = Chem.MolToPDBBlock(ligand_mol)
            
            # Combine protein and ligand
            combined_pdb = cleaned_protein + "\n" + ligand_pdb
            tmp_pdb.write(combined_pdb)
            tmp_pdb_path = tmp_pdb.name
        
        try:
            # Initialize PLIP with error handling
            mol = PDBComplex()
            mol.load_pdb(tmp_pdb_path)
            mol.analyze()
            
            # Extract interactions
            interaction_data = []
            
            for site in mol.interaction_sets:
                site_interactions = mol.interaction_sets[site]
                
                # Process different interaction types
                hbonds = [(i.restype, i.resnr, i.reschain, 'hydrogen_bond') 
                         for i in site_interactions.hbonds_pdon + site_interactions.hbonds_ldon]
                hydrophobic = [(i.restype, i.resnr, i.reschain, 'hydrophobic') 
                              for i in site_interactions.hydrophobic_contacts]
                pi_stacking = [(i.restype, i.resnr, i.reschain, 'pi_stacking') 
                              for i in site_interactions.pistacking]
                salt_bridges = [(i.restype, i.resnr, i.reschain, 'salt_bridge') 
                               for i in site_interactions.saltbridge_pneg + site_interactions.saltbridge_lneg]
                halogen_bonds = [(i.restype, i.resnr, i.reschain, 'halogen_bond') 
                                for i in site_interactions.halogen_bonds]
                
                all_interactions = hbonds + hydrophobic + pi_stacking + salt_bridges + halogen_bonds
                interaction_data.extend(all_interactions)
            
            # Convert to interaction fingerprint
            ifp = self._interactions_to_fingerprint(interaction_data)
            
            # Create interaction summary
            interactions_summary = {
                'total_interactions': len(interaction_data),
                'interaction_types': {
                    'hydrogen_bonds': len([i for i in interaction_data if i[3] == 'hydrogen_bond']),
                    'hydrophobic': len([i for i in interaction_data if i[3] == 'hydrophobic']),
                    'pi_stacking': len([i for i in interaction_data if i[3] == 'pi_stacking']),
                    'salt_bridges': len([i for i in interaction_data if i[3] == 'salt_bridge']),
                    'halogen_bonds': len([i for i in interaction_data if i[3] == 'halogen_bond'])
                },
                'interactions': interaction_data
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_pdb_path)
        
        return ifp, interactions_summary
    
    def _calculate_prolif_interactions(self, protein_path, ligand_mol, ligand_name):
        """Calculate interactions using ProLIF"""
        if not PROLIF_AVAILABLE:
            raise ImportError("ProLIF is not available")
        
        # Create temporary SDF file for ligand
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp_sdf:
            sdf_writer = Chem.SDWriter(tmp_sdf.name)
            sdf_writer.write(ligand_mol)
            sdf_writer.close()
            tmp_sdf_path = tmp_sdf.name
        
        try:
            # Load structures with MDAnalysis
            protein_u = mda.Universe(protein_path)
            ligand_u = mda.Universe(tmp_sdf_path)
            
            # Create ProLIF fingerprint
            fp = plf.Fingerprint()
            
            # Calculate fingerprint
            ifp_df = fp.run(ligand_u.trajectory, ligand_u, protein_u)
            
            # Convert to numpy array format
            ifp_array = ifp_df.values.flatten()
            
            # Extract interaction details
            interaction_data = []
            interactions_summary = {
                'total_interactions': int(np.sum(ifp_array > 0)),
                'interaction_types': {},
                'interactions': interaction_data
            }
            
            # Process fingerprint data
            for col in ifp_df.columns:
                if ifp_df[col].iloc[0] > 0:
                    # Parse column name to get interaction info
                    interaction_type = col[1] if len(col) > 1 else 'unknown'
                    residue_info = col[0] if len(col) > 0 else 'unknown'
                    interaction_data.append((residue_info, 0, '', interaction_type))
            
            # Group by interaction type
            for interaction in interaction_data:
                int_type = interaction[3]
                if int_type in interactions_summary['interaction_types']:
                    interactions_summary['interaction_types'][int_type] += 1
                else:
                    interactions_summary['interaction_types'][int_type] = 1
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_sdf_path)
        
        return ifp_array, interactions_summary
    
    def _interactions_to_fingerprint(self, interaction_data, fp_size=1024):
        """Convert interaction data to binary fingerprint"""
        # Create a simple hash-based fingerprint
        ifp = np.zeros(fp_size, dtype=int)
        
        for interaction in interaction_data:
            restype, resnr, reschain, int_type = interaction
            # Create hash from interaction description
            interaction_str = f"{restype}_{resnr}_{reschain}_{int_type}"
            hash_val = hash(interaction_str) % fp_size
            ifp[hash_val] = 1
        
        return ifp
    
    def _clean_pdb_for_plip(self, protein_path):
        """Clean PDB file to avoid PLIP issues with negative residue numbers"""
        cleaned_lines = []
        residue_mapping = {}
        current_residue_num = 1
        
        with open(protein_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    # Extract residue number (columns 23-26)
                    try:
                        orig_resnum = int(line[22:26].strip())
                        chain = line[21:22]
                        key = (chain, orig_resnum)
                        
                        # Map negative or problematic residue numbers to positive ones
                        if key not in residue_mapping:
                            if orig_resnum < 1 or orig_resnum > 9999:
                                residue_mapping[key] = current_residue_num
                                current_residue_num += 1
                            else:
                                residue_mapping[key] = orig_resnum
                                # Update current_residue_num to avoid conflicts
                                if orig_resnum >= current_residue_num:
                                    current_residue_num = orig_resnum + 1
                        
                        new_resnum = residue_mapping[key]
                        
                        # Reconstruct line with corrected residue number
                        new_line = (line[:22] + 
                                   f"{new_resnum:4d}" + 
                                   line[26:])
                        cleaned_lines.append(new_line)
                    except (ValueError, IndexError):
                        # If we can't parse the residue number, keep the original line
                        cleaned_lines.append(line)
                else:
                    # Keep non-ATOM/HETATM lines as is
                    cleaned_lines.append(line)
        
        return ''.join(cleaned_lines)
    
    def get_available_interaction_types(self):
        """Get list of interaction types supported by current method"""
        if self.ifp_type == "PLIP":
            return ['hydrogen_bond', 'hydrophobic', 'pi_stacking', 'salt_bridge', 'halogen_bond']
        elif self.ifp_type == "PROLIF":
            # ProLIF supports many interaction types
            return ['HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiStacking', 'Anionic', 'Cationic', 'CationPi', 'PiCation']
        else:
            return []