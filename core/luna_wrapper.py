import configparser
import luna
from luna.mol.entry import MolFileEntry
from luna.mol.features import FeatureExtractor
from luna.mol.groups import AtomGroupPerceiver
from luna.config.params import ProjectParams
from luna.interaction.contact import get_contacts_with
from luna.interaction.calc import InteractionCalculator
from luna.interaction.fp.shell import ShellGenerator
from luna.interaction.fp.type import IFPType
from luna.util.default_values import *
from luna.MyBio.util import get_entity_from_entry
from luna.MyBio.PDB.PDBParser import PDBParser
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
import numpy as np
import yaml
from pathlib import Path

class LUNAWrapper:
    def __init__(self, config_path=None, ifp_type=None):
        self.config = self._parse_config(config_path)
        self.params = self._init_params()
        self.ifp_type = self._determine_ifp_type(ifp_type)
        
    def _parse_config(self, config_path):
        if config_path is None:
            config_path = "data/defaults/LUNA_default.cfg"
        
        config_dict = {}
        config_parser = configparser.ConfigParser()
        config_parser.read(config_path)
        
        for section in config_parser.sections():
            config_dict.update(config_parser.items(section))
        
        config_dict['feat_cfg'] = "data/defaults/atom_prop_default.fdef"
        config_dict['inter_cfg'] = "data/defaults/inter_default.cfg"
        config_dict['filter_cfg'] = "data/defaults/filter_default.cfg"
        config_dict['bind_cfg'] = "data/defaults/bind_default.cfg"
        
        return config_dict
    
    def _init_params(self):
        return ProjectParams(self.config, fill_defaults=True)
    
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
            return fp_config.get('default_type', 'LUNA')
        
        return 'LUNA'  # default fallback
    
    def set_ifp_type(self, ifp_type):
        """Set the interaction fingerprint type"""
        valid_types = ['PLIP', 'PLIF', 'LUNA']
        if ifp_type not in valid_types:
            raise ValueError(f"Invalid IFP type: {ifp_type}. Must be one of: {valid_types}")
        self.ifp_type = ifp_type
    
    def calculate_interactions(self, protein_path, ligand_mol, ligand_name="ligand"):
        pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True)
        structure = pdb_parser.get_structure("protein", protein_path)
        
        # Get available chain IDs from the protein structure
        available_chains = [chain.id for chain in structure[0]]
        print(f"Available chains in protein: {available_chains}")
        
        # Create entry with proper chain handling
        entry = MolFileEntry.from_mol_obj("protein", ligand_name, ligand_mol)
        entry.pdb_file = protein_path
        
        # Set the chain ID to the first available chain if default 'z' doesn't exist
        if hasattr(entry, 'chain_id') and entry.chain_id == 'z' and 'z' not in available_chains:
            if available_chains:
                print(f"Changing chain ID from 'z' to '{available_chains[0]}'")
                entry.chain_id = available_chains[0]
        
        try:
            structure = entry.get_biopython_structure(structure, pdb_parser)
        except KeyError as e:
            if "'z'" in str(e):
                # Try with the first available chain
                if available_chains:
                    print(f"Chain 'z' not found, trying with chain '{available_chains[0]}'")
                    entry.chain_id = available_chains[0]
                    structure = entry.get_biopython_structure(structure, pdb_parser)
                else:
                    raise ValueError(f"No chains found in protein structure at {protein_path}")
            else:
                raise
        ligand = get_entity_from_entry(structure, entry)
        ligand.set_as_target(is_target=True)
        
        feature_factory = ChemicalFeatures.BuildFeatureFactory(self.params['atom_prop_file'])
        feature_extractor = FeatureExtractor(feature_factory)
        perceiver = AtomGroupPerceiver(feature_extractor, add_h=True, ph=7.4, amend_mol=True, tmp_path='/tmp')
        
        radius = self.params['inter_calc'].inter_config.get("bsite_cutoff", 6.2)
        nb_pairs = get_contacts_with(structure[0], ligand, level='R', radius=radius)
        nb_compounds = set([x[0] for x in nb_pairs])
        
        mol_objs_dict = {entry.get_biopython_key(): entry.mol_obj}
        atm_grps_mngr = perceiver.perceive_atom_groups(nb_compounds, mol_objs_dict=mol_objs_dict)
        
        calc_func = self.params['inter_calc'].calc_interactions
        interactions_mngr = calc_func(atm_grps_mngr.atm_grps)
        interactions_mngr.entry = entry
        
        atm_grps_mngr.merge_hydrophobic_atoms(interactions_mngr)
        
        bm_filter_func = interactions_mngr.filter_out_by_binding_mode
        bm_filter_func(self.params["binding_mode_filter"])
        
        ifp = self._create_ifp(atm_grps_mngr)
        
        return ifp, interactions_mngr
    
    def _create_ifp(self, atm_grps_mngr):
        ifp_params = self.params
        
        # Map fingerprint types to LUNA IFPType
        ifp_type_mapping = {
            'PLIP': 'PLIP',
            'PLIF': 'PLIF', 
            'LUNA': ifp_params["ifp_type"]  # Use config default for LUNA
        }
        
        luna_ifp_type = ifp_type_mapping.get(self.ifp_type, ifp_params["ifp_type"])
        
        sg = ShellGenerator(
            ifp_params["ifp_num_levels"], 
            ifp_params["ifp_radius_step"],
            diff_comp_classes=ifp_params["ifp_diff_comp_classes"],
            ifp_type=IFPType[luna_ifp_type]
        )
        sm = sg.create_shells(atm_grps_mngr)
        
        unique_shells = not ifp_params["ifp_count"]
        return sm.to_fingerprint(
            fold_to_length=ifp_params["ifp_length"], 
            unique_shells=unique_shells, 
            count_fp=ifp_params["ifp_count"]
        )