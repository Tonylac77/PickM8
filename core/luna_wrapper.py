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
        # Use the exact pattern from luna_utils.py
        pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True, 
                            FIX_EMPTY_CHAINS=True,
                            FIX_ATOM_NAME_CONFLICT=True, 
                            FIX_OBABEL_FLAGS=False)
        
        structure = pdb_parser.get_structure("protein", protein_path)
        
        # Create entry following autoparty pattern
        entry = MolFileEntry.from_mol_obj("protein", ligand_name, ligand_mol)
        entry.pdb_file = protein_path
        
        # Get structure without modifying entry properties
        structure = entry.get_biopython_structure(structure, pdb_parser)
        
        # Determine if we need to add hydrogens (from autoparty)
        add_hydrogen = self._decide_hydrogen_addition(True, pdb_parser.get_header(), entry)
        
        ligand = get_entity_from_entry(structure, entry)
        ligand.set_as_target(is_target=True)
        
        # Use autoparty's perceive_chemical_groups function pattern
        atm_grps_mngr = self._perceive_chemical_groups(entry, structure[0], ligand, add_hydrogen)
        atm_grps_mngr.entry = entry
        
        calc_func = self.params['inter_calc'].calc_interactions
        interactions_mngr = calc_func(atm_grps_mngr.atm_grps)
        interactions_mngr.entry = entry
        
        atm_grps_mngr.merge_hydrophobic_atoms(interactions_mngr)
        
        # Apply binding mode filter
        bm_filter_func = interactions_mngr.filter_out_by_binding_mode
        bm_filter_func(self.params["binding_mode_filter"])
        
        ifp = self._create_ifp(atm_grps_mngr)
        
        return ifp, interactions_mngr

    def _decide_hydrogen_addition(self, try_h_addition, pdb_header, entry):
        # From autoparty luna_utils.py
        if try_h_addition:
            if "structure_method" in pdb_header:
                method = pdb_header["structure_method"]
                # If the method is not a NMR type does not add hydrogen as it usually already has hydrogens.
                if method.upper() in ['NMR']:  # NMR_METHODS from luna.util.default_values
                    return False
            return True
        return False

    def _perceive_chemical_groups(self, entry, entity, ligand, add_h=False):
        # From autoparty pattern
        perceiver = self._get_perceiver(add_h=add_h)
        
        radius = self.params['inter_calc'].inter_config.get("bsite_cutoff", 6.2)
        nb_pairs = get_contacts_with(entity, ligand, level='R', radius=radius)
        nb_compounds = set([x[0] for x in nb_pairs])
        
        mol_objs_dict = {}
        if isinstance(entry, MolFileEntry):
            mol_objs_dict[entry.get_biopython_key()] = entry.mol_obj
        
        atm_grps_mngr = perceiver.perceive_atom_groups(nb_compounds, mol_objs_dict=mol_objs_dict)
        
        return atm_grps_mngr

    def _get_perceiver(self, add_h=False, cache=None):
        # From autoparty pattern
        feature_factory = ChemicalFeatures.BuildFeatureFactory(self.params['atom_prop_file'])
        feature_extractor = FeatureExtractor(feature_factory)
        
        perceiver = AtomGroupPerceiver(feature_extractor, 
            add_h=add_h, ph=self.params.get('ph', 7.4), 
            amend_mol=self.params.get('amend_mol', True), 
            cache=cache, 
            tmp_path='/tmp')
        return perceiver
    
    def _create_ifp(self, atm_grps_mngr):
        ifp_params = self.params
        
        # Get the IFP type from config - it should already be an IFPType enum
        config_ifp_type = ifp_params["ifp_type"]
        
        # Use the config value directly since LUNA loads it as an enum
        luna_ifp_type = config_ifp_type
        
        # Debug: print the type to understand what we're working with
        print(f"Debug: ifp_type from config: {config_ifp_type} (type: {type(config_ifp_type)})")
        print(f"Debug: self.ifp_type: {self.ifp_type}")
        
        sg = ShellGenerator(
            ifp_params["ifp_num_levels"], 
            ifp_params["ifp_radius_step"],
            diff_comp_classes=ifp_params["ifp_diff_comp_classes"],
            ifp_type=luna_ifp_type
        )
        sm = sg.create_shells(atm_grps_mngr)
        
        unique_shells = not ifp_params["ifp_count"]
        return sm.to_fingerprint(
            fold_to_length=ifp_params["ifp_length"], 
            unique_shells=unique_shells, 
            count_fp=ifp_params["ifp_count"]
        )