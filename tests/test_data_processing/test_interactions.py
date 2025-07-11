"""
Tests for protein-ligand interaction analysis.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data_io.molecules import *
from features.interactions import *


class TestInteractions:
    """Test protein-ligand interaction computation."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent.parent.parent / "test_data"
    
    @pytest.fixture
    def protein_file(self, test_data_dir):
        """Path to test protein PDB file."""
        return test_data_dir / "1fvv_p.pdb"
    
    @pytest.fixture
    def ligand_file(self, test_data_dir):
        """Path to test ligand SDF file."""
        return test_data_dir / "1fvv_l.sdf"
    
    @pytest.fixture
    def protein_content(self, protein_file):
        """Load protein content from PDB file."""
        if not protein_file.exists():
            pytest.skip("Test protein file not available")
        
        with open(protein_file, 'r') as f:
            return f.read()

    def test_interactions_real_data(self, ligand_file, protein_content, protein_file):
        """Test protein-ligand interaction computation with real data."""
        if not ligand_file.exists() or not protein_file.exists():
            pytest.skip("Test data files not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        config = {'interaction_type': 'plip', 'ligand_name': 'LIG'}
        
        try:
            ifp_json, interactions_json, num_interactions = compute_interaction_fingerprint(
                mol, protein_content, config
            )
            
            # Should return something (even if None when tools unavailable)
            assert isinstance(num_interactions, int)
            assert num_interactions >= 0
            
            if ifp_json is not None:
                assert isinstance(ifp_json, str)
                assert isinstance(interactions_json, str)
                
        except ImportError:
            # Acceptable if interaction tools not available
            pytest.skip("Interaction analysis tools not available")

    def test_interactions_missing_tools(self):
        """Test interaction computation when tools unavailable."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        protein_content = "ATOM      1  N   ALA A   1      0.0  0.0  0.0  1.00  0.00           N"
        config = {'interaction_type': 'plip'}
        
        # Should handle gracefully when tools unavailable
        result = compute_interaction_fingerprint(mol, protein_content, config)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[2], int)  # num_interactions should be int


if __name__ == '__main__':
    pytest.main([__file__])