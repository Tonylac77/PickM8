"""
Tests for molecule loading and SDF operations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data_io.molecules import *


class TestMoleculeLoading:
    """Test SDF loading and molecule operations."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent.parent.parent / "test_data"
    
    @pytest.fixture
    def ligand_file(self, test_data_dir):
        """Path to test ligand SDF file."""
        return test_data_dir / "1fvv_l.sdf"
    
    @pytest.fixture
    def poses_file(self, test_data_dir):
        """Path to test poses SDF file."""
        return test_data_dir / "example_poses_1fvv.sdf"

    def test_sdf_loading_single_molecule(self, ligand_file):
        """Test loading single molecule from SDF file."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        
        # Essential checks
        assert len(df) == 1
        assert 'mol' in df.columns
        assert 'smiles' in df.columns
        assert 'mol_block' in df.columns
        
        # Verify molecule was loaded correctly
        mol = df['mol'].iloc[0]
        assert mol is not None
        assert mol.GetNumAtoms() > 0
        
        # Verify SMILES is reasonable
        smiles = df['smiles'].iloc[0]
        assert isinstance(smiles, str)
        assert len(smiles) > 5

    def test_sdf_loading_multiple_molecules(self, poses_file):
        """Test loading multiple molecules from SDF file."""
        if not poses_file.exists():
            pytest.skip("Test poses file not available")
        
        df = load_sdf(str(poses_file))
        
        # Essential checks
        assert len(df) > 1
        assert all(df['mol'].notna())
        assert all(df['smiles'].str.len() > 5)
        assert all(df['mol_block'].str.len() > 100)

    def test_sdf_loading_file_not_found(self):
        """Test SDF loading with non-existent file."""
        with pytest.raises(Exception):
            load_sdf("/non/existent/file.sdf")

    def test_molecules_dataframe_save_load(self):
        """Test molecules DataFrame save/load functionality."""
        import tempfile
        import json
        
        # Create sample data
        np.random.seed(42)
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'smiles': ['CCO', 'CC', 'CCC', 'CCCC', 'CCCCC'],
            'score': [-2.5, -3.1, -1.8, -4.2, -2.9],
            'grade': [None, 'A', None, 'B', None],
        }
        df = pd.DataFrame(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test saving
            success = save_molecules_dataframe(df, tmpdir)
            assert success is True
            
            # Test loading
            loaded_df = load_molecules_dataframe(tmpdir)
            
            assert loaded_df is not None
            assert len(loaded_df) == len(df)
            
            # Verify key columns preserved
            for col in ['id', 'name', 'smiles', 'score', 'grade']:
                assert col in loaded_df.columns
                pd.testing.assert_series_equal(loaded_df[col], df[col])


if __name__ == '__main__':
    pytest.main([__file__])