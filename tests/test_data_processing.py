"""
Test suite for data processing functions.
Focuses on core data loading, saving, and DataFrame manipulation functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from rdkit import Chem

from data import molecules
from analysis import grading


class TestDataProcessing:
    """Test suite for data processing functions"""
    
    def test_create_empty_molecules_dataframe(self):
        """Test creation of empty DataFrame with proper schema"""
        df = molecules.create_empty_dataframe()
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        
        # Check required columns exist
        expected_columns = [
            'id', 'name', 'smiles', 'mol_block', 'mol', 'score',
            'morgan_fp', 'rdkit_fp', 'interaction_fp', 'interactions', 'num_interactions',
            'grade', 'grade_timestamp', 'clashes', 'strain_energy',
            'prediction', 'prediction_uncertainty', 'prediction_timestamp'
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check data types
        assert df['id'].dtype == 'int64'
        assert df['score'].dtype == 'float64'
        assert df['num_interactions'].dtype == 'int64'

    @patch('data.molecules.PandasTools.LoadSDF')
    def test_load_sdf_file_success_mocked(self, mock_load_sdf):
        """Test successful SDF file loading (unit test with mocks)"""
        # Mock RDKit molecule
        mock_mol = Mock()
        mock_mol.GetProp.return_value = "test_mol"
        
        # Mock DataFrame from PandasTools
        mock_df = pd.DataFrame({
            'mol': [mock_mol, mock_mol],
            'ID': ['mol1', 'mol2']
        })
        mock_load_sdf.return_value = mock_df
        
        # Mock molecule to SMILES and MolBlock conversion
        with patch('rdkit.Chem.MolToSmiles') as mock_smiles, \
             patch('rdkit.Chem.MolToMolBlock') as mock_molblock:
            
            mock_smiles.return_value = "CCO"
            mock_molblock.return_value = "MOCK_MOL_BLOCK"
            
            result = molecules.load_sdf("dummy_path.sdf")
            
            assert len(result) == 2
            assert 'mol' in result.columns
            assert 'smiles' in result.columns
            assert 'mol_block' in result.columns
            assert result['name'].tolist() == ['mol1', 'mol2']

    def test_load_sdf_file_real_data_single_molecule(self):
        """Test SDF loading with real molecular data (single molecule)"""
        test_file_path = "test_data/1fvv_l.sdf"
        
        # Test with real SDF file
        result = molecules.load_sdf(test_file_path)
        
        # Verify basic structure
        assert len(result) == 1
        assert 'mol' in result.columns
        assert 'smiles' in result.columns
        assert 'mol_block' in result.columns
        assert 'name' in result.columns
        
        # Verify molecule was loaded correctly
        assert result['mol'].iloc[0] is not None
        assert isinstance(result['smiles'].iloc[0], str)
        assert isinstance(result['mol_block'].iloc[0], str)
        
        # Verify SMILES is reasonable (should be a valid molecular string)
        smiles = result['smiles'].iloc[0]
        assert len(smiles) > 5  # Real molecule should have meaningful SMILES
        assert 'C' in smiles or 'N' in smiles or 'O' in smiles  # Should contain atoms
        
        # Verify all expected columns are present with proper defaults
        expected_columns = [
            'id', 'name', 'smiles', 'mol_block', 'mol', 'score',
            'morgan_fp', 'rdkit_fp', 'interaction_fp', 'interactions', 'num_interactions',
            'grade', 'grade_timestamp', 'clashes', 'strain_energy',
            'prediction', 'prediction_uncertainty', 'prediction_timestamp'
        ]
        for col in expected_columns:
            assert col in result.columns

    def test_load_sdf_file_real_data_multiple_molecules(self):
        """Test SDF loading with real molecular data (multiple molecules)"""
        test_file_path = "test_data/example_poses_1fvv.sdf"
        
        # Test with real SDF file containing multiple poses
        result = molecules.load_sdf(test_file_path)
        
        # Verify we loaded multiple molecules
        assert len(result) > 1
        
        # Verify all molecules were processed correctly
        for idx, row in result.iterrows():
            assert row['mol'] is not None
            assert isinstance(row['smiles'], str) 
            assert isinstance(row['mol_block'], str)
            assert len(row['smiles']) > 5  # Real molecules should have meaningful SMILES
            
        # Verify molecule names are assigned correctly
        assert all(result['name'].notna())
        
        # For multiple poses of same ligand, SMILES might be similar but mol_blocks different
        smiles_list = result['smiles'].tolist()
        mol_block_list = result['mol_block'].tolist()
        
        # All should be valid molecular representations
        assert all(len(s) > 5 for s in smiles_list)
        assert all(len(mb) > 100 for mb in mol_block_list)  # Mol blocks should be substantial

    def test_detect_sdf_properties_real_data(self):
        """Test SDF property detection with real molecular data"""
        test_file_path = "test_data/example_poses_1fvv.sdf"
        
        # Test property detection on real SDF file
        properties = molecules.detect_sdf_properties(test_file_path)
        
        # Should return a list of property names
        assert isinstance(properties, list)
        
        # Real SDF files typically have some properties
        # The properties should not include internal RDKit columns
        for prop in properties:
            assert prop not in ['mol', 'ID']
            assert not prop.startswith('_')
            assert isinstance(prop, str)

    def test_process_score_column_real_data(self):
        """Test score column processing with real data"""
        test_file_path = "test_data/example_poses_1fvv.sdf"
        
        # First load the real data
        df = molecules.load_sdf(test_file_path)
        
        # Add a test score column with numeric data
        df['test_score'] = [i * 0.1 for i in range(len(df))]
        
        # Test score processing
        result = molecules.process_score_column(df, 'test_score', 'Lower is better')
        
        # Verify score column was processed correctly
        assert 'score' in result.columns
        assert result['score'].dtype == 'float64'
        assert all(result['score'].notna())
        assert len(result) == len(df)
        
        # Verify original data is preserved
        assert 'test_score' in result.columns

    def test_save_and_load_molecules_dataframe(self):
        """Test saving and loading DataFrame to/from directory"""
        # Create test DataFrame
        df = molecules.create_empty_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'smiles': ['CCO', 'CC'],
            'score': [0.8, 0.6]
        })], ignore_index=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test saving
            success = molecules.save_molecules_dataframe(df, tmpdir)
            assert success is True
            
            # Test loading
            loaded_df = molecules.load_molecules_dataframe(tmpdir)
            assert loaded_df is not None
            assert len(loaded_df) == 2
            assert loaded_df['name'].tolist() == ['mol1', 'mol2']

    def test_load_molecules_dataframe_not_found(self):
        """Test loading from non-existent directory"""
        result = molecules.load_molecules_dataframe("/non/existent/path")
        assert result is None

    def test_add_grade_to_molecule(self):
        """Test adding grade to molecule"""
        df = molecules.create_empty_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'grade': [None, None]
        })], ignore_index=True)
        
        result = molecules.add_grade_to_molecule(df, 1, 'Good')
        
        assert result is not None
        assert len(result) == 2
        # The add_grade_to_molecule function redirects to grading module,
        # so we just verify it doesn't crash

    def test_add_grade_to_nonexistent_molecule(self):
        """Test adding grade to non-existent molecule"""
        df = molecules.create_empty_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'grade': [None, None]
        })], ignore_index=True)
        
        result = molecules.add_grade_to_molecule(df, 999, 'Good')
        
        # Should handle gracefully (exact behavior depends on grading module implementation)
        assert result is not None

    def test_get_graded_molecules(self):
        """Test filtering graded molecules"""
        df = molecules.create_empty_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['mol1', 'mol2', 'mol3'],
            'grade': ['Good', None, 'Bad']
        })], ignore_index=True)
        
        result = grading.get_graded_molecules(df)
        
        assert len(result) == 2  # Only molecules with grades
        assert set(result['grade'].tolist()) == {'Good', 'Bad'}

    def test_get_ungraded_molecules(self):
        """Test filtering ungraded molecules"""
        df = molecules.create_empty_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['mol1', 'mol2', 'mol3'],
            'grade': ['Good', None, 'Bad']
        })], ignore_index=True)
        
        result = grading.get_ungraded_molecules(df)
        
        assert len(result) == 1  # Only molecule without grade
        assert result.iloc[0]['name'] == 'mol2'


if __name__ == '__main__':
    pytest.main([__file__])