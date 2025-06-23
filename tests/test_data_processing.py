"""
Test suite for utils/data_processing.py functions.
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

from utils.data_processing import (
    create_empty_molecules_dataframe,
    load_sdf_file,
    load_pdb_file,
    ensure_numeric_columns,
    save_molecules_dataframe,
    load_molecules_dataframe,
    add_grade_to_molecule,
    get_graded_molecules,
    get_ungraded_molecules,
    update_pose_quality_metrics,
    update_ml_predictions,
    get_molecule_features,
    save_session_metadata,
    load_session_metadata
)


class TestDataProcessing:
    """Test suite for data processing functions"""
    
    def test_create_empty_molecules_dataframe(self):
        """Test creation of empty DataFrame with proper schema"""
        df = create_empty_molecules_dataframe()
        
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

    @patch('utils.data_processing.PandasTools.LoadSDF')
    def test_load_sdf_file_success(self, mock_load_sdf):
        """Test successful SDF file loading"""
        # Mock RDKit molecule
        mock_mol = Mock()
        mock_mol.return_value = None
        
        # Mock DataFrame from PandasTools
        mock_df = pd.DataFrame({
            'mol': [mock_mol, mock_mol],
            'ID': ['mol1', 'mol2']
        })
        mock_load_sdf.return_value = mock_df
        
        with patch('utils.data_processing.Chem.MolToSmiles') as mock_smiles, \
             patch('utils.data_processing.Chem.MolToMolBlock') as mock_molblock:
            
            mock_smiles.return_value = "CCO"
            mock_molblock.return_value = "MOL_BLOCK"
            
            result = load_sdf_file("test.sdf")
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert 'id' in result.columns
            assert 'name' in result.columns
            assert 'smiles' in result.columns
            assert all(result['score'] == 0.0)

    def test_load_pdb_file_success(self):
        """Test successful PDB file loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
            pdb_content = "ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N"
            tmp.write(pdb_content)
            tmp.flush()
            
            result = load_pdb_file(tmp.name)
            assert result == pdb_content
            
            Path(tmp.name).unlink()  # Clean up

    def test_ensure_numeric_columns(self):
        """Test numeric column type enforcement"""
        df = pd.DataFrame({
            'id': [1, 2],
            'clashes': ['2', '3'],
            'strain_energy': ['1.5', '2.0'],
            'num_interactions': ['5', '7'],
            'prediction_uncertainty': ['0.2', '0.3'],
            'prediction': ['A', 'B']
        })
        
        result = ensure_numeric_columns(df)
        
        assert result['clashes'].dtype == 'Int64'
        assert result['strain_energy'].dtype == 'float64' 
        assert result['num_interactions'].dtype == 'Int64'
        assert result['prediction_uncertainty'].dtype == 'float64'
        assert result['prediction'].dtype == 'object'

    def test_save_and_load_molecules_dataframe(self):
        """Test DataFrame save and load round trip"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'score': [1.0, 2.0]
        })], ignore_index=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_molecules_dataframe(df, tmpdir)
            loaded_df = load_molecules_dataframe(tmpdir)
            
            assert loaded_df is not None
            assert len(loaded_df) == 2
            assert list(loaded_df['name']) == ['mol1', 'mol2']
            assert list(loaded_df['score']) == [1.0, 2.0]

    def test_load_molecules_dataframe_not_found(self):
        """Test loading DataFrame when file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_molecules_dataframe(tmpdir)
            assert result is None

    def test_add_grade_to_molecule(self):
        """Test adding grade to specific molecule"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'grade': [None, None]
        })], ignore_index=True)
        
        result = add_grade_to_molecule(df, 1, 'A')
        
        assert result.loc[result['id'] == 1, 'grade'].iloc[0] == 'A'
        assert pd.notna(result.loc[result['id'] == 1, 'grade_timestamp'].iloc[0])
        assert pd.isna(result.loc[result['id'] == 2, 'grade'].iloc[0])

    def test_add_grade_to_nonexistent_molecule(self):
        """Test adding grade to molecule that doesn't exist"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'grade': [None, None]
        })], ignore_index=True)
        
        result = add_grade_to_molecule(df, 999, 'A')
        
        # Should return unchanged DataFrame
        assert len(result) == 2
        assert all(pd.isna(result['grade']))

    def test_get_graded_molecules(self):
        """Test filtering for graded molecules"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['mol1', 'mol2', 'mol3'],
            'grade': ['A', None, 'B']
        })], ignore_index=True)
        
        result = get_graded_molecules(df)
        
        assert len(result) == 2
        assert set(result['id']) == {1, 3}
        assert set(result['grade']) == {'A', 'B'}

    def test_get_ungraded_molecules(self):
        """Test filtering for ungraded molecules"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['mol1', 'mol2', 'mol3'],
            'grade': ['A', None, 'B']
        })], ignore_index=True)
        
        result = get_ungraded_molecules(df)
        
        assert len(result) == 1
        assert result['id'].iloc[0] == 2
        assert pd.isna(result['grade'].iloc[0])

    def test_update_pose_quality_metrics(self):
        """Test updating pose quality metrics"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'clashes': [0, 0],
            'strain_energy': [0.0, 0.0]
        })], ignore_index=True)
        
        metrics = {
            1: {'clashes': 3, 'strain_energy': 15.5},
            2: {'clashes': 1, 'strain_energy': 8.2}
        }
        
        result = update_pose_quality_metrics(df, metrics)
        
        assert result.loc[result['id'] == 1, 'clashes'].iloc[0] == 3
        assert result.loc[result['id'] == 1, 'strain_energy'].iloc[0] == 15.5
        assert result.loc[result['id'] == 2, 'clashes'].iloc[0] == 1
        assert result.loc[result['id'] == 2, 'strain_energy'].iloc[0] == 8.2

    def test_update_ml_predictions(self):
        """Test updating ML predictions"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'prediction': [None, None],
            'prediction_uncertainty': [np.nan, np.nan]
        })], ignore_index=True)
        
        predictions = {
            1: {'prediction': 'A', 'uncertainty': 0.1},
            2: {'prediction': 'B', 'uncertainty': 0.3}
        }
        
        result = update_ml_predictions(df, predictions)
        
        assert result.loc[result['id'] == 1, 'prediction'].iloc[0] == 'A'
        assert result.loc[result['id'] == 1, 'prediction_uncertainty'].iloc[0] == 0.1
        assert result.loc[result['id'] == 2, 'prediction'].iloc[0] == 'B'
        assert result.loc[result['id'] == 2, 'prediction_uncertainty'].iloc[0] == 0.3
        assert pd.notna(result.loc[result['id'] == 1, 'prediction_timestamp'].iloc[0])

    def test_get_molecule_features_empty(self):
        """Test feature extraction with empty DataFrame"""
        df = create_empty_molecules_dataframe()
        
        features, mol_ids = get_molecule_features(df)
        
        assert len(features) == 0
        assert len(mol_ids) == 0

    def test_get_molecule_features_with_data(self):
        """Test feature extraction with fingerprint data"""
        df = create_empty_molecules_dataframe()
        df = pd.concat([df, pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'morgan_fp': [[1, 0, 1, 0], [0, 1, 0, 1]],
            'rdkit_fp': [[1, 1, 0, 0], [0, 0, 1, 1]],
            'interaction_fp': ['[1, 0, 1]', '[0, 1, 0]']
        })], ignore_index=True)
        
        features, mol_ids = get_molecule_features(df)
        
        assert len(features) == 2
        assert len(mol_ids) == 2
        assert mol_ids == [1, 2]
        assert features.shape[1] > 0  # Should have concatenated features

    def test_save_and_load_session_metadata(self):
        """Test session metadata save and load"""
        metadata = {
            'session_id': 'test-123',
            'protein_name': 'test_protein',
            'created_at': '2023-01-01T00:00:00',
            'molecule_count': 100
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_session_metadata(tmpdir, metadata)
            loaded_metadata = load_session_metadata(tmpdir)
            
            assert loaded_metadata is not None
            assert loaded_metadata['session_id'] == 'test-123'
            assert loaded_metadata['protein_name'] == 'test_protein'
            assert loaded_metadata['molecule_count'] == 100

    def test_load_session_metadata_not_found(self):
        """Test loading metadata when file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_session_metadata(tmpdir)
            assert result is None


if __name__ == '__main__':
    pytest.main([__file__])