"""
Test suite for utils/processing.py functions.
Focuses on molecular fingerprint computation and batch processing functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem

from utils.processing import (
    process_molecule_fingerprints,
    compute_fingerprints_batch,
    get_fingerprint_statistics
)


class TestProcessing:
    """Test suite for processing functions"""
    
    def create_test_molecule(self):
        """Create a simple test molecule"""
        return Chem.MolFromSmiles("CCO")  # Ethanol
    
    def create_test_dataframe(self):
        """Create test DataFrame with molecules"""
        mol1 = self.create_test_molecule()
        mol2 = self.create_test_molecule()
        
        return pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'mol': [mol1, mol2],
            'morgan_fp': [None, None],
            'rdkit_fp': [None, None],
            'mapchiral_fp': [None, None],
            'interaction_fp': [None, None],
            'interactions': [None, None],
            'num_interactions': [0, 0]
        })

    @patch('utils.processing.compute_morgan_fingerprint')
    @patch('utils.processing.compute_rdkit_fingerprint')
    @patch('utils.processing.compute_mapchiral_fingerprint')
    @patch('utils.processing.compute_interaction_fingerprint')
    def test_process_molecule_fingerprints_success(self, mock_ifp, mock_mapchiral, mock_rdkit, mock_morgan):
        """Test successful fingerprint processing for single molecule"""
        mol = self.create_test_molecule()
        args = (1, mol, "PDB_CONTENT", 
                {'compute_morgan': True, 'compute_rdkit': True, 'compute_mapchiral': True, 'compute_interactions': True},
                {'interaction_type': 'PLIP'})
        
        # Mock return values
        mock_morgan.return_value = [1, 0, 1, 0]
        mock_rdkit.return_value = [0, 1, 0, 1]
        mock_mapchiral.return_value = [1, 1, 0, 0]
        mock_ifp.return_value = ("[1,0,1]", "interaction_data", 3)
        
        result = process_molecule_fingerprints(args)
        
        assert result['id'] == 1
        assert result['morgan_fp'] == [1, 0, 1, 0]
        assert result['rdkit_fp'] == [0, 1, 0, 1]
        assert result['mapchiral_fp'] == [1, 1, 0, 0]
        assert result['interaction_fp'] == "[1,0,1]"
        assert result['interactions'] == "interaction_data"
        assert result['num_interactions'] == 3

    def test_process_molecule_fingerprints_no_protein(self):
        """Test fingerprint processing without protein"""
        mol = self.create_test_molecule()
        args = (1, mol, None, 
                {'compute_morgan': True, 'compute_rdkit': True, 'compute_mapchiral': True, 'compute_interactions': True},
                {'interaction_type': 'PLIP'})
        
        with patch('utils.processing.compute_morgan_fingerprint') as mock_morgan, \
             patch('utils.processing.compute_rdkit_fingerprint') as mock_rdkit, \
             patch('utils.processing.compute_mapchiral_fingerprint') as mock_mapchiral:
            
            mock_morgan.return_value = [1, 0, 1, 0]
            mock_rdkit.return_value = [0, 1, 0, 1]
            mock_mapchiral.return_value = [1, 1, 0, 0]
            
            result = process_molecule_fingerprints(args)
            
            assert result['id'] == 1
            assert result['morgan_fp'] == [1, 0, 1, 0]
            assert result['rdkit_fp'] == [0, 1, 0, 1]
            assert result['mapchiral_fp'] == [1, 1, 0, 0]
            assert result['interaction_fp'] is None
            assert result['interactions'] is None
            assert result['num_interactions'] == 0

    @patch('utils.processing.compute_morgan_fingerprint')
    def test_process_molecule_fingerprints_error_handling(self, mock_morgan):
        """Test error handling in fingerprint processing"""
        mol = self.create_test_molecule()
        args = (1, mol, None, 
                {'compute_morgan': True, 'compute_rdkit': False, 'compute_interactions': False},
                {})
        
        # Mock an exception
        mock_morgan.side_effect = Exception("Fingerprint computation error")
        
        result = process_molecule_fingerprints(args)
        
        assert result['id'] == 1
        assert result['morgan_fp'] is None
        assert result['rdkit_fp'] is None
        assert result['interaction_fp'] is None

    @patch('utils.processing.process_molecule_fingerprints')
    def test_compute_fingerprints_batch_sequential(self, mock_process):
        """Test batch fingerprint computation with sequential fallback"""
        df = self.create_test_dataframe()
        
        # Mock successful processing
        mock_process.side_effect = [
            {'id': 1, 'morgan_fp': [1, 0], 'rdkit_fp': [0, 1], 'mapchiral_fp': [1, 1], 
             'interaction_fp': None, 'interactions': None, 'num_interactions': 0},
            {'id': 2, 'morgan_fp': [0, 1], 'rdkit_fp': [1, 0], 'mapchiral_fp': [0, 0], 
             'interaction_fp': None, 'interactions': None, 'num_interactions': 0}
        ]
        
        with patch('utils.processing.ProcessPoolExecutor') as mock_executor:
            # Mock ProcessPoolExecutor to raise exception for parallel processing
            mock_executor.side_effect = Exception("Parallel processing failed")
            
            result = compute_fingerprints_batch(
                df, "PDB_CONTENT", 
                {'compute_morgan': True}, 
                {'interaction_type': 'PLIP'},
                n_workers=2
            )
            
            assert len(result) == 2
            assert result.loc[result['id'] == 1, 'morgan_fp'].iloc[0] == [1, 0]
            assert result.loc[result['id'] == 2, 'morgan_fp'].iloc[0] == [0, 1]

    def test_compute_fingerprints_batch_empty_dataframe(self):
        """Test batch processing with empty DataFrame"""
        df = pd.DataFrame(columns=['id', 'mol', 'morgan_fp', 'rdkit_fp', 
                                  'interaction_fp', 'interactions', 'num_interactions'])
        
        result = compute_fingerprints_batch(
            df, "PDB_CONTENT",
            {'compute_morgan': True},
            {'interaction_type': 'PLIP'}
        )
        
        assert len(result) == 0
        assert list(result.columns) == list(df.columns)

    def test_compute_fingerprints_batch_no_valid_molecules(self):
        """Test batch processing with no valid molecules"""
        df = pd.DataFrame({
            'id': [1, 2],
            'mol': [None, None],  # No valid molecules
            'morgan_fp': [None, None],
            'rdkit_fp': [None, None],
            'interaction_fp': [None, None],
            'interactions': [None, None],
            'num_interactions': [0, 0]
        })
        
        result = compute_fingerprints_batch(
            df, "PDB_CONTENT",
            {'compute_morgan': True},
            {'interaction_type': 'PLIP'}
        )
        
        assert len(result) == 2
        # Should return unchanged DataFrame since no valid molecules

    def test_get_fingerprint_statistics_empty(self):
        """Test statistics calculation with empty DataFrame"""
        df = pd.DataFrame(columns=['morgan_fp', 'rdkit_fp', 'interaction_fp', 'num_interactions'])
        
        stats = get_fingerprint_statistics(df)
        
        assert stats['total_molecules'] == 0

    def test_get_fingerprint_statistics_with_data(self):
        """Test statistics calculation with data"""
        df = pd.DataFrame({
            'morgan_fp': [[1, 0], [0, 1], None],
            'rdkit_fp': [[1, 1], [0, 0], [1, 0]],
            'interaction_fp': ['[1,0]', None, '[0,1]'],
            'num_interactions': [2, 0, 3]
        })
        
        stats = get_fingerprint_statistics(df)
        
        assert stats['total_molecules'] == 3
        assert stats['morgan_fp_computed'] == 2
        assert stats['rdkit_fp_computed'] == 3
        assert stats['interaction_fp_computed'] == 2
        assert stats['molecules_with_interactions'] == 2
        assert stats['avg_interactions_per_molecule'] == pytest.approx(5/3)
        assert stats['max_interactions'] == 3
        assert stats['min_interactions'] == 0
        
        # Check percentages
        assert stats['morgan_fp_percentage'] == pytest.approx(66.67, rel=1e-2)
        assert stats['rdkit_fp_percentage'] == 100.0
        assert stats['interaction_fp_percentage'] == pytest.approx(66.67, rel=1e-2)

    def test_get_fingerprint_statistics_all_computed(self):
        """Test statistics with all fingerprints computed"""
        df = pd.DataFrame({
            'morgan_fp': [[1, 0], [0, 1]],
            'rdkit_fp': [[1, 1], [0, 0]],
            'interaction_fp': ['[1,0]', '[0,1]'],
            'num_interactions': [2, 1]
        })
        
        stats = get_fingerprint_statistics(df)
        
        assert stats['total_molecules'] == 2
        assert stats['morgan_fp_computed'] == 2
        assert stats['rdkit_fp_computed'] == 2
        assert stats['interaction_fp_computed'] == 2
        assert stats['molecules_with_interactions'] == 2
        assert stats['morgan_fp_percentage'] == 100.0
        assert stats['rdkit_fp_percentage'] == 100.0
        assert stats['interaction_fp_percentage'] == 100.0

    def test_get_fingerprint_statistics_no_interactions(self):
        """Test statistics with no interactions"""
        df = pd.DataFrame({
            'morgan_fp': [[1, 0], [0, 1]],
            'rdkit_fp': [[1, 1], [0, 0]],
            'interaction_fp': [None, None],
            'num_interactions': [0, 0]
        })
        
        stats = get_fingerprint_statistics(df)
        
        assert stats['total_molecules'] == 2
        assert stats['molecules_with_interactions'] == 0
        assert stats['avg_interactions_per_molecule'] == 0.0
        assert stats['max_interactions'] == 0
        assert stats['min_interactions'] == 0
        assert stats['interaction_fp_percentage'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__])