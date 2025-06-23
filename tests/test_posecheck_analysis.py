"""
Test suite for core/pose_analysis/posecheck.py functions.
Focuses on pose quality analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem

from core.pose_analysis.posecheck import (
    analyze_single_molecule_pose,
    compute_pose_quality_batch,
    analyze_single_molecule_pose_simple,
    compute_pose_quality_simple,
    POSECHECK_AVAILABLE
)


class TestPosecheckAnalysis:
    """Test suite for pose quality analysis functions"""
    
    def create_test_molecule_block(self):
        """Create a simple SDF block for testing"""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        if mol:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            return Chem.MolToMolBlock(mol)
        return None
    
    def create_test_protein_content(self):
        """Create simple PDB content for testing"""
        return """HEADER    TEST PROTEIN
ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N
ATOM      2  CA  ALA A   1      18.699  16.967  15.691  1.00 11.99           C
ATOM      3  C   ALA A   1      18.087  16.967  17.074  1.00 11.99           C
ATOM      4  O   ALA A   1      17.135  16.286  17.315  1.00 11.99           O
END"""
    
    def create_test_dataframe(self):
        """Create test DataFrame with molecules"""
        mol_block = self.create_test_molecule_block()
        return pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'mol_block': [mol_block, mol_block],
            'clashes': [0, 0],
            'strain_energy': [0.0, 0.0]
        })
    
    def test_analyze_single_molecule_pose_no_posecheck(self):
        """Test single molecule analysis when PoseCheck is not available"""
        mol_block = self.create_test_molecule_block()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        with patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', False):
            result = analyze_single_molecule_pose((1, mol_block, protein_content, config))
        
        assert result['id'] == 1
        assert result['error'] == "PoseCheck not available"
        assert result['clashes'] == 0
        assert result['strain_energy'] == 0.0

    @patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', True)
    @patch('core.pose_analysis.posecheck.PoseCheck')
    def test_analyze_single_molecule_pose_with_posecheck(self, mock_posecheck):
        """Test single molecule analysis with PoseCheck available"""
        mol_block = self.create_test_molecule_block()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Mock PoseCheck instance
        mock_pc = Mock()
        mock_pc.calculate_clashes.return_value = [5]  # 5 clashes
        mock_pc.calculate_strain_energy.return_value = [12.5]  # 12.5 strain energy
        mock_posecheck.return_value = mock_pc
        
        result = analyze_single_molecule_pose((1, mol_block, protein_content, config))
        
        assert result['id'] == 1
        assert result['error'] is None
        assert result['clashes'] == 5
        assert result['strain_energy'] == 12.5

    @patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', True)
    @patch('core.pose_analysis.posecheck.PoseCheck')
    def test_analyze_single_molecule_pose_dict_results(self, mock_posecheck):
        """Test single molecule analysis with dictionary results from PoseCheck"""
        mol_block = self.create_test_molecule_block()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Mock PoseCheck instance with dict results
        mock_pc = Mock()
        mock_pc.calculate_clashes.return_value = [{'clashes': 3}]
        mock_pc.calculate_strain_energy.return_value = [{'strain_energy': 8.2}]
        mock_posecheck.return_value = mock_pc
        
        result = analyze_single_molecule_pose((1, mol_block, protein_content, config))
        
        assert result['id'] == 1
        assert result['clashes'] == 3
        assert result['strain_energy'] == 8.2

    @patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', True)
    @patch('core.pose_analysis.posecheck.PoseCheck')
    def test_analyze_single_molecule_pose_calculation_error(self, mock_posecheck):
        """Test handling of calculation errors in PoseCheck"""
        mol_block = self.create_test_molecule_block()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Mock PoseCheck instance with errors
        mock_pc = Mock()
        mock_pc.calculate_clashes.side_effect = Exception("Clash calculation failed")
        mock_pc.calculate_strain_energy.return_value = [2.5]
        mock_posecheck.return_value = mock_pc
        
        result = analyze_single_molecule_pose((1, mol_block, protein_content, config))
        
        assert result['id'] == 1
        assert result['clashes'] == 0  # Should default to 0 on error
        assert result['strain_energy'] == 2.5  # Should still work

    def test_analyze_single_molecule_pose_simple_valid_molecule(self):
        """Test simple pose analysis with valid molecule"""
        mol_block = self.create_test_molecule_block()
        protein_content = self.create_test_protein_content()
        
        result = analyze_single_molecule_pose_simple(1, mol_block, protein_content)
        
        assert result['id'] == 1
        assert result['error'] is None
        assert isinstance(result['clashes'], int)
        assert isinstance(result['strain_energy'], float)
        assert result['clashes'] >= 0
        assert result['strain_energy'] >= 0.0

    def test_analyze_single_molecule_pose_simple_invalid_molecule(self):
        """Test simple pose analysis with invalid molecule"""
        invalid_mol_block = "INVALID MOL BLOCK"
        protein_content = self.create_test_protein_content()
        
        result = analyze_single_molecule_pose_simple(1, invalid_mol_block, protein_content)
        
        assert result['id'] == 1
        assert result['error'] == "Invalid molecule"
        assert result['clashes'] == 0
        assert result['strain_energy'] == 0.0

    def test_analyze_single_molecule_pose_simple_no_conformer(self):
        """Test simple pose analysis with molecule that has no conformer"""
        # Create molecule without conformer
        mol = Chem.MolFromSmiles("CCO")
        mol_block = Chem.MolToMolBlock(mol)  # No 3D coordinates
        protein_content = self.create_test_protein_content()
        
        result = analyze_single_molecule_pose_simple(1, mol_block, protein_content)
        
        assert result['id'] == 1
        assert result['error'] == "No conformer available"

    def test_analyze_single_molecule_pose_simple_large_molecule(self):
        """Test simple pose analysis with large molecule"""
        # Create a larger molecule (should trigger clash heuristic)
        large_smiles = "C" * 60  # Long chain to create large molecule
        mol = Chem.MolFromSmiles(large_smiles)
        if mol:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            mol_block = Chem.MolToMolBlock(mol)
            
            result = analyze_single_molecule_pose_simple(1, mol_block, "")
            
            assert result['id'] == 1
            assert result['clashes'] > 0  # Should detect clashes based on size

    def test_compute_pose_quality_batch_empty_dataframe(self):
        """Test batch processing with empty DataFrame"""
        df = pd.DataFrame(columns=['id', 'mol_block', 'clashes', 'strain_energy'])
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True}
        
        result = compute_pose_quality_batch(df, protein_content, config)
        
        assert len(result) == 0
        assert list(result.columns) == list(df.columns)

    def test_compute_pose_quality_batch_no_protein(self):
        """Test batch processing with no protein content"""
        df = self.create_test_dataframe()
        config = {'calculate_clashes': True}
        
        result = compute_pose_quality_batch(df, "", config)
        
        assert len(result) == len(df)
        # Should return unchanged DataFrame

    @patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', False)
    def test_compute_pose_quality_batch_no_posecheck(self):
        """Test batch processing when PoseCheck is not available"""
        df = self.create_test_dataframe()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True}
        
        with patch('core.pose_analysis.posecheck.compute_pose_quality_simple') as mock_simple:
            mock_simple.return_value = df
            
            result = compute_pose_quality_batch(df, protein_content, config)
            
            mock_simple.assert_called_once()

    @patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', True)
    @patch('core.pose_analysis.posecheck.analyze_single_molecule_pose')
    def test_compute_pose_quality_batch_sequential_fallback(self, mock_analyze):
        """Test batch processing with sequential fallback"""
        df = self.create_test_dataframe()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True}
        
        # Mock successful analysis
        mock_analyze.side_effect = [
            {'id': 1, 'clashes': 2, 'strain_energy': 5.0, 'error': None},
            {'id': 2, 'clashes': 1, 'strain_energy': 3.0, 'error': None}
        ]
        
        with patch('core.pose_analysis.posecheck.ProcessPoolExecutor') as mock_executor:
            # Mock ProcessPoolExecutor to raise exception
            mock_executor.side_effect = Exception("Parallel processing failed")
            
            result = compute_pose_quality_batch(df, protein_content, config, n_workers=2)
            
            assert len(result) == 2
            assert result.loc[result['id'] == 1, 'clashes'].iloc[0] == 2
            assert result.loc[result['id'] == 2, 'clashes'].iloc[0] == 1

    def test_compute_pose_quality_simple_empty_dataframe(self):
        """Test simple pose quality computation with empty DataFrame"""
        df = pd.DataFrame(columns=['id', 'mol_block', 'clashes', 'strain_energy'])
        protein_content = self.create_test_protein_content()
        
        result = compute_pose_quality_simple(df, protein_content)
        
        assert len(result) == 0

    def test_compute_pose_quality_simple_with_data(self):
        """Test simple pose quality computation with data"""
        df = self.create_test_dataframe()
        protein_content = self.create_test_protein_content()
        
        result = compute_pose_quality_simple(df, protein_content)
        
        assert len(result) == 2
        assert all(pd.notna(result['clashes']))
        assert all(pd.notna(result['strain_energy']))
        assert all(result['clashes'] >= 0)
        assert all(result['strain_energy'] >= 0.0)

    def test_compute_pose_quality_simple_invalid_mol_blocks(self):
        """Test simple pose quality computation with invalid molecule blocks"""
        df = pd.DataFrame({
            'id': [1, 2],
            'mol_block': ["INVALID", None],
            'clashes': [0, 0],
            'strain_energy': [0.0, 0.0]
        })
        protein_content = self.create_test_protein_content()
        
        result = compute_pose_quality_simple(df, protein_content)
        
        assert len(result) == 2
        # Should handle invalid blocks gracefully

    @patch('core.pose_analysis.posecheck.POSECHECK_AVAILABLE', True)
    @patch('core.pose_analysis.posecheck.PoseCheck')
    def test_analyze_single_molecule_pose_empty_results(self, mock_posecheck):
        """Test handling of empty results from PoseCheck"""
        mol_block = self.create_test_molecule_block()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Mock PoseCheck instance with empty results
        mock_pc = Mock()
        mock_pc.calculate_clashes.return_value = []  # Empty results
        mock_pc.calculate_strain_energy.return_value = []
        mock_posecheck.return_value = mock_pc
        
        result = analyze_single_molecule_pose((1, mol_block, protein_content, config))
        
        assert result['id'] == 1
        assert result['clashes'] == 0  # Should default to 0
        assert result['strain_energy'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__])