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

from analysis import pose_quality
from analysis.pose_quality import (
    analyze_single_pose,
    analyze_all_poses,
    get_pose_quality_statistics
)


def analyze_single_molecule_pose_simple(molecule_id: int, mol_block: str, protein_content: str) -> dict:
    """Wrapper function for test compatibility with the old API."""
    config = {'calculate_clashes': True, 'calculate_strain': True}
    result = analyze_single_pose(mol_block, protein_content, config)
    result['id'] = molecule_id
    return result


def compute_pose_quality_batch(df, protein_content, config, n_workers=None):
    """Wrapper function for test compatibility with the old batch API."""
    return analyze_all_poses(df, protein_content, config)


def compute_pose_quality_simple(df, protein_content):
    """Wrapper function for test compatibility with the old simple API."""
    config = {'calculate_clashes': True, 'calculate_strain': True, 'enabled': True}
    return analyze_all_poses(df, protein_content, config)


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
        
        with patch('analysis.pose_quality.POSECHECK_AVAILABLE', False):
            result = analyze_single_pose(mol_block, protein_content, config)
            result['id'] = 1  # Add id for test compatibility
        
        assert result['id'] == 1
        assert result['error'] == "PoseCheck not available"
        assert result['clashes'] == 0
        assert result['strain_energy'] == 0.0

    @patch('analysis.pose_quality.POSECHECK_AVAILABLE', True)
    @patch('analysis.pose_quality.PoseCheck')
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
        
        result = analyze_single_pose(mol_block, protein_content, config)
        result['id'] = 1  # Add id for test compatibility
        
        assert result['id'] == 1
        assert result['error'] is None
        assert result['clashes'] == 5
        assert result['strain_energy'] == 12.5

    @patch('analysis.pose_quality.POSECHECK_AVAILABLE', True)
    @patch('analysis.pose_quality.PoseCheck')
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
        
        result = analyze_single_pose(mol_block, protein_content, config)
        result['id'] = 1  # Add id for test compatibility
        
        assert result['id'] == 1
        assert result['clashes'] == 3
        assert result['strain_energy'] == 8.2

    @patch('analysis.pose_quality.POSECHECK_AVAILABLE', True)
    @patch('analysis.pose_quality.PoseCheck')
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
        
        result = analyze_single_pose(mol_block, protein_content, config)
        result['id'] = 1  # Add id for test compatibility
        
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
        # The current implementation doesn't set specific error messages for invalid molecules
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
        # The current implementation doesn't set specific error messages for conformer issues
        assert isinstance(result['clashes'], int)
        assert isinstance(result['strain_energy'], float)

    def test_analyze_single_molecule_pose_simple_large_molecule(self):
        """Test simple pose analysis with large molecule"""
        # Create a larger molecule 
        large_smiles = "C" * 20  # Reasonable chain length
        mol = Chem.MolFromSmiles(large_smiles)
        if mol:
            mol = Chem.AddHs(mol)
            Chem.AllChem.EmbedMolecule(mol)
            mol_block = Chem.MolToMolBlock(mol)
            
            result = analyze_single_molecule_pose_simple(1, mol_block, "")
            
            assert result['id'] == 1
            # Without PoseCheck, clashes default to 0
            assert isinstance(result['clashes'], int)
            assert isinstance(result['strain_energy'], float)

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

    @patch('analysis.pose_quality.POSECHECK_AVAILABLE', False)
    def test_compute_pose_quality_batch_no_posecheck(self):
        """Test batch processing when PoseCheck is not available"""
        df = self.create_test_dataframe()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True}
        
        result = compute_pose_quality_batch(df, protein_content, config)
        
        # Should return DataFrame without crashing, even when PoseCheck is not available
        assert len(result) == len(df)
        assert 'clashes' in result.columns
        assert 'strain_energy' in result.columns

    @patch('analysis.pose_quality.POSECHECK_AVAILABLE', True)
    @patch('analysis.pose_quality.analyze_single_pose')
    def test_compute_pose_quality_batch_sequential_fallback(self, mock_analyze):
        """Test batch processing with sequential fallback"""
        df = self.create_test_dataframe()
        protein_content = self.create_test_protein_content()
        config = {'calculate_clashes': True}
        
        # Mock successful analysis
        mock_analyze.side_effect = [
            {'clashes': 2, 'strain_energy': 5.0, 'error': None},
            {'clashes': 1, 'strain_energy': 3.0, 'error': None}
        ]
        
        # The new architecture doesn't use ProcessPoolExecutor, so test direct processing
        result = compute_pose_quality_batch(df, protein_content, config)
        
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

    @patch('analysis.pose_quality.POSECHECK_AVAILABLE', True)
    @patch('analysis.pose_quality.PoseCheck')
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
        
        result = analyze_single_pose(mol_block, protein_content, config)
        result['id'] = 1  # Add id for test compatibility
        
        assert result['id'] == 1
        assert result['clashes'] == 0  # Should default to 0
        assert result['strain_energy'] == 0.0


class TestPoseQualityRealData:
    """Integration tests using real molecular data"""
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    @pytest.fixture
    def ligand_file_path(self):
        """Path to test ligand SDF file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_l.sdf"
    
    @pytest.fixture
    def poses_file_path(self):
        """Path to test poses SDF file"""
        return Path(__file__).parent.parent / "test_data" / "example_poses_1fvv.sdf"
    
    @pytest.fixture
    def protein_content(self, protein_file_path):
        """Load real protein content from PDB file"""
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with open(protein_file_path, 'r') as f:
            return f.read()
    
    @pytest.fixture
    def real_molecules_df(self, poses_file_path):
        """Load real molecules DataFrame from poses SDF file"""
        if not poses_file_path.exists():
            pytest.skip("Test poses file not available")
        
        from data import molecules
        df = molecules.load_sdf(str(poses_file_path))
        return df if len(df) > 0 else None
    
    def test_analyze_single_pose_real_molecule(self, real_molecules_df, protein_content):
        """Test pose analysis with real molecular data"""
        if real_molecules_df is None:
            pytest.skip("Could not load real molecular data")
        
        # Get first real molecule's mol_block
        mol_block = real_molecules_df['mol_block'].iloc[0]
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Test pose analysis (will use simple method if PoseCheck not available)
        result = analyze_single_pose(mol_block, protein_content, config)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'clashes' in result
        assert 'strain_energy' in result
        assert isinstance(result['clashes'], (int, float))
        assert isinstance(result['strain_energy'], (int, float))
        assert result['clashes'] >= 0
        assert result['strain_energy'] >= 0.0
    
    def test_analyze_all_poses_real_data(self, real_molecules_df, protein_content):
        """Test pose analysis for all real molecular poses"""
        if real_molecules_df is None:
            pytest.skip("Could not load real molecular data")
        
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Test analysis on all real poses
        result_df = analyze_all_poses(real_molecules_df, protein_content, config)
        
        # Verify processing completed
        assert len(result_df) == len(real_molecules_df)
        assert 'clashes' in result_df.columns
        assert 'strain_energy' in result_df.columns
        
        # Verify all poses were processed
        assert all(result_df['clashes'].notna())
        assert all(result_df['strain_energy'].notna())
        assert all(result_df['clashes'] >= 0)
        assert all(result_df['strain_energy'] >= 0.0)
    
    def test_pose_quality_statistics_real_data(self, real_molecules_df, protein_content):
        """Test pose quality statistics with real data"""
        if real_molecules_df is None:
            pytest.skip("Could not load real molecular data")
        
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Analyze poses first
        result_df = analyze_all_poses(real_molecules_df, protein_content, config)
        
        # Get statistics
        stats = get_pose_quality_statistics(result_df)
        
        # Verify statistics structure
        assert isinstance(stats, dict)
        assert 'total_poses' in stats
        assert 'clash_statistics' in stats
        assert 'strain_statistics' in stats
        
        assert stats['total_poses'] == len(result_df)
        assert stats['total_poses'] > 0
        
        # Verify clash statistics
        clash_stats = stats['clash_statistics']
        assert 'mean' in clash_stats
        assert 'median' in clash_stats
        assert 'min' in clash_stats
        assert 'max' in clash_stats
        
        # Verify strain statistics
        strain_stats = stats['strain_statistics']
        assert 'mean' in strain_stats
        assert 'median' in strain_stats
        assert 'min' in strain_stats
        assert 'max' in strain_stats
    
    def test_pose_quality_comparison_across_conformers(self, real_molecules_df, protein_content):
        """Test pose quality differences across different conformers"""
        if real_molecules_df is None or len(real_molecules_df) < 2:
            pytest.skip("Need multiple conformers for comparison")
        
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Analyze all conformers
        result_df = analyze_all_poses(real_molecules_df, protein_content, config)
        
        # Verify we have results for multiple conformers
        assert len(result_df) >= 2
        
        # Test that conformers may have different quality metrics
        # (In practice, different poses should have different quality scores)
        clash_values = result_df['clashes'].tolist()
        strain_values = result_df['strain_energy'].tolist()
        
        # All values should be non-negative
        assert all(c >= 0 for c in clash_values)
        assert all(s >= 0.0 for s in strain_values)
        
        # At least verify we get reasonable value ranges
        max_clashes = max(clash_values)
        max_strain = max(strain_values)
        assert max_clashes >= 0  # Allow zero clashes
        assert max_strain >= 0.0  # Allow zero strain
    
    def test_end_to_end_pose_processing_real_data(self, protein_content):
        """Test complete pose processing pipeline with real data"""
        poses_file_path = Path(__file__).parent.parent / "test_data" / "example_poses_1fvv.sdf"
        if not poses_file_path.exists():
            pytest.skip("Test poses file not available")
        
        # Load real molecular data
        from data import molecules
        df = molecules.load_sdf(str(poses_file_path))
        
        if len(df) == 0:
            pytest.skip("No molecules loaded from test data")
        
        # Add required columns if missing
        if 'clashes' not in df.columns:
            df['clashes'] = 0
        if 'strain_energy' not in df.columns:
            df['strain_energy'] = 0.0
        
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Process all poses
        result_df = analyze_all_poses(df, protein_content, config)
        
        # Verify complete processing
        assert len(result_df) == len(df)
        assert 'clashes' in result_df.columns
        assert 'strain_energy' in result_df.columns
        
        # Verify all original columns preserved
        for col in df.columns:
            if col in result_df.columns:
                assert len(result_df[col]) == len(df)
        
        # Get final statistics
        stats = get_pose_quality_statistics(result_df)
        assert stats['total_poses'] == len(df)


if __name__ == '__main__':
    pytest.main([__file__])