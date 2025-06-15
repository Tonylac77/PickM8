import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from utils.posecheck_utils import PoseCheckAnalyzer, _analyze_molecule_chunk
from rdkit import Chem


class TestPoseCheckAnalyzer:
    """Test suite for PoseCheckAnalyzer class"""
    
    @pytest.fixture
    def sample_protein_content(self):
        """Sample PDB protein content for testing"""
        return """CRYST1  184.952  184.952  212.832  90.00  90.00 120.00 P 62 2 2      1
ATOM      1  N   MET A   1     -14.162 190.982 101.556  1.00 59.83           N  
ATOM      2  CA  MET A   1     -13.221 191.499 102.594  1.00 61.86           C  
ATOM      3  C   MET A   1     -11.759 191.248 102.209  1.00 61.40           C  
ATOM      4  O   MET A   1     -10.826 191.795 102.812  1.00 58.64           O  
END
"""
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    @pytest.fixture
    def sdf_file_path(self):
        """Path to test SDF file with poses"""
        return Path(__file__).parent.parent / "test_data" / "example_poses_1fvv.sdf"
    
    @pytest.fixture
    def sample_mol_blocks(self, sdf_file_path):
        """Extract sample molecule blocks from SDF file"""
        mol_blocks = []
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        
        count = 0
        for mol in supplier:
            if mol is not None and count < 5:  # Get first 5 molecules
                mol_blocks.append(Chem.MolToMolBlock(mol))
                count += 1
        
        return mol_blocks
    
    @pytest.fixture
    def analyzer(self):
        """Create PoseCheckAnalyzer instance"""
        return PoseCheckAnalyzer()
    
    def test_init(self, analyzer):
        """Test PoseCheckAnalyzer initialization"""
        assert analyzer.pc is None
        assert analyzer.protein_loaded is False
        assert analyzer.protein_content is None
    
    @pytest.mark.parametrize("posecheck_available", [True, False])
    def test_load_protein_from_content(self, analyzer, sample_protein_content, posecheck_available):
        """Test loading protein from content string"""
        if posecheck_available:
            with patch('posecheck.PoseCheck') as mock_posecheck:
                mock_pc_instance = Mock()
                mock_pc_instance.load_protein_from_pdb.return_value = None
                mock_posecheck.return_value = mock_pc_instance
                
                analyzer.load_protein_from_content(sample_protein_content)
                
                assert analyzer.protein_loaded is True
                assert analyzer.protein_content == sample_protein_content
                assert analyzer.pc == mock_pc_instance
                mock_pc_instance.load_protein_from_pdb.assert_called_once()
        else:
            # Test ImportError case
            with patch('posecheck.PoseCheck', side_effect=ImportError()):
                analyzer.load_protein_from_content(sample_protein_content)
                
                assert analyzer.protein_loaded is False
                assert analyzer.pc is None
    
    def test_load_protein_from_file(self, analyzer, protein_file_path):
        """Test loading protein from actual PDB file"""
        with open(protein_file_path, 'r') as f:
            protein_content = f.read()
        
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(protein_content)
            
            assert analyzer.protein_loaded is True
            assert analyzer.protein_content == protein_content
    
    def test_analyze_molecule_no_protein_loaded(self, analyzer):
        """Test analyze_molecule when no protein is loaded"""
        mol_block = "test_mol_block"
        clashes, strain = analyzer.analyze_molecule(mol_block)
        
        assert clashes == 0
        assert strain == 0.0
    
    def test_analyze_molecule_with_mock_posecheck(self, analyzer, sample_protein_content, sample_mol_blocks):
        """Test analyze_molecule with mocked PoseCheck"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_pc_instance.load_ligands_from_mols.return_value = None
            mock_pc_instance.calculate_clashes.return_value = [5]
            mock_pc_instance.calculate_strain_energy.return_value = [12.5]
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            if sample_mol_blocks:
                clashes, strain = analyzer.analyze_molecule(sample_mol_blocks[0])
                
                assert clashes == 5
                assert strain == 12.5
                mock_pc_instance.load_ligands_from_mols.assert_called()
                mock_pc_instance.calculate_clashes.assert_called()
                mock_pc_instance.calculate_strain_energy.assert_called()
    
    def test_analyze_molecule_invalid_mol_block(self, analyzer, sample_protein_content):
        """Test analyze_molecule with invalid molecule block"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            # Invalid mol block should return 0, 0.0
            clashes, strain = analyzer.analyze_molecule("invalid_mol_block")
            
            assert clashes == 0
            assert strain == 0.0
    
    def test_analyze_multiple_molecules_no_protein(self, analyzer, sample_mol_blocks):
        """Test analyze_multiple_molecules when no protein is loaded"""
        if sample_mol_blocks:
            clashes, strains = analyzer.analyze_multiple_molecules(sample_mol_blocks[:3])
            
            assert len(clashes) == 3
            assert len(strains) == 3
            assert all(c == 0 for c in clashes)
            assert all(s == 0.0 for s in strains)
    
    def test_analyze_multiple_molecules_with_mock(self, analyzer, sample_protein_content, sample_mol_blocks):
        """Test analyze_multiple_molecules with mocked PoseCheck"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_pc_instance.load_ligands_from_mols.return_value = None
            mock_pc_instance.calculate_clashes.return_value = [1, 2, 3]
            mock_pc_instance.calculate_strain_energy.return_value = [1.1, 2.2, 3.3]
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            if len(sample_mol_blocks) >= 3:
                clashes, strains = analyzer.analyze_multiple_molecules(sample_mol_blocks[:3])
                
                assert len(clashes) == 3
                assert len(strains) == 3
                assert clashes == [1, 2, 3]
                assert strains == [1.1, 2.2, 3.3]
    
    def test_analyze_multiple_molecules_with_none_mols(self, analyzer, sample_protein_content):
        """Test analyze_multiple_molecules with None molecules"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            # Mix of valid and invalid mol blocks
            mol_blocks = ["invalid_block", None, "another_invalid"]
            clashes, strains = analyzer.analyze_multiple_molecules(mol_blocks)
            
            assert len(clashes) == 3
            assert len(strains) == 3
            assert all(c == 0 for c in clashes)
            assert all(s == 0.0 for s in strains)
    
    def test_analyze_multiple_molecules_parallel_small_dataset(self, analyzer, sample_protein_content, sample_mol_blocks):
        """Test parallel processing falls back to sequential for small datasets"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            with patch.object(analyzer, 'analyze_multiple_molecules') as mock_sequential:
                mock_sequential.return_value = ([1, 2, 3], [1.1, 2.2, 3.3])
                
                if len(sample_mol_blocks) >= 3:
                    clashes, strains = analyzer.analyze_multiple_molecules_parallel(sample_mol_blocks[:3])
                    
                    mock_sequential.assert_called_once()
                    assert clashes == [1, 2, 3]
                    assert strains == [1.1, 2.2, 3.3]
    
    def test_analyze_multiple_molecules_parallel_no_protein(self, analyzer, sample_mol_blocks):
        """Test parallel processing when no protein is loaded"""
        if sample_mol_blocks:
            clashes, strains = analyzer.analyze_multiple_molecules_parallel(sample_mol_blocks[:3])
            
            assert len(clashes) == 3
            assert len(strains) == 3
            assert all(c == 0 for c in clashes)
            assert all(s == 0.0 for s in strains)
    
    def test_analyze_multiple_molecules_parallel_empty_list(self, analyzer, sample_protein_content):
        """Test parallel processing with empty molecule list"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            clashes, strains = analyzer.analyze_multiple_molecules_parallel([])
            
            assert clashes == []
            assert strains == []
    
    @patch('utils.posecheck_utils.mp.cpu_count')
    def test_analyze_multiple_molecules_parallel_worker_calculation(self, mock_cpu_count, analyzer, sample_protein_content):
        """Test worker count calculation in parallel processing"""
        mock_cpu_count.return_value = 8
        
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            # Create a large enough dataset to trigger parallel processing
            large_mol_blocks = ["dummy_block"] * 100
            
            # Test that the method can be called without errors
            clashes, strains = analyzer.analyze_multiple_molecules_parallel(large_mol_blocks)
            
            # Verify results structure
            assert len(clashes) == 100
            assert len(strains) == 100
            assert all(isinstance(c, int) for c in clashes)
            assert all(isinstance(s, float) for s in strains)
    
    def test_analyze_multiple_molecules_smart_small_dataset(self, analyzer, sample_protein_content, sample_mol_blocks):
        """Test smart analysis chooses sequential for small dataset"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            with patch.object(analyzer, 'analyze_multiple_molecules') as mock_sequential:
                mock_sequential.return_value = ([1, 2, 3], [1.1, 2.2, 3.3])
                
                if len(sample_mol_blocks) >= 3:
                    clashes, strains = analyzer.analyze_multiple_molecules_smart(sample_mol_blocks[:3])
                    
                    mock_sequential.assert_called_once()
                    assert clashes == [1, 2, 3]
                    assert strains == [1.1, 2.2, 3.3]
    
    def test_analyze_multiple_molecules_smart_large_dataset(self, analyzer, sample_protein_content):
        """Test smart analysis chooses parallel for large dataset"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(sample_protein_content)
            
            with patch.object(analyzer, 'analyze_multiple_molecules_parallel') as mock_parallel:
                mock_parallel.return_value = ([1] * 100, [1.1] * 100)
                
                large_mol_blocks = ["dummy_block"] * 100
                clashes, strains = analyzer.analyze_multiple_molecules_smart(large_mol_blocks, parallel_threshold=50)
                
                mock_parallel.assert_called_once()
                assert len(clashes) == 100
                assert len(strains) == 100


class TestAnalyzeMoleculeChunk:
    """Test suite for _analyze_molecule_chunk function"""
    
    @pytest.fixture
    def sample_protein_content(self):
        """Sample PDB protein content for testing"""
        return """CRYST1  184.952  184.952  212.832  90.00  90.00 120.00 P 62 2 2      1
ATOM      1  N   MET A   1     -14.162 190.982 101.556  1.00 59.83           N  
ATOM      2  CA  MET A   1     -13.221 191.499 102.594  1.00 61.86           C  
END
"""
    
    @pytest.fixture
    def sample_mol_blocks(self):
        """Sample molecule blocks for testing"""
        return [
            """
  Mrv2014 01012021

  1  0  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
""",
            None,  # Test None handling
            "invalid_mol_block"  # Test invalid block
        ]
    
    def test_analyze_molecule_chunk_posecheck_not_available(self, sample_protein_content, sample_mol_blocks):
        """Test _analyze_molecule_chunk when PoseCheck is not available"""
        with patch('posecheck.PoseCheck', side_effect=ImportError()):
            results = _analyze_molecule_chunk(sample_protein_content, sample_mol_blocks)
            
            assert len(results) == 3
            assert all(result == (0, 0.0) for result in results)
    
    def test_analyze_molecule_chunk_with_mock_posecheck(self, sample_protein_content, sample_mol_blocks):
        """Test _analyze_molecule_chunk with mocked PoseCheck"""
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_pc_instance.load_ligands_from_mols.return_value = None
            mock_pc_instance.calculate_clashes.return_value = [2]
            mock_pc_instance.calculate_strain_energy.return_value = [5.5]
            mock_posecheck.return_value = mock_pc_instance
            
            with patch('utils.posecheck_utils.Path') as mock_path:
                mock_path_instance = Mock()
                mock_path_instance.unlink.return_value = None
                mock_path.return_value = mock_path_instance
                
                results = _analyze_molecule_chunk(sample_protein_content, sample_mol_blocks)
                
                assert len(results) == 3
                # First result should be valid, others should be (0, 0.0) due to None/invalid blocks
                assert results[1] == (0, 0.0)  # None block
                assert results[2] == (0, 0.0)  # Invalid block
    
    def test_analyze_molecule_chunk_exception_handling(self, sample_protein_content, sample_mol_blocks):
        """Test _analyze_molecule_chunk exception handling"""
        with patch('posecheck.PoseCheck', side_effect=Exception("Test error")):
            results = _analyze_molecule_chunk(sample_protein_content, sample_mol_blocks)
            
            assert len(results) == 3
            assert all(result == (0, 0.0) for result in results)


class TestIntegrationWithRealData:
    """Integration tests using real test data files"""
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    @pytest.fixture
    def sdf_file_path(self):
        """Path to test SDF file with poses"""
        return Path(__file__).parent.parent / "test_data" / "example_poses_1fvv.sdf"
    
    def test_load_real_protein_file(self, protein_file_path):
        """Test loading real protein file"""
        analyzer = PoseCheckAnalyzer()
        
        if protein_file_path.exists():
            with open(protein_file_path, 'r') as f:
                protein_content = f.read()
            
            with patch('posecheck.PoseCheck') as mock_posecheck:
                mock_pc_instance = Mock()
                mock_pc_instance.load_protein_from_pdb.return_value = None
                mock_posecheck.return_value = mock_pc_instance
                
                analyzer.load_protein_from_content(protein_content)
                
                assert analyzer.protein_loaded is True
                assert len(analyzer.protein_content) > 0
                assert "ATOM" in analyzer.protein_content
    
    def test_read_real_sdf_molecules(self, sdf_file_path):
        """Test reading molecules from real SDF file"""
        if sdf_file_path.exists():
            supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
            mol_count = 0
            
            for mol in supplier:
                if mol is not None:
                    mol_count += 1
                    
                    # Test that we can create mol blocks
                    mol_block = Chem.MolToMolBlock(mol)
                    assert len(mol_block) > 0
                    assert "BEGIN" in mol_block or "M  END" in mol_block
                    
                if mol_count >= 5:  # Test first 5 molecules
                    break
            
            assert mol_count > 0
    
    def test_full_workflow_with_mocked_posecheck(self, protein_file_path, sdf_file_path):
        """Test full workflow with real data but mocked PoseCheck"""
        if not (protein_file_path.exists() and sdf_file_path.exists()):
            pytest.skip("Test data files not available")
        
        analyzer = PoseCheckAnalyzer()
        
        # Load protein
        with open(protein_file_path, 'r') as f:
            protein_content = f.read()
        
        # Load molecules
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        mol_blocks = []
        
        count = 0
        for mol in supplier:
            if mol is not None and count < 3:
                mol_blocks.append(Chem.MolToMolBlock(mol))
                count += 1
        
        # Mock PoseCheck and test workflow
        with patch('posecheck.PoseCheck') as mock_posecheck:
            mock_pc_instance = Mock()
            mock_pc_instance.load_protein_from_pdb.return_value = None
            mock_pc_instance.load_ligands_from_mols.return_value = None
            mock_pc_instance.calculate_clashes.return_value = [1, 2, 3]
            mock_pc_instance.calculate_strain_energy.return_value = [1.5, 2.5, 3.5]
            mock_posecheck.return_value = mock_pc_instance
            
            analyzer.load_protein_from_content(protein_content)
            clashes, strains = analyzer.analyze_multiple_molecules(mol_blocks)
            
            assert len(clashes) == 3
            assert len(strains) == 3
            assert clashes == [1, 2, 3]
            assert strains == [1.5, 2.5, 3.5]