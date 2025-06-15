import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem
import pandas as pd

from core.prolif_interactions import (
    is_prolif_available,
    create_ligand_sdf,
    extract_prolif_interactions,
    create_prolif_summary,
    calculate_prolif_interactions,
    get_prolif_interaction_types,
    PROLIF_AVAILABLE
)


class TestProlifInteractions:
    """Test suite for ProLIF interactions module"""
    
    @pytest.fixture
    def sample_molecule(self):
        """Create a sample RDKit molecule for testing"""
        smiles = "CCO"  # Simple ethanol molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
        return mol
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    @pytest.fixture
    def sample_ifp_dataframe(self):
        """Create a sample ProLIF fingerprint DataFrame for testing"""
        # Create mock DataFrame that resembles ProLIF output
        columns = [
            ('VAL123', 'Hydrophobic'),
            ('SER45', 'HBAcceptor'),
            ('TYR67', 'PiStacking'),
            ('LYS89', 'Cationic')
        ]
        data = [[1, 0, 1, 0]]  # Some interactions present, some not
        return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
    
    @pytest.fixture
    def sample_interaction_data(self):
        """Sample interaction data for testing"""
        return [
            {'type': 'Hydrophobic', 'residue': 'VAL123'},
            {'type': 'PiStacking', 'residue': 'TYR67'}
        ]
    
    def test_is_prolif_available(self):
        """Test ProLIF availability check"""
        result = is_prolif_available()
        assert isinstance(result, bool)
        assert result == PROLIF_AVAILABLE
    
    def test_create_ligand_sdf_valid_molecule(self, sample_molecule):
        """Test creating SDF file from valid molecule"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        sdf_path = create_ligand_sdf(sample_molecule)
        
        try:
            assert os.path.exists(sdf_path)
            assert sdf_path.endswith('.sdf')
            
            # Verify SDF file contents
            with open(sdf_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert 'M  END' in content
                
            # Verify we can read it back
            mol_from_file = Chem.MolFromMolFile(sdf_path)
            assert mol_from_file is not None
            
        finally:
            # Clean up
            if os.path.exists(sdf_path):
                os.unlink(sdf_path)
    
    def test_create_ligand_sdf_none_molecule(self):
        """Test creating SDF file with None molecule"""
        with pytest.raises((AttributeError, Exception)):  # RDKit may raise different exceptions
            create_ligand_sdf(None)
    
    def test_extract_prolif_interactions_with_data(self, sample_ifp_dataframe):
        """Test extracting interactions from ProLIF DataFrame with data"""
        interactions = extract_prolif_interactions(sample_ifp_dataframe)
        
        assert isinstance(interactions, list)
        assert len(interactions) == 2  # Only non-zero interactions
        
        # Check first interaction
        assert interactions[0]['type'] == 'Hydrophobic'
        assert interactions[0]['residue'] == 'VAL123'
        
        # Check second interaction
        assert interactions[1]['type'] == 'PiStacking'
        assert interactions[1]['residue'] == 'TYR67'
    
    def test_extract_prolif_interactions_empty_dataframe(self):
        """Test extracting interactions from empty DataFrame"""
        empty_df = pd.DataFrame()
        interactions = extract_prolif_interactions(empty_df)
        
        assert isinstance(interactions, list)
        assert len(interactions) == 0
    
    def test_extract_prolif_interactions_no_interactions(self):
        """Test extracting interactions when no interactions are present"""
        columns = [('VAL123', 'Hydrophobic'), ('SER45', 'HBAcceptor')]
        data = [[0, 0]]  # No interactions
        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
        
        interactions = extract_prolif_interactions(df)
        
        assert isinstance(interactions, list)
        assert len(interactions) == 0
    
    def test_create_prolif_summary(self, sample_interaction_data):
        """Test creating ProLIF summary from interaction data"""
        ifp_array = np.array([1, 0, 1, 0])
        
        summary = create_prolif_summary(sample_interaction_data, ifp_array)
        
        assert isinstance(summary, dict)
        assert 'total_interactions' in summary
        assert 'interaction_types' in summary
        assert 'interactions' in summary
        
        assert summary['total_interactions'] == 2
        assert summary['interaction_types']['Hydrophobic'] == 1
        assert summary['interaction_types']['PiStacking'] == 1
        assert summary['interactions'] == sample_interaction_data
    
    def test_create_prolif_summary_empty_data(self):
        """Test creating ProLIF summary with empty data"""
        ifp_array = np.array([0, 0, 0, 0])
        
        summary = create_prolif_summary([], ifp_array)
        
        assert summary['total_interactions'] == 0
        assert summary['interaction_types'] == {}
        assert summary['interactions'] == []
    
    def test_create_prolif_summary_multiple_same_type(self):
        """Test creating summary with multiple interactions of same type"""
        interaction_data = [
            {'type': 'Hydrophobic', 'residue': 'VAL123'},
            {'type': 'Hydrophobic', 'residue': 'LEU456'},
            {'type': 'HBAcceptor', 'residue': 'SER45'}
        ]
        ifp_array = np.array([1, 1, 1])
        
        summary = create_prolif_summary(interaction_data, ifp_array)
        
        assert summary['total_interactions'] == 3
        assert summary['interaction_types']['Hydrophobic'] == 2
        assert summary['interaction_types']['HBAcceptor'] == 1
    
    def test_get_prolif_interaction_types(self):
        """Test getting ProLIF interaction types"""
        types = get_prolif_interaction_types()
        
        assert isinstance(types, list)
        assert len(types) > 0
        
        expected_types = [
            'HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiStacking',
            'Anionic', 'Cationic', 'CationPi', 'PiCation',
            'XBAcceptor', 'XBDonor'
        ]
        
        for expected_type in expected_types:
            assert expected_type in types
    
    @pytest.mark.parametrize("prolif_available", [True, False])
    def test_calculate_prolif_interactions_availability(self, sample_molecule, protein_file_path, prolif_available):
        """Test calculate_prolif_interactions with different ProLIF availability"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with patch('core.prolif_interactions.PROLIF_AVAILABLE', prolif_available):
            if prolif_available:
                # Mock ProLIF components
                with patch('core.prolif_interactions.mda') as mock_mda, \
                     patch('core.prolif_interactions.plf') as mock_plf:
                    
                    # Mock MDAnalysis Universe
                    mock_protein_u = Mock()
                    mock_ligand_u = Mock()
                    mock_ligand_u.trajectory = [Mock()]
                    mock_mda.Universe.side_effect = [mock_protein_u, mock_ligand_u]
                    
                    # Mock ProLIF Fingerprint
                    mock_fp = Mock()
                    mock_plf.Fingerprint.return_value = mock_fp
                    
                    # Create mock DataFrame
                    columns = [('VAL123', 'Hydrophobic'), ('SER45', 'HBAcceptor')]
                    data = [[1, 0]]
                    mock_ifp_df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
                    mock_fp.run.return_value = mock_ifp_df
                    
                    ifp_array, summary = calculate_prolif_interactions(
                        str(protein_file_path), 
                        sample_molecule
                    )
                    
                    assert isinstance(ifp_array, np.ndarray)
                    assert isinstance(summary, dict)
                    assert 'total_interactions' in summary
                    assert 'interaction_types' in summary
                    assert 'interactions' in summary
            else:
                # Should raise ImportError when ProLIF not available
                with pytest.raises(ImportError, match="ProLIF is not available"):
                    calculate_prolif_interactions(
                        str(protein_file_path), 
                        sample_molecule
                    )
    
    def test_calculate_prolif_interactions_with_mock_prolif(self, sample_molecule, protein_file_path):
        """Test calculate_prolif_interactions with fully mocked ProLIF"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with patch('core.prolif_interactions.PROLIF_AVAILABLE', True), \
             patch('core.prolif_interactions.mda') as mock_mda, \
             patch('core.prolif_interactions.plf') as mock_plf:
            
            # Mock MDAnalysis Universe
            mock_protein_u = Mock()
            mock_ligand_u = Mock()
            mock_ligand_u.trajectory = [Mock()]
            mock_mda.Universe.side_effect = [mock_protein_u, mock_ligand_u]
            
            # Mock ProLIF Fingerprint
            mock_fp = Mock()
            mock_plf.Fingerprint.return_value = mock_fp
            
            # Create realistic mock DataFrame
            columns = [
                ('VAL123', 'Hydrophobic'),
                ('SER45', 'HBAcceptor'),
                ('TYR67', 'PiStacking'),
                ('LYS89', 'Cationic')
            ]
            data = [[1, 0, 1, 1]]
            mock_ifp_df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
            mock_fp.run.return_value = mock_ifp_df
            
            ifp_array, summary = calculate_prolif_interactions(
                str(protein_file_path), 
                sample_molecule,
                ligand_name="TEST_LIG"
            )
            
            # Verify results
            assert isinstance(ifp_array, np.ndarray)
            assert len(ifp_array) == 4
            assert np.array_equal(ifp_array, [1, 0, 1, 1])
            
            assert isinstance(summary, dict)
            assert summary['total_interactions'] == 3
            assert 'Hydrophobic' in summary['interaction_types']
            assert 'PiStacking' in summary['interaction_types']
            assert 'Cationic' in summary['interaction_types']
            assert 'HBAcceptor' not in summary['interaction_types']  # Was 0
            
            assert len(summary['interactions']) == 3
            
            # Verify MDAnalysis and ProLIF were called correctly
            assert mock_mda.Universe.call_count == 2
            mock_plf.Fingerprint.assert_called_once()
            mock_fp.run.assert_called_once()
    
    def test_calculate_prolif_interactions_file_cleanup(self, sample_molecule, protein_file_path):
        """Test that temporary files are cleaned up even if calculation fails"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with patch('core.prolif_interactions.PROLIF_AVAILABLE', True), \
             patch('core.prolif_interactions.mda') as mock_mda:
            
            # Make MDAnalysis raise an exception
            mock_mda.Universe.side_effect = Exception("Test error")
            
            # Track temporary files
            original_create_sdf = create_ligand_sdf
            created_files = []
            
            def track_create_sdf(mol):
                path = original_create_sdf(mol)
                created_files.append(path)
                return path
            
            with patch('core.prolif_interactions.create_ligand_sdf', side_effect=track_create_sdf):
                with pytest.raises(Exception, match="Test error"):
                    calculate_prolif_interactions(str(protein_file_path), sample_molecule)
                
                # Verify temporary file was cleaned up
                for temp_file in created_files:
                    assert not os.path.exists(temp_file)
    
    def test_calculate_prolif_interactions_custom_ligand_name(self, sample_molecule, protein_file_path):
        """Test calculate_prolif_interactions with custom ligand name"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with patch('core.prolif_interactions.PROLIF_AVAILABLE', True), \
             patch('core.prolif_interactions.mda') as mock_mda, \
             patch('core.prolif_interactions.plf') as mock_plf:
            
            # Setup mocks
            mock_protein_u = Mock()
            mock_ligand_u = Mock()
            mock_ligand_u.trajectory = [Mock()]
            mock_mda.Universe.side_effect = [mock_protein_u, mock_ligand_u]
            
            mock_fp = Mock()
            mock_plf.Fingerprint.return_value = mock_fp
            
            # Empty interactions
            columns = [('VAL123', 'Hydrophobic')]
            data = [[0]]
            mock_ifp_df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
            mock_fp.run.return_value = mock_ifp_df
            
            ifp_array, summary = calculate_prolif_interactions(
                str(protein_file_path), 
                sample_molecule,
                ligand_name="CUSTOM_LIG"
            )
            
            # Should work with custom ligand name
            assert isinstance(ifp_array, np.ndarray)
            assert isinstance(summary, dict)
            assert summary['total_interactions'] == 0


class TestProlifInteractionsWithRealData:
    """Integration tests using real test data"""
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    @pytest.fixture
    def sdf_file_path(self):
        """Path to test SDF file with molecules"""
        return Path(__file__).parent.parent / "test_data" / "example_poses_1fvv.sdf"
    
    def test_create_ligand_sdf_with_real_molecule(self, sdf_file_path):
        """Test creating SDF from real molecule data"""
        if not sdf_file_path.exists():
            pytest.skip("Test SDF file not available")
        
        # Read first molecule from test SDF
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        mol = next(supplier)
        
        if mol is None:
            pytest.skip("Could not read molecule from test SDF")
        
        # Test creating new SDF
        temp_sdf_path = create_ligand_sdf(mol)
        
        try:
            assert os.path.exists(temp_sdf_path)
            
            # Verify we can read the molecule back
            mol_from_temp = Chem.MolFromMolFile(temp_sdf_path)
            assert mol_from_temp is not None
            
            # Should have same number of atoms
            assert mol.GetNumAtoms() == mol_from_temp.GetNumAtoms()
            
        finally:
            if os.path.exists(temp_sdf_path):
                os.unlink(temp_sdf_path)
    
    def test_prolif_functions_integration(self, protein_file_path, sdf_file_path):
        """Test integration of ProLIF functions with real data"""
        if not (protein_file_path.exists() and sdf_file_path.exists()):
            pytest.skip("Test data files not available")
        
        # Read a test molecule
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        mol = next(supplier)
        
        if mol is None:
            pytest.skip("Could not read molecule from test SDF")
        
        # Test with mocked ProLIF to avoid dependency
        with patch('core.prolif_interactions.PROLIF_AVAILABLE', True), \
             patch('core.prolif_interactions.mda') as mock_mda, \
             patch('core.prolif_interactions.plf') as mock_plf:
            
            # Setup comprehensive mock
            mock_protein_u = Mock()
            mock_ligand_u = Mock()
            mock_ligand_u.trajectory = [Mock()]
            mock_mda.Universe.side_effect = [mock_protein_u, mock_ligand_u]
            
            mock_fp = Mock()
            mock_plf.Fingerprint.return_value = mock_fp
            
            # Create realistic interaction fingerprint
            columns = [
                ('ALA100', 'Hydrophobic'),
                ('SER101', 'HBAcceptor'),
                ('TYR102', 'PiStacking'),
                ('ASP103', 'Anionic'),
                ('LYS104', 'Cationic')
            ]
            data = [[1, 1, 0, 1, 0]]  # Mixed interactions
            mock_ifp_df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
            mock_fp.run.return_value = mock_ifp_df
            
            # Test full calculation
            ifp_array, summary = calculate_prolif_interactions(
                str(protein_file_path), 
                mol,
                ligand_name="TEST_MOLECULE"
            )
            
            # Validate results
            assert isinstance(ifp_array, np.ndarray)
            assert len(ifp_array) == 5
            expected_array = np.array([1, 1, 0, 1, 0])
            assert np.array_equal(ifp_array, expected_array)
            
            assert isinstance(summary, dict)
            assert summary['total_interactions'] == 3
            assert len(summary['interactions']) == 3
            
            # Check specific interaction types
            interaction_types = summary['interaction_types']
            assert interaction_types['Hydrophobic'] == 1
            assert interaction_types['HBAcceptor'] == 1
            assert interaction_types['Anionic'] == 1
            
            # Verify interactions list
            interactions = summary['interactions']
            residues = [int_data['residue'] for int_data in interactions]
            types = [int_data['type'] for int_data in interactions]
            
            assert 'ALA100' in residues
            assert 'SER101' in residues
            assert 'ASP103' in residues
            assert 'Hydrophobic' in types
            assert 'HBAcceptor' in types
            assert 'Anionic' in types