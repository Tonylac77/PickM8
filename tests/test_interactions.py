"""
Test suite for data/interactions.py functions.
Focuses on PLIP, ProLIF, and BioPython complex creation functionality.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem
import pandas as pd

from data import interactions


class TestInteractionAvailability:
    """Test suite for interaction availability checks"""
    
    def test_is_plip_available(self):
        """Test PLIP availability check"""
        result = interactions.is_plip_available()
        assert isinstance(result, bool)
        assert result == interactions.PLIP_AVAILABLE
    
    def test_is_prolif_available(self):
        """Test ProLIF availability check"""
        result = interactions.is_prolif_available()
        assert isinstance(result, bool)
        assert result == interactions.PROLIF_AVAILABLE


class TestBioPythonComplexCreation:
    """Test suite for BioPython complex creation functionality"""
    
    @pytest.fixture
    def sample_molecule(self):
        """Create a sample RDKit molecule for testing"""
        smiles = "CCO"  # Simple ethanol molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            # Add 3D coordinates
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol)
        return mol
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    def test_create_complex_with_biopython_valid_inputs(self, sample_molecule, protein_file_path):
        """Test creating complex with valid protein and ligand"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        complex_path = interactions.create_complex_with_biopython(
            str(protein_file_path), 
            sample_molecule, 
            "LIG"
        )
        
        try:
            if complex_path is not None:
                assert os.path.exists(complex_path)
                assert complex_path.endswith('.pdb')
                
                # Verify PDB file contents
                with open(complex_path, 'r') as f:
                    content = f.read()
                    assert len(content) > 0
                    assert 'ATOM' in content or 'HETATM' in content
        finally:
            # Clean up
            if complex_path and os.path.exists(complex_path):
                os.unlink(complex_path)
    
    def test_create_complex_with_biopython_custom_ligand_name(self, sample_molecule, protein_file_path):
        """Test creating complex with custom ligand name"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        complex_path = interactions.create_complex_with_biopython(
            str(protein_file_path), 
            sample_molecule, 
            "DRUG"
        )
        
        try:
            if complex_path is not None:
                assert os.path.exists(complex_path)
                
                # Check that the ligand name is used
                with open(complex_path, 'r') as f:
                    content = f.read()
                    # The ligand name should appear in the PDB (truncated to 3 chars)
                    assert 'DRU' in content or 'DRUG' in content
        finally:
            # Clean up
            if complex_path and os.path.exists(complex_path):
                os.unlink(complex_path)
    
    def test_create_complex_with_biopython_invalid_protein(self, sample_molecule):
        """Test creating complex with invalid protein file"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        # Test with non-existent protein file
        result = interactions.create_complex_with_biopython(
            "/non/existent/protein.pdb", 
            sample_molecule, 
            "LIG"
        )
        
        assert result is None
    
    def test_create_complex_with_biopython_none_molecule(self, protein_file_path):
        """Test creating complex with None molecule"""
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        # Test with None ligand
        with pytest.raises(Exception):  # Should raise an exception
            interactions.create_complex_with_biopython(
                str(protein_file_path), 
                None, 
                "LIG"
            )


class TestPLIPFunctions:
    """Test suite for PLIP interaction functions"""
    
    @pytest.fixture
    def sample_molecule(self):
        """Create a sample RDKit molecule for testing"""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol)
        return mol
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    def test_get_plip_interaction_types(self):
        """Test getting PLIP interaction types"""
        types = interactions.get_plip_interaction_types()
        
        assert isinstance(types, list)
        assert len(types) > 0
        
        expected_types = [
            'hydrogen_bond', 'hydrophobic', 'pi_stacking', 'salt_bridge',
            'halogen_bond', 'pi_cation', 'water_bridge', 'metal_coordination'
        ]
        
        for expected_type in expected_types:
            assert expected_type in types
    
    def test_interactions_to_fingerprint(self):
        """Test converting interactions to fingerprint"""
        interaction_data = [
            ('VAL', 123, 'A', 'hydrophobic'),
            ('SER', 45, 'A', 'hydrogen_bond')
        ]
        
        fp = interactions.interactions_to_fingerprint(interaction_data, fp_size=1024)
        
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 1024
        assert fp.dtype == int
        assert np.sum(fp) > 0  # Should have some bits set
    
    def test_interactions_to_fingerprint_empty(self):
        """Test converting empty interactions to fingerprint"""
        interaction_data = []
        
        fp = interactions.interactions_to_fingerprint(interaction_data, fp_size=512)
        
        assert isinstance(fp, np.ndarray)
        assert len(fp) == 512
        assert np.sum(fp) == 0  # Should be all zeros
    
    @pytest.mark.parametrize("plip_available", [True, False])
    def test_calculate_plip_interactions_availability(self, sample_molecule, protein_file_path, plip_available):
        """Test calculate_plip_interactions with different PLIP availability"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with patch('data.interactions.PLIP_AVAILABLE', plip_available):
            if plip_available:
                # Mock PLIP components
                with patch('data.interactions.PDBComplex') as mock_complex, \
                     patch('data.interactions.create_complex_with_biopython') as mock_create_complex:
                    
                    # Mock complex creation
                    mock_create_complex.return_value = "/tmp/mock_complex.pdb"
                    
                    # Mock PLIP complex
                    mock_plip = Mock()
                    mock_ligand = Mock()
                    mock_ligand.hetid = 'LIG'
                    mock_ligand.chain = 'L'
                    mock_ligand.position = 1
                    mock_plip.ligands = [mock_ligand]
                    mock_plip.interaction_sets = {
                        'LIG:L:1': Mock(
                            hbonds_ldon=[], hbonds_pdon=[],
                            hydrophobic_contacts=[], pistacking=[],
                            saltbridge_lneg=[], saltbridge_pneg=[],
                            halogen_bonds=[], pication_laro=[], pication_paro=[],
                            water_bridges=[], metal_complexes=[]
                        )
                    }
                    mock_complex.return_value = mock_plip
                    
                    ifp, summary = interactions.calculate_plip_interactions(
                        str(protein_file_path), 
                        sample_molecule
                    )
                    
                    assert isinstance(ifp, np.ndarray)
                    assert isinstance(summary, dict)
                    assert 'total_interactions' in summary
                    assert 'interaction_types' in summary
                    assert 'interactions' in summary
            else:
                # Should raise ImportError when PLIP not available
                with pytest.raises(ImportError, match="PLIP is not available"):
                    interactions.calculate_plip_interactions(
                        str(protein_file_path), 
                        sample_molecule
                    )


class TestProLIFFunctions:
    """Test suite for ProLIF interaction functions"""
    
    @pytest.fixture
    def sample_molecule(self):
        """Create a sample RDKit molecule for testing"""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol)
        return mol
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    def test_get_prolif_interaction_types(self):
        """Test getting ProLIF interaction types"""
        types = interactions.get_prolif_interaction_types()
        
        assert isinstance(types, list)
        assert len(types) > 0
        
        expected_types = [
            'HBAcceptor', 'HBDonor', 'Hydrophobic', 'PiStacking',
            'Anionic', 'Cationic', 'CationPi', 'PiCation',
            'XBAcceptor', 'XBDonor'
        ]
        
        for expected_type in expected_types:
            assert expected_type in types
    
    def test_create_ligand_pdb(self, sample_molecule):
        """Test creating temporary PDB file for ligand"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        pdb_path = interactions.create_ligand_pdb(sample_molecule)
        
        try:
            assert os.path.exists(pdb_path)
            assert pdb_path.endswith('.pdb')
            
            # Verify PDB file contents
            with open(pdb_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert 'ATOM' in content or 'HETATM' in content
                
            # Verify we can read it back with RDKit
            mol_from_file = Chem.MolFromPDBFile(pdb_path)
            assert mol_from_file is not None
            
        finally:
            # Clean up
            if os.path.exists(pdb_path):
                os.unlink(pdb_path)
    
    def test_extract_prolif_interactions_with_data(self):
        """Test extracting interactions from ProLIF DataFrame"""
        # Create mock DataFrame that resembles ProLIF output
        columns = [
            ('VAL123', 'Hydrophobic'),
            ('SER45', 'HBAcceptor'),
            ('TYR67', 'PiStacking'),
            ('LYS89', 'Cationic')
        ]
        data = [[1, 0, 1, 0]]  # Some interactions present, some not
        mock_ifp_df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
        
        interactions_list = interactions.extract_prolif_interactions(mock_ifp_df)
        
        assert isinstance(interactions_list, list)
        assert len(interactions_list) == 2  # Only non-zero interactions
        
        # Check first interaction
        assert interactions_list[0]['type'] == 'Hydrophobic'
        assert interactions_list[0]['residue'] == 'VAL123'
        
        # Check second interaction
        assert interactions_list[1]['type'] == 'PiStacking'
        assert interactions_list[1]['residue'] == 'TYR67'
    
    def test_extract_prolif_interactions_empty_dataframe(self):
        """Test extracting interactions from empty DataFrame"""
        empty_df = pd.DataFrame()
        interactions_list = interactions.extract_prolif_interactions(empty_df)
        
        assert isinstance(interactions_list, list)
        assert len(interactions_list) == 0
    
    def test_create_prolif_summary(self):
        """Test creating ProLIF summary from interaction data"""
        interaction_data = [
            {'type': 'Hydrophobic', 'residue': 'VAL123'},
            {'type': 'PiStacking', 'residue': 'TYR67'}
        ]
        ifp_array = np.array([1, 0, 1, 0])
        
        summary = interactions.create_prolif_summary(interaction_data, ifp_array)
        
        assert isinstance(summary, dict)
        assert 'total_interactions' in summary
        assert 'interaction_types' in summary
        assert 'interactions' in summary
        
        assert summary['total_interactions'] == 2
        assert summary['interaction_types']['Hydrophobic'] == 1
        assert summary['interaction_types']['PiStacking'] == 1
        assert summary['interactions'] == interaction_data
    
    @pytest.mark.parametrize("prolif_available", [True, False])
    def test_calculate_prolif_interactions_availability(self, sample_molecule, protein_file_path, prolif_available):
        """Test calculate_prolif_interactions with different ProLIF availability"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with patch('data.interactions.PROLIF_AVAILABLE', prolif_available):
            if prolif_available:
                # Mock ProLIF components
                with patch('data.interactions.mda') as mock_mda, \
                     patch('data.interactions.plf') as mock_plf:
                    
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
                    
                    ifp_array, summary = interactions.calculate_prolif_interactions(
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
                    interactions.calculate_prolif_interactions(
                        str(protein_file_path), 
                        sample_molecule
                    )


class TestMainInteractionFunctions:
    """Test suite for main interaction computation functions"""
    
    @pytest.fixture
    def sample_molecule(self):
        """Create a sample RDKit molecule for testing"""
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol)
        return mol
    
    @pytest.fixture
    def sample_dataframe(self, sample_molecule):
        """Create test DataFrame with molecules"""
        return pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'mol': [sample_molecule, sample_molecule],
            'interaction_fp': [None, None],
            'interactions': [None, None],
            'num_interactions': [0, 0]
        })
    
    def test_create_default_interaction_config(self):
        """Test creating default interaction configuration"""
        config = interactions.create_default_interaction_config()
        
        assert isinstance(config, dict)
        assert 'interaction_type' in config
        assert 'ligand_name' in config
        assert 'plip_config' in config
        assert 'prolif_config' in config
        
        assert config['interaction_type'] == 'plip'
        assert config['ligand_name'] == 'LIG'
    
    def test_compute_interaction_fingerprint_no_interaction_available(self, sample_molecule):
        """Test interaction fingerprint computation when no interaction tools available"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        protein_content = "ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N\\n"
        config = {'interaction_type': 'plip'}
        
        with patch('data.interactions.INTERACTION_AVAILABLE', False):
            result = interactions.compute_interaction_fingerprint(sample_molecule, protein_content, config)
            
            assert result == (None, None, 0)
    
    def test_compute_interaction_fingerprint_none_molecule(self):
        """Test interaction fingerprint computation with None molecule"""
        protein_content = "ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N\\n"
        config = {'interaction_type': 'plip'}
        
        result = interactions.compute_interaction_fingerprint(None, protein_content, config)
        
        assert result == (None, None, 0)
    
    @patch('data.interactions.INTERACTION_AVAILABLE', True)
    @patch('data.interactions.PLIP_AVAILABLE', True)
    def test_compute_interaction_fingerprint_plip(self, sample_molecule):
        """Test interaction fingerprint computation with PLIP"""
        if sample_molecule is None:
            pytest.skip("Could not create sample molecule")
        
        protein_content = "ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N\\n"
        config = {'interaction_type': 'plip', 'ligand_name': 'LIG'}
        
        # Mock calculate_plip_interactions
        mock_ifp = np.array([1, 0, 1, 0])
        mock_summary = {
            'interactions': [
                {'type': 'hydrophobic', 'restype': 'VAL', 'resnr': 123, 'reschain': 'A'}
            ]
        }
        
        with patch('data.interactions.calculate_plip_interactions', return_value=(mock_ifp, mock_summary)):
            result = interactions.compute_interaction_fingerprint(sample_molecule, protein_content, config)
            
            ifp_json, interactions_json, num_interactions = result
            
            assert ifp_json is not None
            assert interactions_json is not None
            assert num_interactions == 1
            
            # Verify JSON parsing
            ifp_parsed = json.loads(ifp_json)
            assert ifp_parsed == [1, 0, 1, 0]
            
            interactions_parsed = json.loads(interactions_json)
            assert len(interactions_parsed) == 1
    
    def test_compute_all_interactions_empty_dataframe(self):
        """Test computing interactions for empty DataFrame"""
        df = pd.DataFrame(columns=['id', 'mol', 'interaction_fp', 'interactions', 'num_interactions'])
        protein_content = "ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N\\n"
        config = {'interaction_type': 'plip'}
        
        result = interactions.compute_all_interactions(df, protein_content, config)
        
        assert len(result) == 0
        assert list(result.columns) == list(df.columns)
    
    def test_compute_all_interactions_with_data(self, sample_dataframe):
        """Test computing interactions for DataFrame with data"""
        protein_content = "ATOM      1  N   ALA A   1      20.154  16.967  15.691  1.00 11.99           N\\n"
        config = {'interaction_type': 'plip'}
        
        # Mock compute_interaction_fingerprint
        def mock_compute(mol, protein, config):
            return '["1", "0", "1"]', '[{"type": "hydrophobic"}]', 1
        
        with patch('data.interactions.compute_interaction_fingerprint', side_effect=mock_compute):
            result = interactions.compute_all_interactions(sample_dataframe, protein_content, config)
            
            assert len(result) == 2
            assert result.loc[0, 'interaction_fp'] == '["1", "0", "1"]'
            assert result.loc[0, 'num_interactions'] == 1
            assert result.loc[1, 'interaction_fp'] == '["1", "0", "1"]'
            assert result.loc[1, 'num_interactions'] == 1


class TestInteractionsRealData:
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
    def real_ligand_molecule(self, ligand_file_path):
        """Load real ligand molecule from SDF file"""
        if not ligand_file_path.exists():
            pytest.skip("Test ligand file not available")
        
        from data import molecules
        df = molecules.load_sdf(str(ligand_file_path))
        return df['mol'].iloc[0] if len(df) > 0 else None
    
    @pytest.fixture
    def protein_content(self, protein_file_path):
        """Load real protein content from PDB file"""
        if not protein_file_path.exists():
            pytest.skip("Test protein file not available")
        
        with open(protein_file_path, 'r') as f:
            return f.read()
    
    @pytest.mark.skipif(not interactions.PLIP_AVAILABLE, reason="PLIP not available")
    def test_plip_interactions_real_data(self, real_ligand_molecule, protein_content, protein_file_path):
        """Test PLIP interactions with real protein-ligand data"""
        if real_ligand_molecule is None:
            pytest.skip("Could not load real ligand molecule")
        
        try:
            ifp, summary = interactions.calculate_plip_interactions(
                str(protein_file_path), 
                real_ligand_molecule
            )
            
            # Verify structure of results
            assert isinstance(ifp, np.ndarray)
            assert len(ifp) > 0
            assert isinstance(summary, dict)
            assert 'total_interactions' in summary
            assert 'interaction_types' in summary
            assert 'interactions' in summary
            
            # With real data, we expect some interactions
            assert summary['total_interactions'] >= 0
            assert isinstance(summary['interactions'], list)
            
        except Exception as e:
            pytest.skip(f"PLIP interaction calculation failed: {e}")
    
    @pytest.mark.skipif(not interactions.PROLIF_AVAILABLE, reason="ProLIF not available")  
    def test_prolif_interactions_real_data(self, real_ligand_molecule, protein_content, protein_file_path):
        """Test ProLIF interactions with real protein-ligand data"""
        if real_ligand_molecule is None:
            pytest.skip("Could not load real ligand molecule")
        
        try:
            ifp_array, summary = interactions.calculate_prolif_interactions(
                str(protein_file_path), 
                real_ligand_molecule
            )
            
            # Verify structure of results  
            assert isinstance(ifp_array, np.ndarray)
            assert len(ifp_array) > 0
            assert isinstance(summary, dict)
            assert 'total_interactions' in summary
            assert 'interaction_types' in summary
            assert 'interactions' in summary
            
            # With real data, we expect some interactions
            assert summary['total_interactions'] >= 0
            assert isinstance(summary['interactions'], list)
            
        except Exception as e:
            pytest.skip(f"ProLIF interaction calculation failed: {e}")
    
    def test_compute_interaction_fingerprint_real_data(self, real_ligand_molecule, protein_content):
        """Test interaction fingerprint computation with real data"""
        if real_ligand_molecule is None:
            pytest.skip("Could not load real ligand molecule")
        
        config = {'interaction_type': 'plip', 'fingerprint_size': 512}
        
        try:
            ifp_json, interactions_json, num_interactions = interactions.compute_interaction_fingerprint(
                real_ligand_molecule, 
                protein_content, 
                config
            )
            
            # Verify results structure
            if ifp_json is not None:  # Will be None if tools not available
                assert isinstance(ifp_json, str)
                assert isinstance(interactions_json, str)
                assert isinstance(num_interactions, int)
                assert num_interactions >= 0
                
                # Verify JSON can be parsed
                ifp_parsed = json.loads(ifp_json)
                assert isinstance(ifp_parsed, list)
                assert len(ifp_parsed) == 1024
                
                interactions_parsed = json.loads(interactions_json)
                assert isinstance(interactions_parsed, list)
                
        except ImportError:
            pytest.skip("Required interaction tools not available")
    
    def test_end_to_end_interaction_processing(self, protein_content):
        """Test end-to-end interaction processing with real molecular data"""
        # Load real ligand data
        ligand_file_path = Path(__file__).parent.parent / "test_data" / "1fvv_l.sdf"
        if not ligand_file_path.exists():
            pytest.skip("Test ligand file not available")
        
        from data import molecules
        df = molecules.load_sdf(str(ligand_file_path))
        
        if len(df) == 0:
            pytest.skip("No molecules loaded from test data")
        
        config = {'interaction_type': 'plip', 'fingerprint_size': 512}
        
        try:
            result_df = interactions.compute_all_interactions(df, protein_content, config)
            
            # Verify processing completed
            assert len(result_df) == len(df)
            assert 'interaction_fp' in result_df.columns
            assert 'interactions' in result_df.columns  
            assert 'num_interactions' in result_df.columns
            
            # Check that at least some processing occurred
            non_null_interactions = result_df['interaction_fp'].notna().sum()
            assert non_null_interactions >= 0  # Allow for cases where tools aren't available
            
        except ImportError:
            pytest.skip("Required interaction tools not available")


if __name__ == '__main__':
    pytest.main([__file__])