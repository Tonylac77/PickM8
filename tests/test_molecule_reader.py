import pytest
import tempfile
import os
from pathlib import Path
from utils.io_handlers import MoleculeReader


class TestMoleculeReader:
    """Test suite for MoleculeReader class"""
    
    @pytest.fixture
    def test_sdf_path(self):
        """Path to test SDF file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_l.sdf"
    
    @pytest.fixture
    def temp_pdb_file(self):
        """Create temporary PDB file for testing"""
        pdb_content = """HEADER    TEST PDB FILE
ATOM      1  N   ALA A   1      20.154  16.967  25.245  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  25.745  1.00 10.00           C  
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)
    
    def test_get_sdf_properties_basic(self, test_sdf_path):
        """Test basic functionality of get_sdf_properties"""
        properties = MoleculeReader.get_sdf_properties(str(test_sdf_path))
        
        assert isinstance(properties, list)
        assert len(properties) > 0
        assert all(isinstance(prop, str) for prop in properties)
    
    def test_get_sdf_properties_expected_properties(self, test_sdf_path):
        """Test that expected properties are found in 1fvv_l.sdf"""
        properties = MoleculeReader.get_sdf_properties(str(test_sdf_path))
        
        expected_properties = [
            'model_server_stats.create_model_time_ms',
            'model_server_stats.element_count', 
            'model_server_stats.encode_time_ms',
            'model_server_stats.io_time_ms',
            'model_server_stats.parse_time_ms',
            'model_server_stats.query_time_ms'
        ]
        
        for prop in expected_properties:
            assert prop in properties
    
    def test_get_sdf_properties_max_molecules(self, test_sdf_path):
        """Test max_molecules parameter"""
        properties_max_1 = MoleculeReader.get_sdf_properties(str(test_sdf_path), max_molecules=1)
        properties_max_5 = MoleculeReader.get_sdf_properties(str(test_sdf_path), max_molecules=5)
        
        assert isinstance(properties_max_1, list)
        assert isinstance(properties_max_5, list)
        assert len(properties_max_1) == len(properties_max_5)
    
    def test_get_sdf_properties_nonexistent_file(self):
        """Test behavior with nonexistent file"""
        properties = MoleculeReader.get_sdf_properties("nonexistent_file.sdf")
        assert properties == []
    
    def test_read_sdf_basic(self, test_sdf_path):
        """Test basic functionality of read_sdf"""
        molecules = MoleculeReader.read_sdf(str(test_sdf_path))
        
        assert isinstance(molecules, list)
        assert len(molecules) == 1
        
        mol = molecules[0]
        assert 'id' in mol
        assert 'name' in mol
        assert 'mol_block' in mol
        assert 'smiles' in mol
        assert 'score' in mol
    
    def test_read_sdf_molecule_data_structure(self, test_sdf_path):
        """Test structure of returned molecule data"""
        molecules = MoleculeReader.read_sdf(str(test_sdf_path))
        mol = molecules[0]
        
        assert isinstance(mol['id'], int)
        assert isinstance(mol['name'], str) 
        assert isinstance(mol['mol_block'], str)
        assert isinstance(mol['smiles'], str)
        assert isinstance(mol['score'], (int, float))
        
        assert mol['id'] == 0
        assert mol['name'] == "107"
        assert len(mol['mol_block']) > 0
        assert len(mol['smiles']) > 0
    
    def test_read_sdf_properties_extraction(self, test_sdf_path):
        """Test that properties are correctly extracted with prop_ prefix"""
        molecules = MoleculeReader.read_sdf(str(test_sdf_path))
        mol = molecules[0]
        
        expected_prop_keys = [
            'prop_model_server_stats.create_model_time_ms',
            'prop_model_server_stats.element_count',
            'prop_model_server_stats.encode_time_ms',
            'prop_model_server_stats.io_time_ms',
            'prop_model_server_stats.parse_time_ms',
            'prop_model_server_stats.query_time_ms'
        ]
        
        for key in expected_prop_keys:
            assert key in mol
            assert isinstance(mol[key], (int, float, str))
    
    def test_read_sdf_custom_score_label(self, test_sdf_path):
        """Test reading with custom score label"""
        molecules = MoleculeReader.read_sdf(str(test_sdf_path), score_label='model_server_stats.query_time_ms')
        mol = molecules[0]
        
        assert 'score' in mol
        assert mol['score'] == 232
        assert 'prop_model_server_stats.query_time_ms' not in mol
    
    def test_read_sdf_nonexistent_score_label(self, test_sdf_path):
        """Test behavior with nonexistent score label"""
        molecules = MoleculeReader.read_sdf(str(test_sdf_path), score_label='nonexistent_score')
        mol = molecules[0]
        
        assert mol['score'] == 0.0
    
    def test_read_sdf_nonexistent_file(self):
        """Test behavior with nonexistent SDF file"""
        with pytest.raises(OSError):
            MoleculeReader.read_sdf("nonexistent_file.sdf")
    
    def test_read_pdb_basic(self, temp_pdb_file):
        """Test basic functionality of read_pdb"""
        content = MoleculeReader.read_pdb(temp_pdb_file)
        
        assert isinstance(content, str)
        assert "HEADER    TEST PDB FILE" in content
        assert "ATOM" in content
        assert "END" in content
    
    def test_read_pdb_nonexistent_file(self):
        """Test behavior with nonexistent PDB file"""
        with pytest.raises(FileNotFoundError):
            MoleculeReader.read_pdb("nonexistent_file.pdb")
    
    def test_property_type_conversion(self, test_sdf_path):
        """Test that numeric properties are correctly converted"""
        molecules = MoleculeReader.read_sdf(str(test_sdf_path))
        mol = molecules[0]
        
        assert isinstance(mol['prop_model_server_stats.io_time_ms'], (int, float))
        assert mol['prop_model_server_stats.io_time_ms'] == 4
        
        assert isinstance(mol['prop_model_server_stats.query_time_ms'], (int, float))
        assert mol['prop_model_server_stats.query_time_ms'] == 232
        
        assert isinstance(mol['prop_model_server_stats.element_count'], (int, float))
        assert mol['prop_model_server_stats.element_count'] == 31