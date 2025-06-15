import pytest
import numpy as np
import tempfile
import os
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from rdkit import Chem
from concurrent.futures import ThreadPoolExecutor

from core.fingerprints import FingerprintHandler


class TestFingerprintHandler:
    """Test suite for FingerprintHandler class"""
    
    @pytest.fixture
    def handler(self):
        """Create basic FingerprintHandler instance"""
        return FingerprintHandler()
    
    @pytest.fixture
    def custom_handler(self):
        """Create FingerprintHandler with custom parameters"""
        return FingerprintHandler(
            fp_type='rdkit',
            fp_size=1024,
            radius=3,
            interaction_fp_type='PROLIF'
        )
    
    @pytest.fixture
    def sample_molecules(self):
        """Create sample RDKit molecules for testing"""
        smiles_list = [
            "CCO",          # Ethanol
            "CC(C)O",       # Isopropanol
            "c1ccccc1",     # Benzene
            "CCN(CC)CC",    # Triethylamine
            "invalid_smiles"  # This should cause an error
        ]
        molecules = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            molecules.append(mol)
        return molecules
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing"""
        config_data = {
            'fingerprinting': {
                'molecule_fp_type': 'morgan',
                'molecule_fp_size': 4096,
                'molecule_fp_radius': 3,
                'default_type': 'PLIP'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_init_default_parameters(self, handler):
        """Test FingerprintHandler initialization with default parameters"""
        assert handler.fp_type == 'morgan'
        assert handler.fp_size == 2048
        assert handler.radius == 2
        assert handler.interaction_fp_type == 'PLIP'
    
    def test_init_custom_parameters(self, custom_handler):
        """Test FingerprintHandler initialization with custom parameters"""
        assert custom_handler.fp_type == 'rdkit'
        assert custom_handler.fp_size == 1024
        assert custom_handler.radius == 3
        assert custom_handler.interaction_fp_type == 'PROLIF'
    
    def test_from_config_existing_file(self, temp_config_file):
        """Test creating FingerprintHandler from existing config file"""
        handler = FingerprintHandler.from_config(temp_config_file)
        
        assert handler.fp_type == 'morgan'
        assert handler.fp_size == 4096
        assert handler.radius == 3
        assert handler.interaction_fp_type == 'PLIP'
    
    def test_from_config_nonexistent_file(self):
        """Test creating FingerprintHandler from nonexistent config file"""
        handler = FingerprintHandler.from_config("nonexistent_config.yaml")
        
        # Should use default values
        assert handler.fp_type == 'morgan'
        assert handler.fp_size == 2048
        assert handler.radius == 2
        assert handler.interaction_fp_type == 'PLIP'
    
    def test_from_config_partial_config(self):
        """Test creating FingerprintHandler from partial config file"""
        partial_config = {
            'fingerprinting': {
                'molecule_fp_size': 512
                # Missing other parameters
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(partial_config, f)
            temp_path = f.name
        
        try:
            handler = FingerprintHandler.from_config(temp_path)
            
            # Should use default for missing values
            assert handler.fp_type == 'morgan'  # default
            assert handler.fp_size == 512       # from config
            assert handler.radius == 2          # default
            assert handler.interaction_fp_type == 'PLIP'  # default
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_compute_fingerprint_morgan_from_mol(self, handler):
        """Test computing Morgan fingerprint from RDKit molecule"""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        
        fp = handler.compute_fingerprint(mol)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 2048  # default fp_size
        assert np.sum(fp) > 0   # Should have some bits set
    
    def test_compute_fingerprint_morgan_from_smiles(self, handler):
        """Test computing Morgan fingerprint from SMILES string"""
        smiles = "CCO"
        
        fp = handler.compute_fingerprint(smiles)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 2048
        assert np.sum(fp) > 0
    
    def test_compute_fingerprint_rdkit_type(self, custom_handler):
        """Test computing RDKit fingerprint"""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        
        fp = custom_handler.compute_fingerprint(mol)
        
        assert isinstance(fp, np.ndarray)
        assert fp.dtype == np.int8
        assert len(fp) == 1024  # custom fp_size
        assert np.sum(fp) > 0
    
    def test_compute_fingerprint_invalid_type(self):
        """Test computing fingerprint with invalid type"""
        handler = FingerprintHandler(fp_type='invalid_type')
        mol = Chem.MolFromSmiles("CCO")
        
        with pytest.raises(ValueError, match="Unknown fingerprint type"):
            handler.compute_fingerprint(mol)
    
    def test_compute_fingerprint_invalid_smiles(self, handler):
        """Test computing fingerprint from invalid SMILES"""
        # RDKit returns None for invalid SMILES, which causes AttributeError in fingerprint generation
        with pytest.raises((AttributeError, TypeError)):
            handler.compute_fingerprint("invalid_smiles_string")
    
    def test_compute_fingerprint_none_input(self, handler):
        """Test computing fingerprint from None input"""
        with pytest.raises((AttributeError, TypeError)):
            handler.compute_fingerprint(None)
    
    def test_ifp_to_array_from_dict(self, handler):
        """Test converting interaction fingerprint dict to array"""
        ifp_dict = {
            '0': 1,
            '5': 2,
            '10': 1,
            '100': 3
        }
        
        fp_array = handler.ifp_to_array(ifp_dict, fp_length=200)
        
        assert isinstance(fp_array, np.ndarray)
        assert len(fp_array) == 200
        assert fp_array[0] == 1
        assert fp_array[5] == 2
        assert fp_array[10] == 1
        assert fp_array[100] == 3
        assert np.sum(fp_array) == 7
    
    def test_ifp_to_array_from_object_with_counts(self, handler):
        """Test converting interaction fingerprint object with counts to array"""
        # Mock object with counts attribute
        mock_ifp = Mock()
        mock_ifp.counts = {
            '0': 2,
            '15': 1,
            '50': 4
        }
        
        fp_array = handler.ifp_to_array(mock_ifp, fp_length=100)
        
        assert isinstance(fp_array, np.ndarray)
        assert len(fp_array) == 100
        assert fp_array[0] == 2
        assert fp_array[15] == 1
        assert fp_array[50] == 4
        assert np.sum(fp_array) == 7
    
    def test_ifp_to_array_out_of_bounds_keys(self, handler):
        """Test ifp_to_array with keys exceeding fp_length"""
        ifp_dict = {
            '0': 1,
            '50': 2,
            '150': 3  # This should be ignored (>= fp_length)
        }
        
        fp_array = handler.ifp_to_array(ifp_dict, fp_length=100)
        
        assert len(fp_array) == 100
        assert fp_array[0] == 1
        assert fp_array[50] == 2
        assert np.sum(fp_array) == 3  # Key '150' should be ignored
    
    def test_ifp_to_array_empty_input(self, handler):
        """Test ifp_to_array with empty input"""
        fp_array = handler.ifp_to_array({}, fp_length=50)
        
        assert isinstance(fp_array, np.ndarray)
        assert len(fp_array) == 50
        assert np.sum(fp_array) == 0
    
    def test_combine_fingerprints(self, handler):
        """Test combining molecular and interaction fingerprints"""
        mol_fp = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        ifp_array = np.array([0, 1, 0, 1, 0], dtype=np.float64)
        
        combined = handler.combine_fingerprints(mol_fp, ifp_array)
        
        assert isinstance(combined, np.ndarray)
        assert len(combined) == 10
        assert np.array_equal(combined[:5], mol_fp)
        assert np.array_equal(combined[5:], ifp_array)
    
    def test_combine_fingerprints_different_sizes(self, handler):
        """Test combining fingerprints of different sizes"""
        mol_fp = np.array([1, 0, 1], dtype=np.int8)
        ifp_array = np.array([0, 1, 0, 1, 0, 1, 1], dtype=np.float64)
        
        combined = handler.combine_fingerprints(mol_fp, ifp_array)
        
        assert len(combined) == 10
        assert np.array_equal(combined[:3], mol_fp)
        assert np.array_equal(combined[3:], ifp_array)
    
    def test_get_interaction_fingerprint_type(self, handler, custom_handler):
        """Test getting interaction fingerprint type"""
        assert handler.get_interaction_fingerprint_type() == 'PLIP'
        assert custom_handler.get_interaction_fingerprint_type() == 'PROLIF'
    
    def test_set_interaction_fingerprint_type_valid(self, handler):
        """Test setting valid interaction fingerprint type"""
        handler.set_interaction_fingerprint_type('PROLIF')
        assert handler.get_interaction_fingerprint_type() == 'PROLIF'
        
        handler.set_interaction_fingerprint_type('PLIP')
        assert handler.get_interaction_fingerprint_type() == 'PLIP'
    
    def test_set_interaction_fingerprint_type_invalid(self, handler):
        """Test setting invalid interaction fingerprint type"""
        with pytest.raises(ValueError, match="Invalid fingerprint type"):
            handler.set_interaction_fingerprint_type('INVALID_TYPE')
    
    def test_compute_fingerprints_batch_small_dataset(self, handler, sample_molecules):
        """Test batch fingerprint computation with small dataset"""
        # Remove None molecules for this test
        valid_molecules = [mol for mol in sample_molecules if mol is not None]
        
        fingerprints, errors = handler.compute_fingerprints_batch(valid_molecules)
        
        assert isinstance(fingerprints, list)
        assert isinstance(errors, dict)
        assert len(fingerprints) == len(valid_molecules)
        
        # Check that all fingerprints are valid arrays
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert len(fp) == handler.fp_size
            assert fp.dtype == np.int8
    
    def test_compute_fingerprints_batch_with_errors(self, handler, sample_molecules):
        """Test batch fingerprint computation with some invalid molecules"""
        fingerprints, errors = handler.compute_fingerprints_batch(sample_molecules)
        
        assert isinstance(fingerprints, list)
        assert isinstance(errors, dict)
        assert len(fingerprints) == len(sample_molecules)
        
        # Should have errors for None and invalid molecules
        assert len(errors) > 0
        
        # Error indices should correspond to None molecules
        none_indices = [i for i, mol in enumerate(sample_molecules) if mol is None]
        for idx in none_indices:
            assert idx in errors
    
    def test_compute_fingerprints_batch_empty_list(self, handler):
        """Test batch fingerprint computation with empty molecule list"""
        fingerprints, errors = handler.compute_fingerprints_batch([])
        
        assert fingerprints == []
        assert errors == {}
    
    @patch('multiprocessing.cpu_count')
    def test_compute_fingerprints_batch_worker_calculation(self, mock_cpu_count, handler):
        """Test automatic worker count calculation"""
        mock_cpu_count.return_value = 8
        
        # Test with different molecule counts
        molecules_10 = [Chem.MolFromSmiles("CCO")] * 10
        molecules_100 = [Chem.MolFromSmiles("CCO")] * 100
        
        # Test that the function runs without errors
        fingerprints_10, errors_10 = handler.compute_fingerprints_batch(molecules_10)
        fingerprints_100, errors_100 = handler.compute_fingerprints_batch(molecules_100)
        
        # Verify results structure
        assert len(fingerprints_10) == 10
        assert len(fingerprints_100) == 100
        assert len(errors_10) == 0
        assert len(errors_100) == 0
    
    def test_compute_fingerprints_batch_custom_workers(self, handler):
        """Test batch computation with custom worker count"""
        molecules = [Chem.MolFromSmiles("CCO")] * 5
        
        fingerprints, errors = handler.compute_fingerprints_batch(molecules, max_workers=2)
        
        assert len(fingerprints) == 5
        assert len(errors) == 0
        
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert len(fp) == handler.fp_size
    
    def test_process_molecules_batch_basic(self, handler):
        """Test basic structure of process_molecules_batch method"""
        molecules = [Chem.MolFromSmiles("CCO")] * 3
        protein_path = "/fake/protein.pdb"
        interaction_context = {"type": "PLIP"}
        
        # Mock the calculate_batch_with_context function
        with patch('core.interaction_functions.calculate_batch_with_context') as mock_calculate_batch:
            # Mock the interaction calculation results
            mock_ifp_results = [{'interaction_count': i} for i in range(3)]
            mock_interaction_results = [[{'type': 'test'}] for _ in range(3)]
            mock_ifp_errors = {}
            
            mock_calculate_batch.return_value = (mock_ifp_results, mock_interaction_results, mock_ifp_errors)
            
            fingerprints, ifp_results, interaction_results, all_errors = handler.process_molecules_batch(
                molecules, protein_path, interaction_context
            )
            
            # Verify results structure
            assert isinstance(fingerprints, list)
            assert len(fingerprints) == 3
            assert isinstance(all_errors, dict)
    
    def test_process_molecules_batch_worker_optimization(self, handler):
        """Test worker count optimization in batch processing"""
        molecules = [Chem.MolFromSmiles("CCO")] * 10
        protein_path = "/fake/protein.pdb"
        interaction_context = {"type": "PLIP"}
        
        # Mock the interaction calculation
        with patch('core.interaction_functions.calculate_batch_with_context') as mock_calculate_batch:
            mock_calculate_batch.return_value = ([], [], {})
            
            # Test that the method can be called with custom worker counts
            fingerprints, ifp_results, interaction_results, all_errors = handler.process_molecules_batch(
                molecules, protein_path, interaction_context,
                max_fp_workers=4, max_ifp_workers=2
            )
            
            # Verify basic structure
            assert isinstance(fingerprints, list)
            assert len(fingerprints) == 10
    
    def test_fingerprint_consistency(self, handler):
        """Test that fingerprints are consistent across multiple calls"""
        mol = Chem.MolFromSmiles("CCO")
        
        fp1 = handler.compute_fingerprint(mol)
        fp2 = handler.compute_fingerprint(mol)
        
        assert np.array_equal(fp1, fp2)
    
    def test_different_molecules_different_fingerprints(self, handler):
        """Test that different molecules produce different fingerprints"""
        mol1 = Chem.MolFromSmiles("CCO")       # Ethanol
        mol2 = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        
        fp1 = handler.compute_fingerprint(mol1)
        fp2 = handler.compute_fingerprint(mol2)
        
        # Should be different fingerprints
        assert not np.array_equal(fp1, fp2)
    
    def test_radius_affects_fingerprint(self):
        """Test that different radius values affect fingerprints"""
        mol = Chem.MolFromSmiles("CCc1ccccc1")  # Ethylbenzene
        
        handler1 = FingerprintHandler(radius=1)
        handler2 = FingerprintHandler(radius=3)
        
        fp1 = handler1.compute_fingerprint(mol)
        fp2 = handler2.compute_fingerprint(mol)
        
        # Different radius should generally produce different fingerprints
        assert not np.array_equal(fp1, fp2)
    
    def test_fp_size_affects_fingerprint_length(self):
        """Test that fp_size parameter affects fingerprint length"""
        mol = Chem.MolFromSmiles("CCO")
        
        handler1 = FingerprintHandler(fp_size=512)
        handler2 = FingerprintHandler(fp_size=1024)
        
        fp1 = handler1.compute_fingerprint(mol)
        fp2 = handler2.compute_fingerprint(mol)
        
        assert len(fp1) == 512
        assert len(fp2) == 1024


class TestFingerprintHandlerIntegration:
    """Integration tests for FingerprintHandler with real data"""
    
    @pytest.fixture
    def sdf_file_path(self):
        """Path to test SDF file with molecules"""
        return Path(__file__).parent.parent / "test_data" / "example_poses_1fvv.sdf"
    
    @pytest.fixture
    def protein_file_path(self):
        """Path to test protein PDB file"""
        return Path(__file__).parent.parent / "test_data" / "1fvv_p.pdb"
    
    def test_compute_fingerprints_real_molecules(self, sdf_file_path):
        """Test computing fingerprints for real molecules from SDF file"""
        if not sdf_file_path.exists():
            pytest.skip("Test SDF file not available")
        
        # Read molecules from SDF
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        molecules = []
        count = 0
        for mol in supplier:
            if mol is not None and count < 5:  # Test first 5 molecules
                molecules.append(mol)
                count += 1
        
        if not molecules:
            pytest.skip("No valid molecules found in test SDF")
        
        handler = FingerprintHandler()
        fingerprints, errors = handler.compute_fingerprints_batch(molecules)
        
        assert len(fingerprints) == len(molecules)
        assert len(errors) == 0  # Should be no errors with valid molecules
        
        # Verify all fingerprints are valid
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert len(fp) == 2048
            assert fp.dtype == np.int8
            assert np.sum(fp) > 0  # Should have some bits set
    
    def test_different_fingerprint_types_real_molecules(self, sdf_file_path):
        """Test different fingerprint types with real molecules"""
        if not sdf_file_path.exists():
            pytest.skip("Test SDF file not available")
        
        # Read one molecule
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        mol = next(supplier)
        
        if mol is None:
            pytest.skip("No valid molecules found in test SDF")
        
        # Test Morgan fingerprint
        morgan_handler = FingerprintHandler(fp_type='morgan', fp_size=1024)
        morgan_fp = morgan_handler.compute_fingerprint(mol)
        
        # Test RDKit fingerprint
        rdkit_handler = FingerprintHandler(fp_type='rdkit', fp_size=1024)
        rdkit_fp = rdkit_handler.compute_fingerprint(mol)
        
        # Should be different fingerprint types
        assert len(morgan_fp) == len(rdkit_fp) == 1024
        assert not np.array_equal(morgan_fp, rdkit_fp)
        
        # Both should have some bits set
        assert np.sum(morgan_fp) > 0
        assert np.sum(rdkit_fp) > 0
    
    def test_full_workflow_integration(self, sdf_file_path, protein_file_path):
        """Test full workflow integration with real data"""
        if not (sdf_file_path.exists() and protein_file_path.exists()):
            pytest.skip("Test data files not available")
        
        # Read test molecules
        supplier = Chem.ForwardSDMolSupplier(str(sdf_file_path))
        molecules = []
        count = 0
        for mol in supplier:
            if mol is not None and count < 3:
                molecules.append(mol)
                count += 1
        
        if not molecules:
            pytest.skip("No valid molecules found")
        
        # Test fingerprint computation only (core functionality)
        handler = FingerprintHandler.from_config("nonexistent_config.yaml")  # Use defaults
        
        fingerprints, errors = handler.compute_fingerprints_batch(molecules)
        
        # Verify results
        assert len(fingerprints) == len(molecules)
        assert len(errors) == 0  # Should be no errors with valid molecules
        
        # Verify fingerprints are valid
        for fp in fingerprints:
            assert isinstance(fp, np.ndarray)
            assert len(fp) == 2048  # default size
            assert fp.dtype == np.int8
            assert np.sum(fp) > 0  # Should have some bits set