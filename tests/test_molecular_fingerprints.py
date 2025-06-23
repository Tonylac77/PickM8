"""
Test suite for core/fingerprints/molecular.py functions.
Focuses on molecular fingerprint computation functions including MapChiral.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from rdkit import Chem

from core.fingerprints.molecular import (
    compute_morgan_fingerprint,
    compute_rdkit_fingerprint,
    compute_mapchiral_fingerprint,
    is_mapchiral_available,
    MAPCHIRAL_AVAILABLE
)


class TestMolecularFingerprints:
    """Test suite for molecular fingerprint functions"""
    
    def create_test_molecule(self):
        """Create a simple test molecule (ethanol)"""
        return Chem.MolFromSmiles("CCO")
    
    def create_invalid_molecule(self):
        """Create an invalid molecule"""
        return None

    def test_compute_morgan_fingerprint_valid_molecule(self):
        """Test Morgan fingerprint computation with valid molecule"""
        mol = self.create_test_molecule()
        
        result = compute_morgan_fingerprint(mol, radius=2, n_bits=1024)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(bit in ['0', '1'] for bit in result)

    def test_compute_morgan_fingerprint_none_molecule(self):
        """Test Morgan fingerprint with None molecule"""
        result = compute_morgan_fingerprint(None)
        
        assert result is None

    def test_compute_morgan_fingerprint_custom_parameters(self):
        """Test Morgan fingerprint with custom parameters"""
        mol = self.create_test_molecule()
        
        result = compute_morgan_fingerprint(mol, radius=3, n_bits=512)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 512

    @patch('core.fingerprints.molecular.rdMolDescriptors.GetMorganFingerprintAsBitVect')
    def test_compute_morgan_fingerprint_error_handling(self, mock_get_fp):
        """Test error handling in Morgan fingerprint computation"""
        mol = self.create_test_molecule()
        
        # Mock an exception in RDKit
        mock_get_fp.side_effect = Exception("RDKit error")
        
        result = compute_morgan_fingerprint(mol)
        
        assert result is None

    def test_compute_rdkit_fingerprint_valid_molecule(self):
        """Test RDKit fingerprint computation with valid molecule"""
        mol = self.create_test_molecule()
        
        result = compute_rdkit_fingerprint(mol, n_bits=1024)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(bit, int) and bit in [0, 1] for bit in result)

    def test_compute_rdkit_fingerprint_none_molecule(self):
        """Test RDKit fingerprint with None molecule"""
        result = compute_rdkit_fingerprint(None)
        
        assert result is None

    def test_compute_rdkit_fingerprint_custom_parameters(self):
        """Test RDKit fingerprint with custom parameters"""
        mol = self.create_test_molecule()
        
        result = compute_rdkit_fingerprint(mol, n_bits=512)
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 512

    @patch('core.fingerprints.molecular.Chem.RDKFingerprint')
    def test_compute_rdkit_fingerprint_error_handling(self, mock_rdkit_fp):
        """Test error handling in RDKit fingerprint computation"""
        mol = self.create_test_molecule()
        
        # Mock an exception in RDKit
        mock_rdkit_fp.side_effect = Exception("RDKit fingerprint error")
        
        result = compute_rdkit_fingerprint(mol)
        
        assert result is None

    def test_fingerprint_consistency(self):
        """Test that fingerprints are consistent for same molecule"""
        mol1 = self.create_test_molecule()
        mol2 = self.create_test_molecule()
        
        # Morgan fingerprints should be identical for same molecule
        fp1_morgan = compute_morgan_fingerprint(mol1, radius=2, n_bits=1024)
        fp2_morgan = compute_morgan_fingerprint(mol2, radius=2, n_bits=1024)
        
        assert fp1_morgan == fp2_morgan
        
        # RDKit fingerprints should be identical for same molecule
        fp1_rdkit = compute_rdkit_fingerprint(mol1, n_bits=1024)
        fp2_rdkit = compute_rdkit_fingerprint(mol2, n_bits=1024)
        
        assert fp1_rdkit == fp2_rdkit

    def test_different_molecules_different_fingerprints(self):
        """Test that different molecules produce different fingerprints"""
        mol1 = Chem.MolFromSmiles("CCO")     # Ethanol
        mol2 = Chem.MolFromSmiles("CC(C)O")  # Isopropanol
        
        # Morgan fingerprints should be different
        fp1_morgan = compute_morgan_fingerprint(mol1)
        fp2_morgan = compute_morgan_fingerprint(mol2)
        
        assert fp1_morgan != fp2_morgan
        
        # RDKit fingerprints should be different
        fp1_rdkit = compute_rdkit_fingerprint(mol1)
        fp2_rdkit = compute_rdkit_fingerprint(mol2)
        
        assert fp1_rdkit != fp2_rdkit

    def test_fingerprint_default_parameters(self):
        """Test fingerprint computation with default parameters"""
        mol = self.create_test_molecule()
        
        # Test Morgan with defaults
        morgan_fp = compute_morgan_fingerprint(mol)
        assert morgan_fp is not None
        assert len(morgan_fp) == 2048  # Default n_bits
        
        # Test RDKit with defaults
        rdkit_fp = compute_rdkit_fingerprint(mol)
        assert rdkit_fp is not None
        assert len(rdkit_fp) == 2048  # Default n_bits

    def test_very_small_fingerprint_size(self):
        """Test fingerprint computation with very small size"""
        mol = self.create_test_molecule()
        
        # Test with very small fingerprint
        morgan_fp = compute_morgan_fingerprint(mol, n_bits=8)
        rdkit_fp = compute_rdkit_fingerprint(mol, n_bits=8)
        
        assert len(morgan_fp) == 8
        assert len(rdkit_fp) == 8

    def test_large_fingerprint_size(self):
        """Test fingerprint computation with large size"""
        mol = self.create_test_molecule()
        
        # Test with large fingerprint
        morgan_fp = compute_morgan_fingerprint(mol, n_bits=8192)
        rdkit_fp = compute_rdkit_fingerprint(mol, n_bits=8192)
        
        assert len(morgan_fp) == 8192
        assert len(rdkit_fp) == 8192

    def test_different_morgan_radius(self):
        """Test Morgan fingerprints with different radius values"""
        mol = self.create_test_molecule()
        
        fp_r1 = compute_morgan_fingerprint(mol, radius=1)
        fp_r2 = compute_morgan_fingerprint(mol, radius=2)
        fp_r3 = compute_morgan_fingerprint(mol, radius=3)
        
        # Different radius should generally produce different fingerprints
        # (though for very simple molecules they might be the same)
        assert isinstance(fp_r1, list)
        assert isinstance(fp_r2, list)
        assert isinstance(fp_r3, list)
        assert len(fp_r1) == len(fp_r2) == len(fp_r3) == 2048


class TestMapChiralFingerprints:
    """Test suite for MapChiral fingerprint functions"""
    
    def create_test_molecule(self):
        """Create a simple test molecule (ethanol)"""
        return Chem.MolFromSmiles("CCO")
    
    def create_chiral_molecules(self):
        """Create chiral molecules for MapChiral testing"""
        # Enantiomers from the example code
        mol1 = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@@H]1N2C(=O)C3=CC=CC=C3C2=O')
        mol2 = Chem.MolFromSmiles('C1CC(=O)NC(=O)[C@H]1N2C(=O)C3=CC=CC=C3C2=O')
        return mol1, mol2
    
    def test_is_mapchiral_available(self):
        """Test MapChiral availability check"""
        result = is_mapchiral_available()
        assert isinstance(result, bool)
        assert result == MAPCHIRAL_AVAILABLE
    
    def test_compute_mapchiral_fingerprint_none_molecule(self):
        """Test MapChiral fingerprint computation with None molecule"""
        fp = compute_mapchiral_fingerprint(None)
        assert fp is None
    
    @pytest.mark.parametrize("mapchiral_available", [True, False])
    def test_compute_mapchiral_fingerprint_availability(self, mapchiral_available):
        """Test MapChiral fingerprint with different availability states"""
        mol = self.create_test_molecule()
        
        with patch('core.fingerprints.molecular.MAPCHIRAL_AVAILABLE', mapchiral_available):
            if mapchiral_available:
                # Mock the mapchiral_encode function
                with patch('core.fingerprints.molecular.mapchiral_encode') as mock_encode:
                    # Create a mock fingerprint array
                    mock_fp = np.array([1, 0, 1, 1, 0] * 409 + [1, 0, 1])  # 2048 bits
                    mock_encode.return_value = mock_fp
                    
                    fp = compute_mapchiral_fingerprint(mol)
                    
                    assert fp is not None
                    assert isinstance(fp, list)
                    assert len(fp) == 2048
                    
                    # Verify the encode function was called with correct parameters
                    mock_encode.assert_called_once_with(
                        mol, 
                        max_radius=2, 
                        n_permutations=2048, 
                        mapping=False
                    )
            else:
                # Should return None when MapChiral not available
                fp = compute_mapchiral_fingerprint(mol)
                assert fp is None
    
    def test_compute_mapchiral_fingerprint_custom_params(self):
        """Test MapChiral fingerprint with custom parameters"""
        mol = self.create_test_molecule()
        
        with patch('core.fingerprints.molecular.MAPCHIRAL_AVAILABLE', True), \
             patch('core.fingerprints.molecular.mapchiral_encode') as mock_encode:
            
            # Create a mock fingerprint array
            mock_fp = np.array([0, 1] * 512)  # 1024 bits
            mock_encode.return_value = mock_fp
            
            fp = compute_mapchiral_fingerprint(
                mol,
                max_radius=3,
                n_permutations=1024,
                mapping=True
            )
            
            assert fp is not None
            assert len(fp) == 1024
            
            # Verify custom parameters were passed
            mock_encode.assert_called_once_with(
                mol,
                max_radius=3,
                n_permutations=1024,
                mapping=True
            )
    
    def test_compute_mapchiral_fingerprint_with_chiral_molecules(self):
        """Test MapChiral fingerprint with actual chiral molecules"""  
        mol1, mol2 = self.create_chiral_molecules()
        
        with patch('core.fingerprints.molecular.MAPCHIRAL_AVAILABLE', True), \
             patch('core.fingerprints.molecular.mapchiral_encode') as mock_encode:
            
            # Mock different fingerprints for the two enantiomers
            mock_fp1 = np.array([1, 0, 1, 0] * 512)  # 2048 bits
            mock_fp2 = np.array([0, 1, 0, 1] * 512)  # 2048 bits, different pattern
            
            # Configure mock to return different fingerprints for different molecules
            def side_effect(mol, **kwargs):
                if mol == mol1:
                    return mock_fp1
                elif mol == mol2:
                    return mock_fp2
                else:
                    return np.zeros(2048)
            
            mock_encode.side_effect = side_effect
            
            fp1 = compute_mapchiral_fingerprint(mol1)
            fp2 = compute_mapchiral_fingerprint(mol2)
            
            assert fp1 is not None
            assert fp2 is not None
            assert fp1 != fp2  # Enantiomers should have different fingerprints
            assert len(fp1) == len(fp2) == 2048
    
    def test_compute_mapchiral_fingerprint_error_handling(self):
        """Test MapChiral fingerprint error handling"""
        mol = self.create_test_molecule()
        
        with patch('core.fingerprints.molecular.MAPCHIRAL_AVAILABLE', True), \
             patch('core.fingerprints.molecular.mapchiral_encode') as mock_encode:
            
            # Make the encode function raise an exception
            mock_encode.side_effect = Exception("Test error")
            
            fp = compute_mapchiral_fingerprint(mol)
            
            # Should return None on error
            assert fp is None
    
    def test_mapchiral_fingerprint_consistency(self):
        """Test that MapChiral fingerprints are consistent across multiple calls"""
        mol = self.create_test_molecule()
        
        # Test MapChiral fingerprint consistency (with mocking)
        with patch('core.fingerprints.molecular.MAPCHIRAL_AVAILABLE', True), \
             patch('core.fingerprints.molecular.mapchiral_encode') as mock_encode:
            
            mock_fp = np.array([1, 0] * 1024)
            mock_encode.return_value = mock_fp
            
            fp1 = compute_mapchiral_fingerprint(mol)
            fp2 = compute_mapchiral_fingerprint(mol)
            assert fp1 == fp2
    
    def test_mapchiral_fingerprint_default_parameters(self):
        """Test MapChiral fingerprint computation with default parameters"""
        mol = self.create_test_molecule()
        
        with patch('core.fingerprints.molecular.MAPCHIRAL_AVAILABLE', True), \
             patch('core.fingerprints.molecular.mapchiral_encode') as mock_encode:
            
            mock_fp = np.array([1, 0, 1, 0] * 512)  # 2048 bits
            mock_encode.return_value = mock_fp
            
            fp = compute_mapchiral_fingerprint(mol)
            
            assert fp is not None
            assert len(fp) == 2048  # Default n_permutations
            
            # Verify default parameters were used
            mock_encode.assert_called_once_with(
                mol,
                max_radius=2,
                n_permutations=2048,
                mapping=False
            )


if __name__ == '__main__':
    pytest.main([__file__])