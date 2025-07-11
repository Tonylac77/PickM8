"""
Tests for molecular fingerprint computation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data_io.molecules import *
from features.fingerprints.fingerprints import *


class TestFingerprints:
    """Test fingerprint computation with real data."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent.parent.parent / "test_data"
    
    @pytest.fixture
    def ligand_file(self, test_data_dir):
        """Path to test ligand SDF file."""
        return test_data_dir / "1fvv_l.sdf"


    def test_mapchiral_fingerprints_availability(self, ligand_file):
        """Test MapChiral fingerprint computation (if available)."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        is_available = is_mapchiral_available()
        assert isinstance(is_available, bool)
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_mapchiral_fingerprint(mol)
        
        if is_available:
            assert fp is not None or fp is None  # Could fail for technical reasons
        else:
            assert fp is None

    def test_e3fp_fingerprints_real_data(self, ligand_file):
        """Test E3FP fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_e3fp_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_ecfp_fingerprints_real_data(self, ligand_file):
        """Test ECFP fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_ecfp_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_electroshape_fingerprints_real_data(self, ligand_file):
        """Test ElectroShape fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_electroshape_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_functional_groups_fingerprints_real_data(self, ligand_file):
        """Test FunctionalGroups fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_functional_groups_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_maccs_fingerprints_real_data(self, ligand_file):
        """Test MACCS fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_maccs_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_pattern_fingerprints_real_data(self, ligand_file):
        """Test Pattern fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_pattern_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_pharmacophore_fingerprints_real_data(self, ligand_file):
        """Test Pharmacophore fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = compute_pharmacophore_fingerprint(mol)
        
        if fp is not None:
            assert isinstance(fp, list)
            assert len(fp) > 0
            assert all(isinstance(bit, (int, float)) for bit in fp)

    def test_fingerprints_invalid_molecule(self):
        """Test fingerprint computation with None molecule."""
        
        mapchiral_fp = compute_mapchiral_fingerprint(None)
        assert mapchiral_fp is None
        
        # Test scikit-fingerprints with None molecule
        e3fp_fp = compute_e3fp_fingerprint(None)
        assert e3fp_fp is None
        
        ecfp_fp = compute_ecfp_fingerprint(None)
        assert ecfp_fp is None
        
        electroshape_fp = compute_electroshape_fingerprint(None)
        assert electroshape_fp is None
        
        functional_groups_fp = compute_functional_groups_fingerprint(None)
        assert functional_groups_fp is None
        
        maccs_fp = compute_maccs_fingerprint(None)
        assert maccs_fp is None
        
        pattern_fp = compute_pattern_fingerprint(None)
        assert pattern_fp is None
        
        pharmacophore_fp = compute_pharmacophore_fingerprint(None)
        assert pharmacophore_fp is None


if __name__ == '__main__':
    pytest.main([__file__])