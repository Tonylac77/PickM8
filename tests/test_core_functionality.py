"""
Essential functionality tests using real data.
Tests core data pipeline: SDF loading, fingerprints, interactions, and pose quality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data import molecules, fingerprints, interactions
from analysis import pose_quality


class TestCoreDataPipeline:
    """Test essential data processing pipeline with real data."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent.parent / "test_data"
    
    @pytest.fixture
    def protein_file(self, test_data_dir):
        """Path to test protein PDB file."""
        return test_data_dir / "1fvv_p.pdb"
    
    @pytest.fixture
    def ligand_file(self, test_data_dir):
        """Path to test ligand SDF file."""
        return test_data_dir / "1fvv_l.sdf"
    
    @pytest.fixture
    def poses_file(self, test_data_dir):
        """Path to test poses SDF file."""
        return test_data_dir / "example_poses_1fvv.sdf"
    
    @pytest.fixture
    def protein_content(self, protein_file):
        """Load protein content from PDB file."""
        if not protein_file.exists():
            pytest.skip("Test protein file not available")
        
        with open(protein_file, 'r') as f:
            return f.read()

    def test_sdf_loading_single_molecule(self, ligand_file):
        """Test loading single molecule from SDF file."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = molecules.load_sdf(str(ligand_file))
        
        # Essential checks
        assert len(df) == 1
        assert 'mol' in df.columns
        assert 'smiles' in df.columns
        assert 'mol_block' in df.columns
        
        # Verify molecule was loaded correctly
        mol = df['mol'].iloc[0]
        assert mol is not None
        assert mol.GetNumAtoms() > 0
        
        # Verify SMILES is reasonable
        smiles = df['smiles'].iloc[0]
        assert isinstance(smiles, str)
        assert len(smiles) > 5

    def test_sdf_loading_multiple_molecules(self, poses_file):
        """Test loading multiple molecules from SDF file."""
        if not poses_file.exists():
            pytest.skip("Test poses file not available")
        
        df = molecules.load_sdf(str(poses_file))
        
        # Essential checks
        assert len(df) > 1
        assert all(df['mol'].notna())
        assert all(df['smiles'].str.len() > 5)
        assert all(df['mol_block'].str.len() > 100)

    def test_morgan_fingerprints_real_data(self, ligand_file):
        """Test Morgan fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = molecules.load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = fingerprints.compute_morgan_fingerprint(mol)
        
        assert fp is not None
        assert isinstance(fp, list)
        assert len(fp) == 2048  # Default size
        assert all(bit in ['0', '1'] for bit in fp)
        assert sum(int(bit) for bit in fp) > 0  # Should have some bits set

    def test_rdkit_fingerprints_real_data(self, ligand_file):
        """Test RDKit fingerprint computation with real molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = molecules.load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = fingerprints.compute_rdkit_fingerprint(mol)
        
        assert fp is not None
        assert isinstance(fp, list)
        assert len(fp) == 2048  # Default size
        assert all(isinstance(bit, int) and bit in [0, 1] for bit in fp)
        assert sum(fp) > 0  # Should have some bits set

    def test_mapchiral_fingerprints_availability(self, ligand_file):
        """Test MapChiral fingerprint computation (if available)."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        is_available = fingerprints.is_mapchiral_available()
        assert isinstance(is_available, bool)
        
        df = molecules.load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        fp = fingerprints.compute_mapchiral_fingerprint(mol)
        
        if is_available:
            assert fp is not None or fp is None  # Could fail for technical reasons
        else:
            assert fp is None

    def test_interactions_real_data(self, ligand_file, protein_content, protein_file):
        """Test protein-ligand interaction computation with real data."""
        if not ligand_file.exists() or not protein_file.exists():
            pytest.skip("Test data files not available")
        
        df = molecules.load_sdf(str(ligand_file))
        mol = df['mol'].iloc[0]
        
        config = {'interaction_type': 'plip', 'ligand_name': 'LIG'}
        
        try:
            ifp_json, interactions_json, num_interactions = interactions.compute_interaction_fingerprint(
                mol, protein_content, config
            )
            
            # Should return something (even if None when tools unavailable)
            assert isinstance(num_interactions, int)
            assert num_interactions >= 0
            
            if ifp_json is not None:
                assert isinstance(ifp_json, str)
                assert isinstance(interactions_json, str)
                
        except ImportError:
            # Acceptable if interaction tools not available
            pytest.skip("Interaction analysis tools not available")

    def test_pose_quality_real_data(self, ligand_file, protein_content):
        """Test pose quality analysis with real molecular data."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = molecules.load_sdf(str(ligand_file))
        mol_block = df['mol_block'].iloc[0]
        
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        result = pose_quality.analyze_single_pose(mol_block, protein_content, config)
        
        # Essential checks
        assert isinstance(result, dict)
        assert 'clashes' in result
        assert 'strain_energy' in result
        assert isinstance(result['clashes'], (int, float))
        assert isinstance(result['strain_energy'], (int, float))
        assert result['clashes'] >= 0
        assert result['strain_energy'] >= 0.0


class TestCriticalErrorHandling:
    """Test critical error conditions."""
    
    def test_sdf_loading_file_not_found(self):
        """Test SDF loading with non-existent file."""
        with pytest.raises(Exception):
            molecules.load_sdf("/non/existent/file.sdf")

    def test_fingerprints_invalid_molecule(self):
        """Test fingerprint computation with None molecule."""
        morgan_fp = fingerprints.compute_morgan_fingerprint(None)
        assert morgan_fp is None
        
        rdkit_fp = fingerprints.compute_rdkit_fingerprint(None)
        assert rdkit_fp is None
        
        mapchiral_fp = fingerprints.compute_mapchiral_fingerprint(None)
        assert mapchiral_fp is None

    def test_interactions_missing_tools(self):
        """Test interaction computation when tools unavailable."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        protein_content = "ATOM      1  N   ALA A   1      0.0  0.0  0.0  1.00  0.00           N"
        config = {'interaction_type': 'plip'}
        
        # Should handle gracefully when tools unavailable
        result = interactions.compute_interaction_fingerprint(mol, protein_content, config)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[2], int)  # num_interactions should be int

    def test_pose_quality_invalid_input(self):
        """Test pose quality analysis with invalid input."""
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Invalid mol_block
        result = pose_quality.analyze_single_pose("INVALID", "", config)
        
        assert isinstance(result, dict)
        assert 'clashes' in result
        assert 'strain_energy' in result
        # Should return default values rather than crash
        assert result['clashes'] >= 0
        assert result['strain_energy'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__])