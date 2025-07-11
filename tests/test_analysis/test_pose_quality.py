"""
Tests for pose quality analysis.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from data_io.molecules import *
from analysis.pose_quality import *


class TestPoseQuality:
    """Test pose quality analysis."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent.parent.parent / "test_data"
    
    @pytest.fixture
    def protein_file(self, test_data_dir):
        """Path to test protein PDB file."""
        return test_data_dir / "1fvv_p.pdb"
    
    @pytest.fixture
    def ligand_file(self, test_data_dir):
        """Path to test ligand SDF file."""
        return test_data_dir / "1fvv_l.sdf"
    
    @pytest.fixture
    def protein_content(self, protein_file):
        """Load protein content from PDB file."""
        if not protein_file.exists():
            pytest.skip("Test protein file not available")
        
        with open(protein_file, 'r') as f:
            return f.read()

    def test_pose_quality_real_data(self, ligand_file, protein_content):
        """Test pose quality analysis with real molecular data."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        df = load_sdf(str(ligand_file))
        mol_block = df['mol_block'].iloc[0]
        
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        result = analyze_single_pose(mol_block, protein_content, config)
        
        # Essential checks
        assert isinstance(result, dict)
        assert 'clashes' in result
        assert 'strain_energy' in result
        assert isinstance(result['clashes'], (int, float))
        assert isinstance(result['strain_energy'], (int, float))
        assert result['clashes'] >= 0
        assert result['strain_energy'] >= 0.0

    def test_pose_quality_invalid_input(self):
        """Test pose quality analysis with invalid input."""
        config = {'calculate_clashes': True, 'calculate_strain': True}
        
        # Invalid mol_block
        result = analyze_single_pose("INVALID", "", config)
        
        assert isinstance(result, dict)
        assert 'clashes' in result
        assert 'strain_energy' in result
        # Should return default values rather than crash
        assert result['clashes'] >= 0
        assert result['strain_energy'] >= 0.0


if __name__ == '__main__':
    pytest.main([__file__])