"""
Tests for session persistence and management.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import shutil
from pathlib import Path

import sys
import os
sys.path.insert(0, '/home/tony/PickM8')
# Add the project root to sys.path to avoid conflicts
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import directly from the sessions module
from sessions.sessions import save_session, load_session


@pytest.fixture
def test_session_cleanup():
    """Fixture to track and cleanup test sessions."""
    created_sessions = []
    
    def track_session(session_id):
        created_sessions.append(session_id)
        return session_id
    
    # Yield the tracking function
    yield track_session
    
    # Cleanup after test
    sessions_dir = Path("data/sessions")
    for session_id in created_sessions:
        session_path = sessions_dir / session_id
        if session_path.exists():
            shutil.rmtree(session_path)


class TestSessionPersistence:
    """Test session save/load functionality."""
    
    def create_sample_session_data(self):
        """Create sample molecular data for session testing."""
        np.random.seed(42)
        
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'smiles': ['CCO', 'CC', 'CCC', 'CCCC', 'CCCCC'],
            'score': [-2.5, -3.1, -1.8, -4.2, -2.9],
            'ecfp_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(5)],
            'functional_groups_fp': [np.random.randint(0, 2, 85).tolist() for _ in range(5)],
            'maccs_fp': [np.random.randint(0, 2, 167).tolist() for _ in range(5)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(5)],
            'interactions': [json.dumps([{'type': 'hydrophobic', 'residue': f'VAL{i}'}]) for i in range(5)],
            'num_interactions': [1, 2, 0, 3, 1],
            'grade': [None, 'A', None, 'B', None],
            'grade_timestamp': [None, '2024-01-01 12:00:00', None, '2024-01-01 13:00:00', None],
            'clashes': [0, 1, 2, 0, 1],
            'strain_energy': [0.0, 1.5, 2.3, 0.5, 1.1],
            'prediction': [None, None, 'B', None, 'A'],
            'prediction_timestamp': [None, None, '2024-01-01 14:00:00', None, '2024-01-01 14:30:00']
        }
        
        return pd.DataFrame(data)

    def test_session_save_and_load(self, test_session_cleanup):
        """Test saving and loading session data."""
        df = self.create_sample_session_data()
        session_id = test_session_cleanup("test_session_123")
        metadata = {"test": "metadata"}
        
        # Test saving
        success = save_session(session_id, df, metadata)
        assert success is True
        
        # Test loading
        result = load_session(session_id)
        assert result is not None
        
        loaded_df, loaded_metadata = result
        
        # Verify data integrity
        assert loaded_df is not None
        assert len(loaded_df) == len(df)
        assert list(loaded_df.columns) == list(df.columns)
        assert loaded_metadata == metadata
        
        # Verify specific data preservation
        pd.testing.assert_frame_equal(loaded_df, df)

    def test_session_load_nonexistent(self):
        """Test loading from non-existent session."""
        result = load_session("nonexistent_session_id")
        assert result is None

    def test_session_metadata_preservation(self, test_session_cleanup):
        """Test that session metadata is properly preserved."""
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['mol1', 'mol2'],
            'score': [1.0, 2.0]
        })
        
        metadata = {
            'session_name': 'test_session',
            'created_at': '2024-01-01',
            'protein_file': 'test.pdb',
            'num_molecules': 2
        }
        
        session_id = test_session_cleanup("metadata_test_session")
        
        # Save with metadata
        success = save_session(session_id, df, metadata)
        assert success is True
        
        # Load and verify metadata preservation
        result = load_session(session_id)
        assert result is not None
        
        loaded_df, loaded_metadata = result
        
        assert loaded_metadata == metadata
        assert loaded_metadata['session_name'] == 'test_session'
        assert loaded_metadata['num_molecules'] == 2


if __name__ == '__main__':
    pytest.main([__file__])