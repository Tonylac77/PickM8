"""
Session management and grading workflow tests.
Tests data persistence, grading operations, and session management.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import shutil
from pathlib import Path

from data import molecules, sessions
from analysis import grading


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
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(5)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(5)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(5)],
            'interactions': [json.dumps([{'type': 'hydrophobic', 'residue': f'VAL{i}'}]) for i in range(5)],
            'num_interactions': [1, 2, 0, 3, 1],
            'grade': [None, 'A', None, 'B', None],
            'grade_timestamp': [None, '2024-01-01 12:00:00', None, '2024-01-01 13:00:00', None],
            'clashes': [0, 1, 2, 0, 1],
            'strain_energy': [0.0, 1.5, 2.3, 0.5, 1.1],
            'prediction': [None, None, 'B', None, 'A'],
            'prediction_uncertainty': [None, None, 0.3, None, 0.7],
            'prediction_timestamp': [None, None, '2024-01-01 14:00:00', None, '2024-01-01 14:30:00']
        }
        
        return pd.DataFrame(data)

    def test_session_save_and_load(self, test_session_cleanup):
        """Test saving and loading session data."""
        df = self.create_sample_session_data()
        session_id = test_session_cleanup("test_session_123")
        metadata = {"test": "metadata"}
        
        # Test saving
        success = sessions.save_session(session_id, df, metadata)
        assert success is True
        
        # Test loading
        result = sessions.load_session(session_id)
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
        result = sessions.load_session("nonexistent_session_id")
        assert result is None

    def test_molecules_dataframe_save_load(self):
        """Test molecules DataFrame save/load functionality."""
        df = self.create_sample_session_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test saving
            success = molecules.save_molecules_dataframe(df, tmpdir)
            assert success is True
            
            # Test loading
            loaded_df = molecules.load_molecules_dataframe(tmpdir)
            
            assert loaded_df is not None
            assert len(loaded_df) == len(df)
            
            # Verify key columns preserved
            for col in ['id', 'name', 'smiles', 'score', 'grade']:
                assert col in loaded_df.columns
                pd.testing.assert_series_equal(loaded_df[col], df[col])


class TestGradingWorkflow:
    """Test grading operations and workflow."""
    
    def create_grading_test_data(self):
        """Create test data for grading operations."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'score': [-2.5, -3.1, -1.8, -4.2, -2.9],
            'grade': [None, 'A', None, 'B', None],
            'prediction': [None, None, 'B', None, 'A'],
            'prediction_uncertainty': [None, None, 0.3, None, 0.7]
        })

    def test_add_grade_workflow(self):
        """Test adding grades to molecules."""
        df = self.create_grading_test_data()
        
        # Add grade to first molecule (ID 1)
        updated_df = grading.add_grade(df, 1, 'A')
        
        # Find the row with ID 1
        mask = updated_df['id'] == 1
        assert updated_df.loc[mask, 'grade'].iloc[0] == 'A'
        assert pd.notna(updated_df.loc[mask, 'grade_timestamp'].iloc[0])
        
        # Verify other molecules unchanged
        mask_2 = updated_df['id'] == 2
        assert updated_df.loc[mask_2, 'grade'].iloc[0] == 'A'  # Was already graded
        
        mask_3 = updated_df['id'] == 3
        assert pd.isna(updated_df.loc[mask_3, 'grade'].iloc[0])  # Still ungraded
        
        # Test adding grade to non-existent molecule
        unchanged_df = grading.add_grade(df, 999, 'C')
        pd.testing.assert_frame_equal(unchanged_df, df)

    def test_graded_molecules_filtering(self):
        """Test filtering graded vs ungraded molecules."""
        df = self.create_grading_test_data()
        
        # Test graded molecules
        graded = grading.get_graded_molecules(df)
        assert len(graded) == 2
        assert all(graded['grade'].notna())
        assert set(graded['grade']) == {'A', 'B'}
        
        # Test ungraded molecules
        ungraded = grading.get_ungraded_molecules(df)
        assert len(ungraded) == 3
        assert all(ungraded['grade'].isna())

    def test_grading_statistics(self):
        """Test grading statistics calculation."""
        df = self.create_grading_test_data()
        
        stats = grading.get_grading_statistics(df)
        
        # Essential statistics checks
        assert stats['total_molecules'] == 5
        assert stats['graded_count'] == 2
        assert stats['ungraded_count'] == 3
        assert stats['grading_percentage'] == 40.0
        
        # Grade distribution
        assert 'grade_distribution' in stats
        assert stats['grade_distribution']['A'] == 1
        assert stats['grade_distribution']['B'] == 1

    def test_molecule_sorting_strategies(self):
        """Test different molecule sorting strategies."""
        df = self.create_grading_test_data()
        
        # Test Best Score strategy (lower is better by default)
        sorted_by_score = grading.get_molecules_by_strategy(df, 'Best Score')
        scores = sorted_by_score['score'].tolist()
        assert scores == sorted(scores)  # Should be ascending
        
        # Verify we get ungraded molecules sorted by score
        assert len(sorted_by_score) > 0
        if len(scores) > 0:
            assert scores[0] <= scores[-1]  # First <= last (ascending order)
        
        # Test Best Predictions strategy
        sorted_by_pred = grading.get_molecules_by_strategy(df, 'Best Predictions')
        # Should prioritize ungraded molecules and sort by prediction
        assert len(sorted_by_pred) > 0
        
        # Test Highest Uncertainty strategy
        sorted_by_uncertainty = grading.get_molecules_by_strategy(df, 'Highest Uncertainty')
        # Should work even with some None uncertainties
        assert len(sorted_by_uncertainty) > 0
        
        # Test Random strategy
        sorted_random = grading.get_molecules_by_strategy(df, 'Random')
        assert len(sorted_random) == 3  # Only ungraded molecules
        assert set(sorted_random['id']) == {1, 3, 5}  # Ungraded molecule IDs

    def test_model_detection(self):
        """Test detection of trained models."""
        df = self.create_grading_test_data()
        
        # Initially should detect model (has some predictions)
        assert grading.has_trained_model(df) == True
        
        # Remove all predictions
        df_no_model = df.copy()
        df_no_model['prediction'] = None
        assert grading.has_trained_model(df_no_model) == False


class TestDataIntegrity:
    """Test data integrity after various operations."""
    
    def test_dataframe_immutability(self):
        """Test that operations don't mutate original DataFrames."""
        original_df = pd.DataFrame({
            'id': [1, 2, 3],
            'grade': [None, 'A', None],
            'score': [1.0, 2.0, 3.0]
        })
        
        # Make a copy to compare against
        original_copy = original_df.copy()
        
        # Perform operations that should not mutate original
        grading.add_grade(original_df, 0, 'B')
        grading.get_graded_molecules(original_df)
        grading.get_ungraded_molecules(original_df)
        grading.get_molecules_by_strategy(original_df, 'Random')
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(original_df, original_copy)

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
        success = sessions.save_session(session_id, df, metadata)
        assert success is True
        
        # Load and verify metadata preservation
        result = sessions.load_session(session_id)
        assert result is not None
        
        loaded_df, loaded_metadata = result
        
        assert loaded_metadata == metadata
        assert loaded_metadata['session_name'] == 'test_session'
        assert loaded_metadata['num_molecules'] == 2

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        # Create empty DataFrame with required columns for graceful handling
        empty_df = pd.DataFrame(columns=['grade'])
        
        # Should handle gracefully without errors
        stats = grading.get_grading_statistics(empty_df)
        assert stats['total_molecules'] == 0
        assert stats['graded_count'] == 0
        assert stats['ungraded_count'] == 0
        
        graded = grading.get_graded_molecules(empty_df)
        assert len(graded) == 0
        
        ungraded = grading.get_ungraded_molecules(empty_df)
        assert len(ungraded) == 0


if __name__ == '__main__':
    pytest.main([__file__])