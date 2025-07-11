"""
Tests for grading operations and workflow.
"""

import pytest
import pandas as pd
import numpy as np

from analysis.grading import *


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
        })

    def test_add_grade_workflow(self):
        """Test adding grades to molecules."""
        df = self.create_grading_test_data()
        
        # Add grade to first molecule (ID 1)
        updated_df = add_grade(df, 1, 'A')
        
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
        unchanged_df = add_grade(df, 999, 'C')
        pd.testing.assert_frame_equal(unchanged_df, df)

    def test_graded_molecules_filtering(self):
        """Test filtering graded vs ungraded molecules."""
        df = self.create_grading_test_data()
        
        # Test graded molecules
        graded = get_graded_molecules(df)
        assert len(graded) == 2
        assert all(graded['grade'].notna())
        assert set(graded['grade']) == {'A', 'B'}
        
        # Test ungraded molecules
        ungraded = get_ungraded_molecules(df)
        assert len(ungraded) == 3
        assert all(ungraded['grade'].isna())

    def test_grading_statistics(self):
        """Test grading statistics calculation."""
        df = self.create_grading_test_data()
        
        stats = get_grading_statistics(df)
        
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
        sorted_by_score = get_molecules_by_strategy(df, 'Best Score')
        scores = sorted_by_score['score'].tolist()
        assert scores == sorted(scores)  # Should be ascending
        
        # Verify we get ungraded molecules sorted by score
        assert len(sorted_by_score) > 0
        if len(scores) > 0:
            assert scores[0] <= scores[-1]  # First <= last (ascending order)
        
        # Test Best Predictions strategy
        sorted_by_pred = get_molecules_by_strategy(df, 'Best Predictions')
        # Should prioritize ungraded molecules and sort by prediction
        assert len(sorted_by_pred) > 0
        
        
        # Test Random strategy
        sorted_random = get_molecules_by_strategy(df, 'Random')
        assert len(sorted_random) == 3  # Only ungraded molecules
        assert set(sorted_random['id']) == {1, 3, 5}  # Ungraded molecule IDs

    def test_model_detection(self):
        """Test detection of trained models."""
        df = self.create_grading_test_data()
        
        # Initially should detect model (has some predictions)
        assert has_trained_model(df) == True
        
        # Remove all predictions
        df_no_model = df.copy()
        df_no_model['prediction'] = None
        assert has_trained_model(df_no_model) == False

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        # Create empty DataFrame with required columns for graceful handling
        empty_df = pd.DataFrame(columns=['grade'])
        
        # Should handle gracefully without errors
        stats = get_grading_statistics(empty_df)
        assert stats['total_molecules'] == 0
        assert stats['graded_count'] == 0
        assert stats['ungraded_count'] == 0
        
        graded = get_graded_molecules(empty_df)
        assert len(graded) == 0
        
        ungraded = get_ungraded_molecules(empty_df)
        assert len(ungraded) == 0


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
        add_grade(original_df, 0, 'B')
        get_graded_molecules(original_df)
        get_ungraded_molecules(original_df)
        get_molecules_by_strategy(original_df, 'Random')
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(original_df, original_copy)


if __name__ == '__main__':
    pytest.main([__file__])