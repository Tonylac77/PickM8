"""Test suite for analysis/grading.py functions."""

import pytest
import pandas as pd
import numpy as np
import json

from analysis import grading


class TestGrading:
    """Test suite for grading analysis functions."""
    
    def create_sample_dataframe(self, n_molecules=10):
        """Create sample molecules DataFrame for testing."""
        np.random.seed(42)
        
        data = {
            'id': list(range(n_molecules)),
            'name': [f'mol_{i}' for i in range(n_molecules)],
            'score': np.random.uniform(-10, 0, n_molecules),
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(n_molecules)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(n_molecules)],
            'interaction_fp': [json.dumps([1, 0, 1, 0])] * n_molecules,
            'grade': [None] * n_molecules,
            'prediction': [None] * n_molecules,
            'prediction_uncertainty': [None] * n_molecules
        }
        
        return pd.DataFrame(data)
    
    def test_add_grade(self):
        """Test adding grades to molecules."""
        df = self.create_sample_dataframe(5)
        
        # Add a grade
        updated_df = grading.add_grade(df, 0, 'A')
        
        assert updated_df.loc[0, 'grade'] == 'A'
        assert pd.notna(updated_df.loc[0, 'grade_timestamp'])
        
        # Try adding grade to non-existent molecule
        result_df = grading.add_grade(df, 999, 'B')
        assert result_df.equals(df)  # Should be unchanged
    
    def test_get_graded_molecules(self):
        """Test filtering graded molecules."""
        df = self.create_sample_dataframe(5)
        df.loc[0:2, 'grade'] = ['A', 'B', 'C']
        
        graded = grading.get_graded_molecules(df)
        
        assert len(graded) == 3
        assert graded['grade'].notna().all()
    
    def test_get_ungraded_molecules(self):
        """Test filtering ungraded molecules."""
        df = self.create_sample_dataframe(5)
        df.loc[0:2, 'grade'] = ['A', 'B', 'C']
        
        ungraded = grading.get_ungraded_molecules(df)
        
        assert len(ungraded) == 2
        assert ungraded['grade'].isna().all()
    
    def test_get_grading_statistics(self):
        """Test grading statistics calculation."""
        df = self.create_sample_dataframe(10)
        df.loc[0:4, 'grade'] = ['A', 'B', 'C', 'A', 'B']
        
        stats = grading.get_grading_statistics(df)
        
        assert stats['total_molecules'] == 10
        assert stats['graded_count'] == 5
        assert stats['ungraded_count'] == 5
        assert stats['grading_percentage'] == 50.0
        assert 'grade_distribution' in stats
        assert stats['grade_distribution']['A'] == 2
        assert stats['grade_distribution']['B'] == 2
        assert stats['grade_distribution']['C'] == 1
    
    def test_has_trained_model(self):
        """Test model detection."""
        df = self.create_sample_dataframe(5)
        
        # No model
        assert grading.has_trained_model(df) == False
        
        # Add predictions
        df.loc[0:2, 'prediction'] = [0, 1, 2]
        assert grading.has_trained_model(df) == True
    
    def test_best_predictions_sorting(self):
        """Test that Best Predictions shows A grades first, then B, C, etc."""
        df = self.create_sample_dataframe(6)
        
        # Set predictions: A=0, B=1, C=2
        df.loc[:, 'prediction'] = [2, 0, 1, 2, 1, 0]  # C, A, B, C, B, A
        
        # All molecules should be ungraded for this test
        assert df['grade'].isna().all()
        
        # Test Best Predictions strategy
        sorted_df = grading.get_molecules_by_strategy(df, 'Best Predictions')
        
        # Should be sorted in ascending order of prediction values (A=0 first, then B=1, then C=2)
        expected_order = [0, 0, 1, 1, 2, 2]
        actual_order = sorted_df['prediction'].tolist()
        
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
        
        # Verify that A grades (prediction=0) come first
        assert sorted_df.iloc[0]['prediction'] == 0
        assert sorted_df.iloc[1]['prediction'] == 0
    
    def test_highest_uncertainty_sorting(self):
        """Test Highest Uncertainty strategy."""
        df = self.create_sample_dataframe(5)
        df.loc[:, 'prediction_uncertainty'] = [0.1, 0.8, 0.3, 0.9, 0.2]
        
        sorted_df = grading.get_molecules_by_strategy(df, 'Highest Uncertainty')
        
        # Should be sorted in descending order of uncertainty
        uncertainties = sorted_df['prediction_uncertainty'].tolist()
        assert uncertainties == sorted(uncertainties, reverse=True)
        assert uncertainties[0] == 0.9  # Highest uncertainty first
    
    def test_best_score_sorting(self):
        """Test Best Score strategy with different score directions."""
        df = self.create_sample_dataframe(5)
        df.loc[:, 'score'] = [-2.5, -8.1, -4.3, -1.2, -6.7]
        
        # Test default (lower is better)
        sorted_df = grading.get_molecules_by_strategy(df, 'Best Score')
        scores = sorted_df['score'].tolist()
        assert scores == sorted(scores)  # Ascending order
        assert scores[0] == -8.1  # Lowest (best) score first
        
        # Test with metadata specifying higher is better
        metadata = {'score_direction': 'Higher is better'}
        sorted_df = grading.get_molecules_by_strategy(df, 'Best Score', metadata)
        scores = sorted_df['score'].tolist()
        assert scores == sorted(scores, reverse=True)  # Descending order
        assert scores[0] == -1.2  # Highest (best) score first
    
    def test_random_strategy(self):
        """Test Random strategy."""
        df = self.create_sample_dataframe(5)
        
        # Random should return all ungraded molecules
        sorted_df = grading.get_molecules_by_strategy(df, 'Random')
        assert len(sorted_df) == 5
        
        # Order might be different but should contain same molecules
        assert set(sorted_df['id']) == set(df['id'])
    
    def test_unknown_strategy(self):
        """Test unknown strategy defaults to Best Score."""
        df = self.create_sample_dataframe(5)
        df.loc[:, 'score'] = [-2.5, -8.1, -4.3, -1.2, -6.7]
        
        sorted_df = grading.get_molecules_by_strategy(df, 'Unknown Strategy')
        
        # Should default to Best Score behavior
        scores = sorted_df['score'].tolist()
        assert scores == sorted(scores)  # Ascending order (lower is better)
    
    def test_filter_and_sort_edge_cases(self):
        """Test edge cases in filtering and sorting."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        result = grading.filter_and_sort_molecules(empty_df)
        assert len(result) == 0
        
        # DataFrame with no predictions
        df = self.create_sample_dataframe(3)
        result = grading.filter_and_sort_molecules(df, sort_by='best_prediction')
        assert len(result) == 3  # Should not crash
        
        # DataFrame with no uncertainty data
        result = grading.filter_and_sort_molecules(df, sort_by='uncertainty')
        assert len(result) == 3  # Should not crash


if __name__ == '__main__':
    pytest.main([__file__])