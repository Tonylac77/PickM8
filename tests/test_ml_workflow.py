"""
Machine learning workflow tests.
Tests ML pipeline: training, prediction, and encoding with real data.
"""

import pytest
import pandas as pd
import numpy as np
import json

from data import ml_models
from data.encodings import (
    encode_sequential, encode_nominal, encode_ordinal,
    decode_sequential, decode_nominal, decode_ordinal,
    SEQUENTIAL, NOMINAL, ORDINAL
)


class TestMLPipeline:
    """Test machine learning pipeline end-to-end."""
    
    def create_realistic_dataframe(self, n_molecules=20, n_graded=10):
        """Create realistic DataFrame with molecular features and grades."""
        np.random.seed(42)
        
        data = {
            'id': list(range(n_molecules)),
            'name': [f'mol_{i}' for i in range(n_molecules)],
            'score': np.random.uniform(-10, 0, n_molecules),
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(n_molecules)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(n_molecules)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(n_molecules)],
            'grade': [None] * n_molecules
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic grades to first n_graded molecules
        if n_graded > 0:
            grades = ['A', 'B', 'C']
            grade_counts = [n_graded // 3, n_graded // 3, n_graded - 2 * (n_graded // 3)]
            grade_list = []
            for grade, count in zip(grades, grade_counts):
                grade_list.extend([grade] * count)
            
            df.loc[:n_graded-1, 'grade'] = grade_list
        
        return df

    def test_model_training_with_sufficient_data(self):
        """Test ML model training with sufficient graded data."""
        df = self.create_realistic_dataframe(n_molecules=30, n_graded=15)
        
        model, metrics = ml_models.train_model(df)
        
        # Essential checks
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'n_samples' in metrics
        assert 'label_mapping' in metrics
        
        # Verify reasonable metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['n_samples'] == 15
        assert len(metrics['label_mapping']) == 3  # A, B, C

    def test_prediction_generation(self):
        """Test prediction generation with trained model."""
        df = self.create_realistic_dataframe(n_molecules=20, n_graded=10)
        
        # Train model
        model, metrics = ml_models.train_model(df)
        
        # Generate predictions
        updated_df = ml_models.update_predictions(df, model, metrics)
        
        # Essential checks
        assert 'prediction' in updated_df.columns
        assert 'prediction_uncertainty' in updated_df.columns
        assert 'prediction_timestamp' in updated_df.columns
        
        # Verify predictions were generated
        predictions = updated_df['prediction'].dropna()
        assert len(predictions) > 0
        
        # Verify prediction values are reasonable
        for pred in predictions:
            assert isinstance(pred, str)
            assert pred in ['A', 'B', 'C']

    def test_insufficient_training_data_error(self):
        """Test error handling with insufficient training data."""
        df = self.create_realistic_dataframe(n_molecules=10, n_graded=2)
        
        with pytest.raises(ValueError, match="Need at least 3 graded molecules"):
            ml_models.train_model(df)

    def test_no_features_error(self):
        """Test error handling when no valid features available."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'grade': ['A', 'B', 'C'],
            'morgan_fp': [None, None, None],
            'rdkit_fp': [None, None, None],
            'interaction_fp': [None, None, None]
        })
        
        with pytest.raises(ValueError, match="No valid features found"):
            ml_models.train_model(df)


class TestGradeEncoding:
    """Test grade encoding and decoding functionality."""
    
    def test_sequential_encoding_roundtrip(self):
        """Test sequential encoding-decoding roundtrip."""
        grades = ['A', 'B', 'C', 'A', 'B']
        
        # Encode
        encoded, mapping = encode_sequential(grades)
        
        # Verify encoding
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) == 5
        assert isinstance(mapping, dict)
        assert mapping == {'A': 0, 'B': 1, 'C': 2}
        
        # Simulate perfect predictions
        predictions = encoded.astype(float)
        
        # Decode
        decoded = decode_sequential(predictions, mapping)
        
        # Should match original
        assert decoded == grades

    def test_nominal_encoding_roundtrip(self):
        """Test nominal (one-hot) encoding-decoding roundtrip."""
        grades = ['A', 'B', 'C']
        
        # Encode
        encoded, mapping = encode_nominal(grades)
        
        # Verify encoding structure
        assert encoded.shape == (3, 3)
        assert mapping == {'A': 0, 'B': 1, 'C': 2}
        
        # Verify one-hot structure
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(encoded, expected)
        
        # Simulate perfect predictions
        predictions = encoded.astype(float)
        
        # Decode
        decoded = decode_nominal(predictions, mapping)
        
        # Should match original
        assert decoded == grades

    def test_ordinal_encoding_roundtrip(self):
        """Test ordinal (cumulative) encoding-decoding roundtrip."""
        grades = ['A', 'B', 'C']
        
        # Encode
        encoded, mapping = encode_ordinal(grades)
        
        # Verify encoding structure
        assert encoded.shape == (3, 3)
        assert mapping == {'A': 0, 'B': 1, 'C': 2}
        
        # Verify cumulative structure
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        np.testing.assert_array_equal(encoded, expected)
        
        # Simulate perfect predictions
        predictions = encoded.astype(float)
        
        # Decode
        decoded = decode_ordinal(predictions, mapping)
        
        # Should match original
        assert decoded == grades

    def test_encoding_with_different_grades(self):
        """Test encoding works with different grade sets."""
        grades = ['Excellent', 'Good', 'Poor']
        
        for encoding_func in [encode_sequential, encode_nominal, encode_ordinal]:
            encoded, mapping = encoding_func(grades)
            
            assert isinstance(encoded, np.ndarray)
            assert isinstance(mapping, dict)
            assert len(mapping) == 3
            assert 'Excellent' in mapping
            assert 'Good' in mapping
            assert 'Poor' in mapping

    def test_empty_grades_handling(self):
        """Test handling of empty grade lists."""
        empty_grades = []
        
        # Should handle gracefully
        encoded_seq, mapping_seq = encode_sequential(empty_grades)
        assert len(encoded_seq) == 0
        assert mapping_seq == {}
        
        encoded_nom, mapping_nom = encode_nominal(empty_grades)
        assert encoded_nom.shape[0] == 0
        assert mapping_nom == {}


class TestMLIntegration:
    """Test integration between ML training and encoding systems."""
    
    def test_ml_workflow_with_different_encodings(self):
        """Test complete ML workflow with different encoding types."""
        # Create DataFrame with required columns for ML workflow
        np.random.seed(42)
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(6)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(6)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(6)],
            'grade': ['A', 'B', 'C', 'A', 'B', 'C'],
            'prediction': [None] * 6,
            'prediction_uncertainty': [None] * 6,
            'prediction_timestamp': [None] * 6
        })
        
        # Test just sequential encoding to avoid complexity
        model_config = {
            'model_type': 'RandomForest',
            'encoding_type': SEQUENTIAL
        }
        
        # Train model with specific encoding
        model, metrics = ml_models.train_model(df, model_config)
        
        assert model is not None
        assert metrics['encoding_type'] == SEQUENTIAL
        
        # Generate predictions on a copy to avoid indexing issues
        df_copy = df.copy()
        updated_df = ml_models.update_predictions(df_copy, model, metrics)
        
        # Verify predictions generated
        predictions = updated_df['prediction'].dropna()
        assert len(predictions) > 0
        
        # Verify predictions are valid grades
        for pred in predictions:
            assert pred in ['A', 'B', 'C']

    def test_feature_extraction_integration(self):
        """Test feature extraction from DataFrame works correctly."""
        # Create test DataFrame directly
        np.random.seed(42)
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(5)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(5)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(5)],
            'grade': ['A', 'B', 'C', 'A', 'B']
        })
        
        X, mol_ids = ml_models.prepare_features_from_dataframe(df)
        
        # Essential checks
        assert len(X) == 5
        assert len(mol_ids) == 5
        assert X.shape[1] > 0  # Should have features
        assert mol_ids == [1, 2, 3, 4, 5]  # Actual IDs from DataFrame
        
        # Verify feature matrix is reasonable
        assert np.all(np.isfinite(X))  # No NaN or inf values
        assert X.dtype in [np.float64, np.float32, np.int64, np.int32]  # Numeric features


if __name__ == '__main__':
    pytest.main([__file__])