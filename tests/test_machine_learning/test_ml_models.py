"""
Tests for machine learning model training and prediction.
"""

import pytest
import pandas as pd
import numpy as np
import json

from machine_learning import ml_models
from machine_learning.models.ml_base import MLModelBase
from machine_learning.encoding.encodings import SEQUENTIAL, ORDINAL_REGRESSION


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
            # Use only A, B, C to avoid cross-validation issues with small samples
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
        assert len(metrics['label_mapping']) >= 3  # A, B, C, possibly D

    def test_prediction_generation(self):
        """Test prediction generation with trained model."""
        df = self.create_realistic_dataframe(n_molecules=20, n_graded=10)
        
        # Train model
        model, metrics = ml_models.train_model(df)
        
        # Generate predictions
        updated_df = ml_models.update_predictions(df, model, metrics)
        
        # Essential checks
        assert 'prediction' in updated_df.columns
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


class TestCurrentModelIntegration:
    """Test integration of currently supported models (RandomForest, GaussianProcess, LogisticAT)."""
    
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
            # Use only A, B, C to avoid cross-validation issues with small samples
            grades = ['A', 'B', 'C']
            grade_counts = [n_graded // 3, n_graded // 3, n_graded - 2 * (n_graded // 3)]
            grade_list = []
            for grade, count in zip(grades, grade_counts):
                grade_list.extend([grade] * count)
            
            df.loc[:n_graded-1, 'grade'] = grade_list
        
        return df
    
    def test_random_forest_classification(self):
        """Test RandomForest with classification (sequential encoding)."""
        df = self.create_realistic_dataframe(n_molecules=30, n_graded=15)
        
        model_config = {
            'model_type': 'RandomForest',
            'encoding_type': SEQUENTIAL,
            'model_params': {
                'n_estimators': 10,  # Small for testing
                'max_depth': 3,
                'random_state': 42
            }
        }
        
        model, metrics = ml_models.train_model(df, model_config)
        
        assert model is not None
        assert isinstance(model, MLModelBase)
        assert model.backend == 'sklearn'
        assert model.is_classifier
        assert metrics['model_type'] == 'RandomForest'
        assert 0.0 <= metrics['accuracy'] <= 1.0
    
    def test_random_forest_regression(self):
        """Test RandomForest with regression (ordinal_regression encoding)."""
        df = self.create_realistic_dataframe(n_molecules=20, n_graded=10)
        
        model_config = {
            'model_type': 'RandomForest',
            'encoding_type': ORDINAL_REGRESSION,
            'model_params': {
                'n_estimators': 10,
                'random_state': 42
            }
        }
        
        # Train model
        model, metrics = ml_models.train_model(df, model_config)
        
        assert model is not None
        assert model.is_regressor
        assert metrics['model_type'] == 'RandomForest'
        assert metrics['encoding_type'] == ORDINAL_REGRESSION
    
    def test_logistic_at_sequential(self):
        """Test LogisticAT with sequential encoding."""
        df = self.create_realistic_dataframe(n_molecules=20, n_graded=10)
        
        model_config = {
            'model_type': 'LogisticAT',
            'encoding_type': SEQUENTIAL,
            'model_params': {
                'alpha': 1.0,
                'max_iter': 100
            }
        }
        
        try:
            # Train model
            model, metrics = ml_models.train_model(df, model_config)
            
            assert model is not None
            assert hasattr(model, 'model_category') and model.model_category == 'ordinal'
            assert metrics['model_type'] == 'LogisticAT'
            assert metrics['encoding_type'] == SEQUENTIAL
            
        except ImportError:
            pytest.skip("mord library not available")
    
    def test_gaussian_process_classification(self):
        """Test GaussianProcess with classification encoding."""
        df = self.create_realistic_dataframe(n_molecules=15, n_graded=8)  # Smaller for GP
        
        model_config = {
            'model_type': 'GaussianProcess',
            'encoding_type': SEQUENTIAL,
            'model_params': {
                'kernel': 'RBF',
                'n_restarts_optimizer': 0  # Faster for testing
            }
        }
        
        # Train model
        model, metrics = ml_models.train_model(df, model_config)
        
        assert model is not None
        assert model.is_classifier
        assert metrics['model_type'] == 'GaussianProcess'
        
        # Test predictions
        X, _ = ml_models.prepare_features_from_dataframe(df)
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)


if __name__ == '__main__':
    pytest.main([__file__])