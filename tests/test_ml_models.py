"""Test suite for data/ml_models.py - simplified ML functions."""

import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, Mock

from data import ml_models


class TestMLModels:
    """Test suite for ML model functions."""
    
    def create_sample_dataframe(self, n_molecules=20, add_random_grades=False):
        """Create sample molecules DataFrame for testing."""
        np.random.seed(42)
        
        # Create base DataFrame
        data = {
            'id': list(range(n_molecules)),
            'name': [f'mol_{i}' for i in range(n_molecules)],
            'smiles': ['CCO'] * n_molecules,
            'score': np.random.uniform(-10, 0, n_molecules),
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(n_molecules)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(n_molecules)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(n_molecules)],
            'grade': [None] * n_molecules  # Initialize with no grades
        }
        
        df = pd.DataFrame(data)
        
        # Optionally add random grades to some molecules
        if add_random_grades:
            grades = ['A', 'B', 'C', 'D']
            grade_indices = np.random.choice(n_molecules, size=min(10, n_molecules), replace=False)
            df.loc[grade_indices, 'grade'] = np.random.choice(grades, size=len(grade_indices))
        
        return df
    
    def test_prepare_features_from_dataframe(self):
        """Test feature extraction from DataFrame."""
        df = self.create_sample_dataframe(5)
        
        X, mol_ids = ml_models.prepare_features_from_dataframe(df)
        
        assert len(X) == 5
        assert len(mol_ids) == 5
        assert X.shape[1] > 0  # Should have features
        assert mol_ids == list(range(5))
    
    def test_prepare_features_empty_dataframe(self):
        """Test feature extraction with empty DataFrame."""
        df = pd.DataFrame()
        
        X, mol_ids = ml_models.prepare_features_from_dataframe(df)
        
        assert len(X) == 0
        assert len(mol_ids) == 0
    
    def test_encode_grades_for_training(self):
        """Test grade encoding."""
        df = self.create_sample_dataframe(10)
        # Ensure we have some grades
        df.loc[0:4, 'grade'] = ['A', 'B', 'C', 'A', 'B']
        
        y, label_mapping = ml_models.encode_grades_for_training(df)
        
        assert len(y) == 5  # Only graded molecules
        assert len(label_mapping) == 3  # A, B, C
        assert 'A' in label_mapping
        assert 'B' in label_mapping
        assert 'C' in label_mapping
    
    def test_encode_grades_no_grades(self):
        """Test grade encoding with no grades."""
        df = self.create_sample_dataframe(5)
        df['grade'] = None  # No grades
        
        y, label_mapping = ml_models.encode_grades_for_training(df)
        
        assert len(y) == 0
        assert len(label_mapping) == 0
    
    def test_train_model_success(self):
        """Test successful model training."""
        df = self.create_sample_dataframe(20)
        # Add sufficient grades for training
        df.loc[0:9, 'grade'] = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
        
        model, metrics = ml_models.train_model(df)
        
        assert model is not None
        assert 'accuracy' in metrics
        assert 'n_samples' in metrics
        assert 'n_features' in metrics
        assert 'label_mapping' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert metrics['n_samples'] == 10
    
    def test_train_model_insufficient_data(self):
        """Test training with insufficient data."""
        df = self.create_sample_dataframe(5)
        # Only 2 graded molecules
        df.loc[0:1, 'grade'] = ['A', 'B']
        
        with pytest.raises(ValueError, match="Need at least 3 graded molecules"):
            ml_models.train_model(df)
    
    def test_train_model_no_features(self):
        """Test training with molecules that have no valid features."""
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'grade': ['A', 'B', 'C'],
            'morgan_fp': [None, None, None],
            'rdkit_fp': [None, None, None],
            'interaction_fp': [None, None, None]
        })
        
        with pytest.raises(ValueError, match="No valid features found"):
            ml_models.train_model(df)
    
    def test_update_predictions(self):
        """Test prediction updates."""
        df = self.create_sample_dataframe(10)
        # Train a model first
        df.loc[0:4, 'grade'] = ['A', 'B', 'C', 'A', 'B']
        model, _ = ml_models.train_model(df)
        
        # Update predictions
        updated_df = ml_models.update_predictions(df, model)
        
        assert 'prediction' in updated_df.columns
        assert 'prediction_uncertainty' in updated_df.columns
        assert 'prediction_timestamp' in updated_df.columns
        
        # Check that predictions were added
        predictions = updated_df['prediction'].dropna()
        assert len(predictions) > 0
    
    def test_update_predictions_no_features(self):
        """Test prediction updates with no valid features."""
        df = pd.DataFrame({
            'id': [1, 2],
            'morgan_fp': [None, None],
            'rdkit_fp': [None, None],
            'interaction_fp': [None, None]
        })
        
        # Mock model
        from unittest.mock import Mock
        model = Mock()
        
        updated_df = ml_models.update_predictions(df, model)
        
        # Should return unchanged DataFrame
        assert len(updated_df) == 2
        model.predict.assert_not_called()
    
    @patch('data.ml_models.CalibratedClassifierCV')
    def test_train_model_with_calibration(self, mock_calibrated):
        """Test model training uses calibration with sufficient data."""
        df = self.create_sample_dataframe(30)
        # Add sufficient grades for calibration (at least 3 per class for 3-fold CV)
        grades = ['A'] * 6 + ['B'] * 6 + ['C'] * 6  # 18 total, 6 per class
        df.loc[0:17, 'grade'] = grades
        
        # Mock to return a fitted model
        mock_instance = Mock()
        mock_instance.predict.return_value = np.array([0, 1, 2] * 6)
        mock_calibrated.return_value = mock_instance
        
        model, metrics = ml_models.train_model(df)
        
        # Should use calibration
        mock_calibrated.assert_called_once()
        assert model is not None
    
    def test_train_model_no_calibration_insufficient_data(self):
        """Test model training without calibration for small datasets."""
        df = self.create_sample_dataframe(10)
        # Small dataset - should not use calibration
        df.loc[0:4, 'grade'] = ['A', 'B', 'A', 'B', 'A']
        
        model, metrics = ml_models.train_model(df)
        
        assert model is not None
        # Should be base RandomForest, not calibrated
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)


class TestEncodingIntegration:
    """Test encoding integration with ML models."""
    
    def create_graded_dataframe(self, n_molecules=10):
        """Create DataFrame with grades for encoding tests."""
        np.random.seed(42)
        
        data = {
            'id': list(range(n_molecules)),
            'name': [f'mol_{i}' for i in range(n_molecules)],
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(n_molecules)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(n_molecules)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(n_molecules)],
            'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'][:n_molecules]
        }
        
        return pd.DataFrame(data)
    
    def test_encode_grades_for_training_sequential(self):
        """Test encoding grades with sequential encoding."""
        df = self.create_graded_dataframe()
        
        encoded, mapping = ml_models.encode_grades_for_training(df, 'sequential')
        
        assert len(encoded) == len(df)
        assert isinstance(encoded, np.ndarray)
        assert isinstance(mapping, dict)
        assert 'A' in mapping and 'B' in mapping and 'C' in mapping
    
    def test_encode_grades_for_training_nominal(self):
        """Test encoding grades with nominal encoding."""
        df = self.create_graded_dataframe()
        
        encoded, mapping = ml_models.encode_grades_for_training(df, 'nominal')
        
        assert encoded.shape[0] == len(df)
        assert encoded.shape[1] == 3  # A, B, C
        assert isinstance(mapping, dict)
    
    def test_encode_grades_for_training_ordinal(self):
        """Test encoding grades with ordinal encoding."""
        df = self.create_graded_dataframe()
        
        encoded, mapping = ml_models.encode_grades_for_training(df, 'ordinal')
        
        assert encoded.shape[0] == len(df)
        assert encoded.shape[1] == 3  # A, B, C
        assert isinstance(mapping, dict)
    
    def test_train_model_with_encoding_type(self):
        """Test training model with specific encoding type."""
        df = self.create_graded_dataframe()
        
        model_config = {
            'model_type': 'RandomForest',
            'encoding_type': 'sequential'
        }
        
        model, metrics = ml_models.train_model(df, model_config)
        
        assert model is not None
        assert metrics['encoding_type'] == 'sequential'
        assert 'ml_strategy' in metrics
        assert metrics['ml_strategy'] == 'regression'
    
    def test_update_predictions_with_metrics(self):
        """Test updating predictions with encoding metrics."""
        df = self.create_graded_dataframe()
        
        # Train model to get metrics
        model, metrics = ml_models.train_model(df)
        
        # Update predictions with metrics
        updated_df = ml_models.update_predictions(df, model, metrics)
        
        assert 'prediction' in updated_df.columns
        # Predictions should be grade strings in new system
        predictions = updated_df['prediction'].dropna()
        if len(predictions) > 0:
            # Should be valid grades or fallback format
            for pred in predictions:
                assert isinstance(pred, str)
    
    def test_load_encoding_config(self):
        """Test loading encoding configuration."""
        config = ml_models.load_encoding_config()
        
        assert isinstance(config, dict)
        assert 'type' in config
        assert 'default_grades' in config
        assert config['type'] in ['sequential', 'nominal', 'ordinal']
    
    def test_ensure_backward_compatibility(self):
        """Test backward compatibility function."""
        df = self.create_graded_dataframe()
        
        # Test with no model config
        config = ml_models.ensure_backward_compatibility(df, None)
        assert config['encoding_type'] == 'sequential'
        
        # Test with existing config
        existing_config = {'model_type': 'RandomForest'}
        config = ml_models.ensure_backward_compatibility(df, existing_config)
        assert config['encoding_type'] == 'sequential'
        assert config['model_type'] == 'RandomForest'
        
        # Test with invalid encoding type
        invalid_config = {'encoding_type': 'invalid_type'}
        config = ml_models.ensure_backward_compatibility(df, invalid_config)
        assert config['encoding_type'] == 'sequential'
    
    @patch('data.ml_models.load_encoding_config')
    def test_encoding_config_fallback(self, mock_load_config):
        """Test encoding configuration fallback behavior."""
        # Mock config loading to return None
        mock_load_config.return_value = {'type': 'sequential'}
        
        df = self.create_graded_dataframe()
        encoded, mapping = ml_models.encode_grades_for_training(df)
        
        assert len(encoded) == len(df)
        assert isinstance(mapping, dict)


if __name__ == '__main__':
    pytest.main([__file__])