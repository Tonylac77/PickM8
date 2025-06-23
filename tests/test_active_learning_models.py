"""
Test suite for core/active_learning/models.py functions.
Focuses on ML model creation, training, and prediction functions.
"""

import sys
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.active_learning.models import (
    create_ml_model,
    train_model_with_calibration,
    predict_with_uncertainty,
    save_model_and_encoders,
    load_model_and_encoders
)


class TestActivelearningModels:
    """Test suite for active learning models"""
    
    def create_sample_data(self, n_samples=100, n_features=10, n_classes=3):
        """Create sample training data"""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return X, y
    
    def test_create_ml_model_random_forest(self):
        """Test Random Forest model creation"""
        model = create_ml_model("random_forest", n_estimators=50, max_depth=5)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert model.random_state == 42  # Default value

    def test_create_ml_model_logistic_regression(self):
        """Test Logistic Regression model creation"""
        model = create_ml_model("logistic_regression", C=0.5, max_iter=500)
        
        assert isinstance(model, LogisticRegression)
        assert model.C == 0.5
        assert model.max_iter == 500
        assert model.random_state == 42

    def test_create_ml_model_svm(self):
        """Test SVM model creation"""
        model = create_ml_model("svm", C=2.0, kernel='linear')
        
        assert isinstance(model, SVC)
        assert model.C == 2.0
        assert model.kernel == 'linear'
        assert model.probability == True  # Should be enabled for uncertainty estimation

    def test_create_ml_model_unknown_type(self):
        """Test error handling for unknown model type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_ml_model("unknown_model")

    def test_train_model_with_calibration_empty_data(self):
        """Test training with empty data"""
        X = np.array([])
        y = np.array([])
        
        with pytest.raises(ValueError, match="Empty training data"):
            train_model_with_calibration(X, y)

    def test_train_model_with_calibration_sufficient_data(self):
        """Test training with sufficient data for calibration"""
        X, y = self.create_sample_data(n_samples=50, n_classes=2)
        
        model, metrics = train_model_with_calibration(
            X, y, model_type="random_forest", use_calibration=True
        )
        
        assert model is not None
        assert 'train_accuracy' in metrics
        assert isinstance(metrics['train_accuracy'], float)
        assert 0.0 <= metrics['train_accuracy'] <= 1.0

    def test_train_model_with_calibration_insufficient_data(self):
        """Test training with insufficient data for calibration"""
        X, y = self.create_sample_data(n_samples=10, n_classes=2)
        
        model, metrics = train_model_with_calibration(
            X, y, model_type="random_forest", use_calibration=True
        )
        
        assert model is not None
        assert 'train_accuracy' in metrics
        # Should use base model due to insufficient data

    def test_train_model_without_calibration(self):
        """Test training without calibration"""
        X, y = self.create_sample_data(n_samples=30, n_classes=2)
        
        model, metrics = train_model_with_calibration(
            X, y, model_type="logistic_regression", use_calibration=False
        )
        
        assert model is not None
        assert isinstance(model, LogisticRegression)
        assert 'train_accuracy' in metrics

    def test_train_model_single_class(self):
        """Test training with single class data"""
        X, _ = self.create_sample_data(n_samples=20, n_classes=1)
        y = np.zeros(20)  # All same class
        
        model, metrics = train_model_with_calibration(X, y)
        
        assert model is not None
        assert metrics['train_accuracy'] == 1.0  # Should be 1.0 for single class

    def test_predict_with_uncertainty_empty_data(self):
        """Test prediction with empty data"""
        model = create_ml_model("random_forest")
        X = np.array([]).reshape(0, 5)  # Empty with 5 features
        
        predictions, probabilities, uncertainties = predict_with_uncertainty(model, X)
        
        assert len(predictions) == 0
        assert len(probabilities) == 0
        assert len(uncertainties) == 0

    def test_predict_with_uncertainty_trained_model(self):
        """Test prediction with trained model"""
        X_train, y_train = self.create_sample_data(n_samples=50, n_features=5, n_classes=3)
        X_test, _ = self.create_sample_data(n_samples=10, n_features=5, n_classes=3)
        
        model, _ = train_model_with_calibration(X_train, y_train, use_calibration=False)
        predictions, probabilities, uncertainties = predict_with_uncertainty(model, X_test)
        
        assert len(predictions) == 10
        assert len(probabilities) == 10
        assert len(uncertainties) == 10
        assert probabilities.shape[1] == 3  # 3 classes
        assert all(0.0 <= u <= 1.0 for u in uncertainties)

    @patch('core.active_learning.models.pickle.dump')
    @patch('core.active_learning.models.json.dump')
    def test_save_model_and_encoders(self, mock_json_dump, mock_pickle_dump):
        """Test model and encoders saving"""
        model = create_ml_model("random_forest")
        encoders = {'grade': {'A': 0, 'B': 1, 'C': 2}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_and_encoders(model, encoders, tmpdir, "test_model")
            
            # Check that files would be created
            model_path = Path(tmpdir) / "test_model.pkl"
            encoders_path = Path(tmpdir) / "test_model_encoders.json"
            
            # Verify paths exist (directories created)
            assert model_path.parent.exists()
            assert encoders_path.parent.exists()
            
            # Verify dump functions were called
            mock_pickle_dump.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_save_and_load_model_round_trip(self):
        """Test model save and load round trip"""
        X, y = self.create_sample_data(n_samples=30, n_classes=2)
        model, _ = train_model_with_calibration(X, y, use_calibration=False)
        encoders = {'grade': {'A': 0, 'B': 1}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_model_and_encoders(model, encoders, tmpdir, "test_model")
            
            # Load
            loaded_model, loaded_encoders = load_model_and_encoders(tmpdir, "test_model")
            
            assert loaded_model is not None
            assert loaded_encoders is not None
            assert loaded_encoders['grade'] == encoders['grade']
            
            # Test that loaded model can make predictions
            X_test, _ = self.create_sample_data(n_samples=5, n_features=X.shape[1], n_classes=2)
            predictions = loaded_model.predict(X_test)
            assert len(predictions) == 5

    def test_load_model_and_encoders_not_found(self):
        """Test loading when files don't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            model, encoders = load_model_and_encoders(tmpdir, "nonexistent_model")
            
            assert model is None
            assert encoders is None

    def test_predict_with_uncertainty_no_proba_method(self):
        """Test prediction with model that doesn't have predict_proba"""
        # Create a mock model without predict_proba
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0])
        # Remove predict_proba method
        del mock_model.predict_proba
        
        X = np.random.randn(3, 5)
        
        predictions, probabilities, uncertainties = predict_with_uncertainty(mock_model, X)
        
        assert len(predictions) == 3
        assert len(probabilities) == 3
        assert len(uncertainties) == 3
        assert all(u == 0.5 for u in uncertainties)  # Medium uncertainty fallback

    def test_train_model_with_gradient_boosting(self):
        """Test training with Gradient Boosting model"""
        X, y = self.create_sample_data(n_samples=40, n_classes=2)
        
        model, metrics = train_model_with_calibration(
            X, y, model_type="gradient_boosting", 
            n_estimators=50, learning_rate=0.05
        )
        
        assert model is not None
        assert 'train_accuracy' in metrics

    @patch('core.active_learning.models.CalibratedClassifierCV')
    def test_train_model_calibration_failure(self, mock_calibrated):
        """Test handling of calibration failure"""
        X, y = self.create_sample_data(n_samples=50, n_classes=2)
        
        # Mock calibration to raise an exception
        mock_calibrated.side_effect = Exception("Calibration failed")
        
        model, metrics = train_model_with_calibration(
            X, y, model_type="random_forest", use_calibration=True
        )
        
        # Should fall back to base model
        assert model is not None
        assert isinstance(model, RandomForestClassifier)
        assert 'train_accuracy' in metrics


if __name__ == '__main__':
    pytest.main([__file__])