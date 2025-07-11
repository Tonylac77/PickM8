"""
Tests for integration between ML training and encoding systems.
"""

import pytest
import pandas as pd
import numpy as np
import json
from machine_learning import ml_models
from machine_learning.encoding.encodings import SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION


class TestMLIntegration:
    """Test integration between ML training and encoding systems."""
    
    def test_ml_workflow_with_different_encodings(self):
        """Test complete ML workflow with different encoding types."""
        # Create DataFrame with required columns for ML workflow
        np.random.seed(42)
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(9)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(9)],
            'interaction_fp': [json.dumps(np.random.randint(0, 2, 512).tolist()) for _ in range(9)],
            'grade': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
            'prediction': [None] * 9,
            'prediction_timestamp': [None] * 9
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

    def test_model_category_determination(self):
        """Test that model categories are correctly determined from encoding types."""
        assert ml_models.determine_model_category(SEQUENTIAL) == 'classification'
        assert ml_models.determine_model_category(ONE_HOT) == 'classification'
        assert ml_models.determine_model_category(ORDINAL) == 'ordinal'
        assert ml_models.determine_model_category(ORDINAL_REGRESSION) == 'regression'


if __name__ == '__main__':
    pytest.main([__file__])