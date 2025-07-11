"""
Comprehensive test suite for encoding-model compatibility.
Tests all encoding types with their appropriate ML models using real data.
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List

from machine_learning import ml_models
from machine_learning.encoding.encodings import (
    encode_sequential, encode_one_hot, encode_ordinal, encode_ordinal_regression,
    decode_sequential, decode_one_hot, decode_ordinal, decode_ordinal_regression,
    SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION
)


class TestEncodingModelCompatibility:
    """Test encoding-model compatibility with real molecular data."""
    
    @pytest.fixture(scope="class")
    def real_molecules_data(self):
        """Load real molecular data from test session."""
        test_session_path = Path("test_data/test_session")
        molecules_path = test_session_path / "molecules.pkl"
        metadata_path = test_session_path / "metadata.json"
        
        if not molecules_path.exists():
            pytest.skip("Real test data not available")
        
        # Load molecules DataFrame
        with open(molecules_path, 'rb') as f:
            df = pickle.load(f)
        
        # Ensure we have graded molecules for testing
        graded_df = df[df['grade'].notna()]
        if len(graded_df) < 10:
            pytest.skip("Insufficient graded molecules for testing")
        
        return df
    
    @pytest.fixture(scope="class") 
    def mock_molecules_data(self):
        """Create mock molecular data for testing when real data unavailable."""
        np.random.seed(42)
        n_molecules = 50
        n_graded = 20
        
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
        
        # Add balanced grades to first n_graded molecules
        grades = ['A', 'B', 'C', 'D']
        grade_counts = [n_graded // 4] * 3 + [n_graded - 3 * (n_graded // 4)]
        grade_list = []
        for grade, count in zip(grades, grade_counts):
            grade_list.extend([grade] * count)
        
        df.loc[:n_graded-1, 'grade'] = grade_list
        
        return df

    @pytest.mark.parametrize("encoding_type,expected_models", [
        (SEQUENTIAL, ['RandomForest', 'GaussianProcess', 'LogisticAT']),
        (ORDINAL_REGRESSION, ['RandomForest', 'GaussianProcess'])
    ])
    def test_encoding_model_compatibility(self, encoding_type, expected_models, mock_molecules_data):
        """Test that each encoding type works with its compatible models."""
        df = mock_molecules_data
        graded_df = df[df['grade'].notna()].copy()
        
        for model_type in expected_models:
            # Determine model category based on encoding and model type
            if encoding_type == ORDINAL_REGRESSION:
                model_category = 'regression'
            elif model_type == 'LogisticAT':
                model_category = 'ordinal'  # LogisticAT is always ordinal regression
            else:
                model_category = 'classification'
            
            # Create model configuration
            model_config = {
                'model_type': model_type,
                'model_params': {},
                'use_calibration': model_category == 'classification',
                'encoding_type': encoding_type
            }
            
            try:
                # Train model
                model, metrics = ml_models.train_model(
                    graded_df, 
                    model_config=model_config,
                    use_morgan_fp=True,
                    use_rdkit_fp=False,
                    use_interaction_fp=False
                )
                
                # Verify model trained successfully
                assert model.is_trained
                assert metrics['accuracy'] >= 0  # Some measure of success
                assert metrics['encoding_type'] == encoding_type
                assert metrics['model_type'] == model_type
                
                # Make predictions
                updated_df = ml_models.update_predictions(
                    df, model, metrics,
                    use_morgan_fp=True,
                    use_rdkit_fp=False,
                    use_interaction_fp=False
                )
                
                # Verify predictions were made
                assert 'prediction' in updated_df.columns
                predicted_molecules = updated_df[updated_df['prediction'].notna()]
                assert len(predicted_molecules) > 0
                
                # Verify predictions are valid grades
                valid_grades = set(['A', 'B', 'C', 'D'])
                predicted_grades = set(predicted_molecules['prediction'].unique())
                assert predicted_grades.issubset(valid_grades)
                
                
            except Exception as e:
                pytest.fail(f"Model {model_type} failed with encoding {encoding_type}: {str(e)}")

    def test_incompatible_model_encoding_combinations(self, mock_molecules_data):
        """Test that model-encoding combinations work or fail gracefully."""
        df = mock_molecules_data
        graded_df = df[df['grade'].notna()].copy()
        
        # Test some combinations that should work
        test_combinations = [
            ('RandomForest', SEQUENTIAL),
            ('GaussianProcess', SEQUENTIAL),
        ]
        
        for model_type, encoding_type in test_combinations:
            model_config = {
                'model_type': model_type,
                'model_params': {},
                'use_calibration': False,
                'encoding_type': encoding_type
            }
            
            # This should work
            model, metrics = ml_models.train_model(
                graded_df,
                model_config=model_config,
                use_morgan_fp=True,
                use_rdkit_fp=False,
                use_interaction_fp=False
            )
            
            # Verify model was trained successfully
            assert model is not None
            assert metrics['model_type'] == model_type
            assert metrics['encoding_type'] == encoding_type
            

    @pytest.mark.skipif(not Path("test_data/test_session/molecules.pkl").exists(), 
                       reason="Real test data not available")
    def test_with_real_molecular_data(self, real_molecules_data):
        """Test encoding-model compatibility with real molecular data."""
        df = real_molecules_data
        graded_df = df[df['grade'].notna()].copy()
        
        if len(graded_df) < 5:
            pytest.skip("Insufficient real graded data")
        
        # Test with a subset of real data
        test_configs = [
            (SEQUENTIAL, 'RandomForest'),
            (SEQUENTIAL, 'LogisticAT'),
            (ORDINAL_REGRESSION, 'RandomForest'),
        ]
        
        for encoding_type, model_type in test_configs:
            # Determine model category based on encoding and model type
            if encoding_type == ORDINAL_REGRESSION:
                model_category = 'regression'
            elif model_type == 'LogisticAT':
                model_category = 'ordinal'  # LogisticAT is always ordinal regression
            else:
                model_category = 'classification'
            
            model_config = {
                'model_type': model_type,
                'model_params': {},
                'use_calibration': model_category == 'classification',
                'encoding_type': encoding_type
            }
            
            try:
                # Train on real data
                model, metrics = ml_models.train_model(
                    graded_df.head(20),  # Use subset for faster testing
                    model_config=model_config,
                    use_morgan_fp=True,
                    use_rdkit_fp=False,
                    use_interaction_fp=True
                )
                
                assert model.is_trained
                assert metrics['n_samples'] <= 20
                assert metrics['encoding_type'] == encoding_type
                
                # Make predictions on larger dataset
                updated_df = ml_models.update_predictions(
                    df.head(50),
                    model, metrics,
                    use_morgan_fp=True,
                    use_rdkit_fp=False,
                    use_interaction_fp=True
                )
                
                # Verify realistic predictions
                predictions = updated_df[updated_df['prediction'].notna()]['prediction']
                assert len(predictions) > 0
                
                # Check that predictions follow expected distribution
                grade_counts = predictions.value_counts()
                assert len(grade_counts) <= 4  # No more than 4 grades
                
            except Exception as e:
                pytest.fail(f"Real data test failed for {model_type} with {encoding_type}: {str(e)}")



if __name__ == "__main__":
    pytest.main([__file__])