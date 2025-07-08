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

from active_learning import ml_models
from active_learning.encodings import (
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

    def test_sequential_encoding_reversed_mapping(self):
        """Test that sequential encoding uses reversed mapping (A=3, B=2, C=1, D=0)."""
        grades = ['A', 'B', 'C', 'D']
        
        encoded, mapping = encode_sequential(grades)
        
        # Verify reversed mapping
        assert mapping == {'A': 3, 'B': 2, 'C': 1, 'D': 0}
        
        # Verify encoding values
        expected_encoded = np.array([3, 2, 1, 0])
        np.testing.assert_array_equal(encoded, expected_encoded)
        
        # Test roundtrip
        decoded = decode_sequential(encoded.astype(float), mapping)
        assert decoded == grades

    def test_ordinal_regression_encoding_continuous(self):
        """Test ordinal regression encoding produces continuous values."""
        grades = ['A', 'B', 'C', 'D']
        
        encoded, mapping = encode_ordinal_regression(grades)
        
        # Verify continuous values (midpoints of ranges)
        expected_mapping = {'D': 12.5, 'C': 37.5, 'B': 62.5, 'A': 87.5}
        assert mapping == expected_mapping
        
        # Verify encoding values
        expected_encoded = np.array([87.5, 62.5, 37.5, 12.5])
        np.testing.assert_array_equal(encoded, expected_encoded)
        
        # Test roundtrip with binning
        decoded = decode_ordinal_regression(encoded, mapping)
        assert decoded == grades
        
        # Test edge cases for binning
        edge_values = np.array([24.9, 25.1, 49.9, 50.1, 74.9, 75.1])
        edge_decoded = decode_ordinal_regression(edge_values, mapping)
        expected_edge = ['D', 'C', 'C', 'B', 'B', 'A']
        assert edge_decoded == expected_edge

    def test_all_encodings_produce_correct_formats(self):
        """Test all encoding types produce expected output formats."""
        grades = ['A', 'B', 'C', 'D']
        
        # Sequential: 1D array of integers
        seq_encoded, seq_mapping = encode_sequential(grades)
        assert seq_encoded.ndim == 1
        assert seq_encoded.dtype in [np.int64, np.int32]
        
        # One-hot: 2D array of binary values
        oh_encoded, oh_mapping = encode_one_hot(grades)
        assert oh_encoded.ndim == 2
        assert oh_encoded.shape == (4, 4)
        assert np.all((oh_encoded == 0) | (oh_encoded == 1))
        
        # Ordinal: 2D array of cumulative binary values
        ord_encoded, ord_mapping = encode_ordinal(grades)
        assert ord_encoded.ndim == 2
        assert ord_encoded.shape == (4, 4)
        assert np.all((ord_encoded == 0) | (ord_encoded == 1))
        
        # Ordinal regression: 1D array of continuous values
        ord_reg_encoded, ord_reg_mapping = encode_ordinal_regression(grades)
        assert ord_reg_encoded.ndim == 1
        assert ord_reg_encoded.dtype in [np.float64, np.float32]
        assert np.all((ord_reg_encoded >= 0) & (ord_reg_encoded <= 100))

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
                
                # Verify uncertainty estimates
                assert 'prediction_uncertainty' in updated_df.columns
                uncertainties = updated_df[updated_df['prediction_uncertainty'].notna()]['prediction_uncertainty']
                assert len(uncertainties) > 0
                assert np.all(uncertainties >= 0)
                
            except Exception as e:
                pytest.fail(f"Model {model_type} failed with encoding {encoding_type}: {str(e)}")

    def test_model_category_determination(self):
        """Test that model categories are correctly determined from encoding types."""
        from active_learning.ml_models import determine_model_category
        
        assert determine_model_category(SEQUENTIAL) == 'classification'
        assert determine_model_category(ONE_HOT) == 'classification'
        assert determine_model_category(ORDINAL) == 'ordinal'
        assert determine_model_category(ORDINAL_REGRESSION) == 'regression'

    def test_incompatible_model_encoding_combinations(self, mock_molecules_data):
        """Test that incompatible model-encoding combinations are handled properly."""
        df = mock_molecules_data
        graded_df = df[df['grade'].notna()].copy()
        
        # These combinations should not work or should fall back gracefully
        incompatible_combinations = [
            # ONE_HOT encoding should not work with sklearn models (produces 2D arrays)
            ('RandomForest', ONE_HOT),
            ('GaussianProcess', ONE_HOT),
        ]
        
        for model_type, encoding_type in incompatible_combinations:
            model_config = {
                'model_type': model_type,
                'model_params': {},
                'use_calibration': True,  # This will trigger the error with ONE_HOT
                'encoding_type': encoding_type
            }
            
            # This should fail due to incompatible format
            with pytest.raises((ValueError, ImportError)) as exc_info:
                model, metrics = ml_models.train_model(
                    graded_df,
                    model_config=model_config,
                    use_morgan_fp=True,
                    use_rdkit_fp=False,
                    use_interaction_fp=False
                )
            
            # Should fail due to 2D array format
            assert "1d array" in str(exc_info.value) or "shape" in str(exc_info.value)

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

    def test_uncertainty_estimation_by_encoding(self, mock_molecules_data):
        """Test that uncertainty estimation works correctly for each encoding type."""
        df = mock_molecules_data
        graded_df = df[df['grade'].notna()].copy()
        
        encoding_model_pairs = [
            (SEQUENTIAL, 'RandomForest'),
            (SEQUENTIAL, 'LogisticAT'),
            (ORDINAL_REGRESSION, 'RandomForest'),
        ]
        
        for encoding_type, model_type in encoding_model_pairs:
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
            
            model, metrics = ml_models.train_model(
                graded_df,
                model_config=model_config,
                use_morgan_fp=True,
                use_rdkit_fp=False,
                use_interaction_fp=False
            )
            
            # Test uncertainty on ungraded molecules
            ungraded_df = df[df['grade'].isna()].copy()
            if len(ungraded_df) > 0:
                updated_df = ml_models.update_predictions(
                    ungraded_df.head(10),
                    model, metrics,
                    use_morgan_fp=True,
                    use_rdkit_fp=False,
                    use_interaction_fp=False
                )
                
                uncertainties = updated_df['prediction_uncertainty'].dropna()
                assert len(uncertainties) > 0
                assert np.all(uncertainties >= 0)
                
                # Verify uncertainty values make sense for model type
                if encoding_type == ORDINAL_REGRESSION:
                    # Regression uncertainties can be much higher (measured as standard deviation)
                    assert np.mean(uncertainties) >= 0
                    assert np.all(uncertainties <= 2000)  # Reasonable upper bound for regression
                elif model_type == 'LogisticAT':
                    # Ordinal regression uncertainties (entropy-based)
                    assert np.mean(uncertainties) >= 0
                    assert np.all(uncertainties <= 2.5)  # Reasonable upper bound for ordinal entropy
                else:
                    # Classification uncertainties should be between 0 and 1 (entropy)
                    assert np.all(uncertainties <= 1.5)


class TestEncodingEdgeCases:
    """Test edge cases and error conditions for encodings."""
    
    def test_empty_grades(self):
        """Test encoding with empty grade lists."""
        empty_grades = []
        
        # All encodings should handle empty input gracefully
        for encode_func in [encode_sequential, encode_one_hot, encode_ordinal, encode_ordinal_regression]:
            encoded, mapping = encode_func(empty_grades)
            assert len(encoded) == 0
            assert mapping == {}
    
    def test_single_grade(self):
        """Test encoding with single grade."""
        single_grade = ['A']
        
        # Sequential
        seq_encoded, seq_mapping = encode_sequential(single_grade)
        assert len(seq_encoded) == 1
        assert seq_mapping == {'A': 0}
        
        # One-hot
        oh_encoded, oh_mapping = encode_one_hot(single_grade)
        assert oh_encoded.shape == (1, 1)
        assert oh_mapping == {'A': 0}
        
        # Ordinal regression
        ord_reg_encoded, ord_reg_mapping = encode_ordinal_regression(single_grade)
        assert len(ord_reg_encoded) == 1
        assert ord_reg_mapping == {'A': 87.5}
    
    def test_duplicate_grades(self):
        """Test encoding with duplicate grades."""
        duplicate_grades = ['A', 'A', 'B', 'B', 'A']
        
        for encode_func in [encode_sequential, encode_one_hot, encode_ordinal, encode_ordinal_regression]:
            encoded, mapping = encode_func(duplicate_grades)
            assert len(encoded) == 5  # Should preserve all input grades
            assert len(mapping) == 2  # Should have mapping for unique grades only


if __name__ == "__main__":
    pytest.main([__file__])