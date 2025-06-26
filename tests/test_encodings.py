"""Tests for grade encoding/decoding functionality."""
import pytest
import numpy as np
import pandas as pd
from data.encodings import (
    encode_sequential, encode_nominal, encode_ordinal,
    decode_sequential, decode_nominal, decode_ordinal,
    get_encoding_function, get_decoding_function,
    get_ml_strategy, calculate_uncertainty_score,
    get_active_learning_ranking, encode_grades_for_training,
    decode_predictions, SEQUENTIAL, NOMINAL, ORDINAL
)


class TestEncodingFunctions:
    """Test encoding functions for all three types."""
    
    def test_encode_sequential(self):
        """Test sequential encoding functionality."""
        grades = ['A', 'B', 'C', 'A', 'B']
        encoded, mapping = encode_sequential(grades)
        
        # Check output shape and type
        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (5,)
        assert isinstance(mapping, dict)
        
        # Check mapping correctness (alphabetical order)
        expected_mapping = {'A': 0, 'B': 1, 'C': 2}
        assert mapping == expected_mapping
        
        # Check encoded values
        expected_encoded = np.array([0, 1, 2, 0, 1])
        np.testing.assert_array_equal(encoded, expected_encoded)
    
    def test_encode_nominal(self):
        """Test nominal (one-hot) encoding functionality."""
        grades = ['A', 'B', 'C']
        encoded, mapping = encode_nominal(grades)
        
        # Check output shape and type
        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (3, 3)
        assert isinstance(mapping, dict)
        
        # Check mapping correctness
        expected_mapping = {'A': 0, 'B': 1, 'C': 2}
        assert mapping == expected_mapping
        
        # Check one-hot encoding
        expected_encoded = np.array([
            [1, 0, 0],  # A
            [0, 1, 0],  # B
            [0, 0, 1]   # C
        ])
        np.testing.assert_array_equal(encoded, expected_encoded)
    
    def test_encode_ordinal(self):
        """Test ordinal (cumulative) encoding functionality."""
        grades = ['A', 'B', 'C']
        encoded, mapping = encode_ordinal(grades)
        
        # Check output shape and type
        assert isinstance(encoded, np.ndarray)
        assert encoded.shape == (3, 3)
        assert isinstance(mapping, dict)
        
        # Check mapping correctness
        expected_mapping = {'A': 0, 'B': 1, 'C': 2}
        assert mapping == expected_mapping
        
        # Check ordinal encoding (cumulative)
        expected_encoded = np.array([
            [1, 0, 0],  # A
            [1, 1, 0],  # B
            [1, 1, 1]   # C
        ])
        np.testing.assert_array_equal(encoded, expected_encoded)
    
    def test_encode_with_different_grades(self):
        """Test encoding with different grade sets."""
        grades = ['F', 'A', 'D', 'B']
        
        # Test sequential
        encoded_seq, mapping_seq = encode_sequential(grades)
        expected_mapping = {'A': 0, 'B': 1, 'D': 2, 'F': 3}  # Alphabetical order
        assert mapping_seq == expected_mapping
        
        # Test nominal
        encoded_nom, mapping_nom = encode_nominal(grades)
        assert encoded_nom.shape == (4, 4)
        
        # Test ordinal
        encoded_ord, mapping_ord = encode_ordinal(grades)
        assert encoded_ord.shape == (4, 4)


class TestDecodingFunctions:
    """Test decoding functions for all three types."""
    
    def test_decode_sequential(self):
        """Test sequential decoding functionality."""
        predictions = np.array([0.2, 1.8, 2.1, 0.9])
        label_mapping = {'A': 0, 'B': 1, 'C': 2}
        
        decoded = decode_sequential(predictions, label_mapping)
        expected = ['A', 'C', 'C', 'B']  # Rounded to nearest int: 0.2->0, 1.8->2, 2.1->2, 0.9->1
        assert decoded == expected
    
    def test_decode_nominal(self):
        """Test nominal decoding functionality."""
        predictions = np.array([
            [0.8, 0.1, 0.1],  # Mostly A
            [0.2, 0.7, 0.1],  # Mostly B
            [0.1, 0.2, 0.7]   # Mostly C
        ])
        label_mapping = {'A': 0, 'B': 1, 'C': 2}
        
        decoded = decode_nominal(predictions, label_mapping)
        expected = ['A', 'B', 'C']
        assert decoded == expected
    
    def test_decode_ordinal(self):
        """Test ordinal decoding functionality."""
        predictions = np.array([
            [0.9, 0.1, 0.1],  # A: only first is >0.5
            [0.8, 0.7, 0.2],  # B: first two are >0.5
            [0.9, 0.8, 0.6]   # C: all three are >0.5
        ])
        label_mapping = {'A': 0, 'B': 1, 'C': 2}
        
        decoded = decode_ordinal(predictions, label_mapping)
        expected = ['A', 'B', 'C']
        assert decoded == expected
    
    def test_decode_edge_cases(self):
        """Test decoding edge cases."""
        label_mapping = {'A': 0, 'B': 1}
        
        # Empty predictions
        empty_predictions = np.array([])
        assert decode_sequential(empty_predictions, label_mapping) == []
        
        # Single prediction
        single_pred = np.array([0.7])
        assert decode_sequential(single_pred, label_mapping) == ['B']


class TestFactoryFunctions:
    """Test factory functions for encoding/decoding selection."""
    
    def test_get_encoding_function(self):
        """Test encoding function factory."""
        assert get_encoding_function(SEQUENTIAL) == encode_sequential
        assert get_encoding_function(NOMINAL) == encode_nominal
        assert get_encoding_function(ORDINAL) == encode_ordinal
        
        with pytest.raises(ValueError):
            get_encoding_function("invalid_type")
    
    def test_get_decoding_function(self):
        """Test decoding function factory."""
        assert get_decoding_function(SEQUENTIAL) == decode_sequential
        assert get_decoding_function(NOMINAL) == decode_nominal
        assert get_decoding_function(ORDINAL) == decode_ordinal
        
        with pytest.raises(ValueError):
            get_decoding_function("invalid_type")
    
    def test_get_ml_strategy(self):
        """Test ML strategy recommendation."""
        assert get_ml_strategy(SEQUENTIAL) == "regression"
        assert get_ml_strategy(NOMINAL) == "multiclass"
        assert get_ml_strategy(ORDINAL) == "ordinal"
        
        with pytest.raises(ValueError):
            get_ml_strategy("invalid_type")


class TestUncertaintyCalculation:
    """Test uncertainty calculation for different encoding types."""
    
    def test_sequential_uncertainty(self):
        """Test uncertainty calculation for sequential encoding."""
        predictions = np.array([0.2, 1.8, 2.0, 1.0])
        uncertainty = calculate_uncertainty_score(predictions, SEQUENTIAL)
        
        # Should be distance from nearest integer
        expected = np.array([0.2, 0.2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(uncertainty, expected)
    
    def test_nominal_uncertainty(self):
        """Test uncertainty calculation for nominal encoding."""
        predictions = np.array([
            [0.9, 0.05, 0.05],  # Low uncertainty (confident)
            [0.4, 0.3, 0.3],    # High uncertainty (uncertain)
        ])
        uncertainty = calculate_uncertainty_score(predictions, NOMINAL)
        
        # First should have lower uncertainty than second
        assert uncertainty[0] < uncertainty[1]
        assert len(uncertainty) == 2
    
    def test_ordinal_uncertainty(self):
        """Test uncertainty calculation for ordinal encoding."""
        predictions = np.array([
            [0.9, 0.9, 0.9],    # Low variance (confident)
            [0.9, 0.5, 0.1],    # High variance (uncertain)
        ])
        uncertainty = calculate_uncertainty_score(predictions, ORDINAL)
        
        # First should have lower uncertainty than second
        assert uncertainty[0] < uncertainty[1]
        assert len(uncertainty) == 2


class TestActiveLearningRanking:
    """Test active learning ranking for different encoding types."""
    
    def test_uncertainty_ranking(self):
        """Test uncertainty-based ranking."""
        predictions = np.array([0.2, 1.8, 2.0, 1.0])  # Sequential predictions
        ranking = get_active_learning_ranking(predictions, SEQUENTIAL, "uncertainty")
        
        # Should rank by uncertainty descending (most uncertain first)
        assert len(ranking) == 4
        assert isinstance(ranking, np.ndarray)
    
    def test_best_predictions_ranking_sequential(self):
        """Test best predictions ranking for sequential encoding."""
        predictions = np.array([2.0, 0.5, 1.5, 0.1])  # A=0 is best
        ranking = get_active_learning_ranking(predictions, SEQUENTIAL, "best_predictions")
        
        # Should be sorted by prediction value ascending (best grades first)
        sorted_predictions = predictions[ranking]
        assert sorted_predictions[0] == 0.1  # Best prediction first
        assert sorted_predictions[-1] == 2.0  # Worst prediction last
    
    def test_best_predictions_ranking_nominal(self):
        """Test best predictions ranking for nominal encoding."""
        predictions = np.array([
            [0.9, 0.1, 0.0],    # High confidence
            [0.6, 0.3, 0.1],    # Medium confidence
            [0.4, 0.4, 0.2]     # Low confidence
        ])
        ranking = get_active_learning_ranking(predictions, NOMINAL, "best_predictions")
        
        # Should rank by max probability descending (most confident first)
        max_probs = np.max(predictions, axis=1)
        sorted_max_probs = max_probs[ranking]
        assert sorted_max_probs[0] >= sorted_max_probs[1] >= sorted_max_probs[2]
    
    def test_invalid_strategy(self):
        """Test invalid strategy handling."""
        predictions = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            get_active_learning_ranking(predictions, SEQUENTIAL, "invalid_strategy")


class TestMainFunctions:
    """Test main encoding/decoding functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'grade': ['A', 'B', 'C', 'A'],
            'name': ['mol1', 'mol2', 'mol3', 'mol4']
        })
    
    def test_encode_grades_for_training(self):
        """Test main encoding function."""
        # Test sequential encoding
        encoded, mapping = encode_grades_for_training(self.df, SEQUENTIAL)
        assert len(encoded) == 4
        assert 'A' in mapping and 'B' in mapping and 'C' in mapping
        
        # Test nominal encoding
        encoded_nom, mapping_nom = encode_grades_for_training(self.df, NOMINAL)
        assert encoded_nom.shape == (4, 3)
        
        # Test ordinal encoding
        encoded_ord, mapping_ord = encode_grades_for_training(self.df, ORDINAL)
        assert encoded_ord.shape == (4, 3)
    
    def test_decode_predictions(self):
        """Test main decoding function."""
        label_mapping = {'A': 0, 'B': 1, 'C': 2}
        
        # Test sequential decoding
        predictions_seq = np.array([0.1, 1.8, 2.2, 0.9])
        decoded = decode_predictions(predictions_seq, label_mapping, SEQUENTIAL)
        assert len(decoded) == 4
        assert all(grade in ['A', 'B', 'C'] for grade in decoded)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame({'grade': []})  # Empty but with grade column
        encoded, mapping = encode_grades_for_training(empty_df, SEQUENTIAL)
        
        assert len(encoded) == 0
        assert mapping == {}
    
    def test_no_grades_dataframe(self):
        """Test handling of DataFrames with no grades."""
        no_grades_df = pd.DataFrame({
            'id': [1, 2, 3],
            'grade': [None, None, None]
        })
        encoded, mapping = encode_grades_for_training(no_grades_df, SEQUENTIAL)
        
        assert len(encoded) == 0
        assert mapping == {}


class TestIntegration:
    """Integration tests for encoding/decoding workflow."""
    
    def test_encode_decode_roundtrip_sequential(self):
        """Test encode-decode roundtrip for sequential encoding."""
        grades = ['A', 'B', 'C', 'D', 'F']
        
        # Encode
        encoded, mapping = encode_sequential(grades)
        
        # Simulate predictions (exact integers)
        predictions = encoded.astype(float)
        
        # Decode
        decoded = decode_sequential(predictions, mapping)
        
        # Should match original grades
        assert decoded == grades
    
    def test_encode_decode_roundtrip_nominal(self):
        """Test encode-decode roundtrip for nominal encoding."""
        grades = ['A', 'B', 'C']
        
        # Encode
        encoded, mapping = encode_nominal(grades)
        
        # Simulate perfect predictions
        predictions = encoded
        
        # Decode
        decoded = decode_nominal(predictions, mapping)
        
        # Should match original grades
        assert decoded == grades
    
    def test_encode_decode_roundtrip_ordinal(self):
        """Test encode-decode roundtrip for ordinal encoding."""
        grades = ['A', 'B', 'C']
        
        # Encode
        encoded, mapping = encode_ordinal(grades)
        
        # Simulate perfect predictions
        predictions = encoded
        
        # Decode
        decoded = decode_ordinal(predictions, mapping)
        
        # Should match original grades
        assert decoded == grades
    
    def test_all_encoding_types_with_same_data(self):
        """Test all encoding types with the same input data."""
        grades = ['A', 'B', 'C', 'A', 'B']
        
        # Test all encoding types
        for encoding_type in [SEQUENTIAL, NOMINAL, ORDINAL]:
            encoding_func = get_encoding_function(encoding_type)
            decoding_func = get_decoding_function(encoding_type)
            
            # Encode
            encoded, mapping = encoding_func(grades)
            
            # Should have consistent mapping
            assert isinstance(mapping, dict)
            assert len(mapping) == 3  # A, B, C
            
            # Should be able to get ML strategy
            strategy = get_ml_strategy(encoding_type)
            assert strategy in ["regression", "multiclass", "ordinal"]


if __name__ == "__main__":
    pytest.main([__file__])