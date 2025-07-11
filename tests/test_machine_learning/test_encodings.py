"""
Tests for grade encoding and decoding functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json

from machine_learning.encoding.encodings import (
    encode_sequential, encode_one_hot, encode_ordinal, encode_ordinal_regression,
    decode_sequential, decode_one_hot, decode_ordinal, decode_ordinal_regression,
    SEQUENTIAL, ONE_HOT, ORDINAL, ORDINAL_REGRESSION
)


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
        assert mapping == {'A': 2, 'B': 1, 'C': 0}  # Reversed mapping
        
        # Simulate perfect predictions
        predictions = encoded.astype(float)
        
        # Decode
        decoded = decode_sequential(predictions, mapping)
        
        # Should match original
        assert decoded == grades

    def test_one_hot_encoding_roundtrip(self):
        """Test one-hot encoding-decoding roundtrip."""
        grades = ['A', 'B', 'C']
        
        # Encode
        encoded, mapping = encode_one_hot(grades)
        
        # Verify encoding structure
        assert encoded.shape == (3, 3)
        assert mapping == {'A': 0, 'B': 1, 'C': 2}
        
        # Verify one-hot structure
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(encoded, expected)
        
        # Simulate perfect predictions
        predictions = encoded.astype(float)
        
        # Decode
        decoded = decode_one_hot(predictions, mapping)
        
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

    def test_encoding_with_different_grades(self):
        """Test encoding works with different grade sets."""
        grades = ['Excellent', 'Good', 'Poor']
        
        for encoding_func in [encode_sequential, encode_one_hot, encode_ordinal]:
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
        
        encoded_nom, mapping_nom = encode_one_hot(empty_grades)
        assert encoded_nom.shape[0] == 0
        assert mapping_nom == {}

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


if __name__ == '__main__':
    pytest.main([__file__])