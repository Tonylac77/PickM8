"""
Label encoding utilities for active learning.
"""

import numpy as np
from typing import List, Tuple, Dict


def encode_grades_for_training(grades: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encode grade labels for ML training.
    
    Args:
        grades: List of grade strings ('A', 'B', 'C', 'D', 'F')
        
    Returns:
        Tuple of (encoded_labels, label_to_int, int_to_label)
    """
    # Define grade hierarchy (A=4, B=3, C=2, D=1, F=0)
    grade_to_int = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    int_to_grade = {v: k for k, v in grade_to_int.items()}
    
    # Handle any grades not in the standard set
    unique_grades = set(grades)
    for grade in unique_grades:
        if grade not in grade_to_int:
            # Assign to middle grade for unknown grades
            grade_to_int[grade] = 2
    
    encoded = np.array([grade_to_int.get(grade, 2) for grade in grades])
    
    return encoded, grade_to_int, int_to_grade