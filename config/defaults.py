"""
Default configuration values for PickM8.

This module provides default configuration values that serve as fallbacks
when config.yaml is missing or incomplete.
"""

DEFAULT_CONFIG = {
    # Machine Learning Models Configuration
    "ml_models": {
        "default_type": "RandomForest",
        
        # Feature Engineering Settings
        "feature_engineering": {
            "enabled": True,
            "variance_threshold": 0.01,
            "use_importance_selection": True,
            "use_dimensionality_reduction": False,
            "normalize": True,
        },
        
        # Model-specific parameters
        "RandomForest": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        
        "GaussianProcess": {
            "kernel": "Matern",
            "optimize_kernel": True,
            "n_restarts_optimizer": 10,
            "random_state": 42,
        },
        
        "LogisticAT": {
            "alpha": 1.0,
            "max_iter": 1000,
            "random_state": 42,
        },
    },
    
    # Encoding Configuration
    "encoding": {
        "type": "sequential",
        "default_grades": ["A", "B", "C", "D"],
        
        # Encoding-specific settings
        "ordinal_regression": {
            "grade_ranges": {
                "D": [0, 25],
                "C": [25, 50],
                "B": [50, 75],
                "A": [75, 100],
            }
        },
    },
    
    # Interaction Analysis Configuration
    "interactions": {
        "interaction_type": "plip",
        "ligand_name": "LIG",
        "max_workers": 28,
        "plip_config": {
            "hydrogen_bonds": True,
            "hydrophobic_contacts": True,
            "pi_stacking": True,
            "salt_bridges": True,
            "halogen_bonds": True,
        },
        "prolif_config": {
            "interactions": ["Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking", "Anionic", "Cationic"]
        },
    },
    
    # Pose Quality Configuration
    "pose_quality": {
        "enabled": True,
        "calculate_clashes": True,
        "calculate_strain": True,
        "max_workers": 28,
    },
    
    # Processing Configuration
    "processing": {
        "fingerprint_config": {
            "compute_morgan": True,
            "compute_rdkit": False,
            "compute_mapchiral": False,
            "morgan_radius": 2,
            "morgan_bits": 2048,
            "rdkit_bits": 2048,
            "mapchiral_max_radius": 2,
            "mapchiral_n_permutations": 2048,
        },
        
        # GRADE Descriptor Configuration
        "grade_config": {
            "enabled": False,
            "extended": False,
            "normalize_charges": False,
            "max_workers": 8,
        },
    },
}