"""
Integration tests for the core data pipeline.
Tests the complete workflow from SDF loading to ML predictions.
"""

import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path

from data_io.molecules import *
from features.fingerprints.fingerprints import *
from features.interactions import *
from analysis.pose_quality import *
from analysis.grading import *
from machine_learning import ml_models
import sys
sys.path.insert(0, '/home/tony/PickM8')
from sessions import save_session, load_session, list_sessions


class TestCoreDataPipeline:
    """Test essential data processing pipeline with real data."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test data directory."""
        return Path(__file__).parent.parent.parent / "test_data"
    
    @pytest.fixture
    def protein_file(self, test_data_dir):
        """Path to test protein PDB file."""
        return test_data_dir / "1fvv_p.pdb"
    
    @pytest.fixture
    def ligand_file(self, test_data_dir):
        """Path to test ligand SDF file."""
        return test_data_dir / "1fvv_l.sdf"
    
    @pytest.fixture
    def poses_file(self, test_data_dir):
        """Path to test poses SDF file."""
        return test_data_dir / "example_poses_1fvv.sdf"
    
    @pytest.fixture
    def protein_content(self, protein_file):
        """Load protein content from PDB file."""
        if not protein_file.exists():
            pytest.skip("Test protein file not available")
        
        with open(protein_file, 'r') as f:
            return f.read()

    def test_complete_pipeline_single_molecule(self, ligand_file, protein_content):
        """Test complete pipeline with single molecule."""
        if not ligand_file.exists():
            pytest.skip("Test ligand file not available")
        
        # 1. Load SDF
        df = load_sdf(str(ligand_file))
        assert len(df) == 1
        
        # 2. Compute fingerprints
        mol = df['mol'].iloc[0]
        morgan_fp = compute_ecfp_fingerprint(mol)
        macc_fp = compute_maccs_fingerprint(mol)
        
        assert morgan_fp is not None
        assert macc_fp is not None
        
        # 3. Compute interactions
        config = {'interaction_type': 'plip', 'ligand_name': 'LIG'}
        try:
            ifp_json, interactions_json, num_interactions = compute_interaction_fingerprint(
                mol, protein_content, config
            )
            assert isinstance(num_interactions, int)
        except ImportError:
            pytest.skip("Interaction analysis tools not available")
        
        # 4. Compute pose quality
        mol_block = df['mol_block'].iloc[0]
        config = {'calculate_clashes': True, 'calculate_strain': True}
        result = analyze_single_pose(mol_block, protein_content, config)
        
        assert isinstance(result, dict)
        assert 'clashes' in result
        assert 'strain_energy' in result

    def test_complete_pipeline_multiple_molecules(self, poses_file, protein_content):
        """Test complete pipeline with multiple molecules."""
        if not poses_file.exists():
            pytest.skip("Test poses file not available")
        
        # 1. Load SDF
        df = load_sdf(str(poses_file))
        assert len(df) > 1
        
        # 2. Process first few molecules
        for i in range(min(3, len(df))):
            mol = df['mol'].iloc[i]
            
            # Compute fingerprints
            morgan_fp = compute_ecfp_fingerprint(mol)
            macc_fp = compute_maccs_fingerprint(mol)
            
            assert morgan_fp is not None
            assert rdkit_fp is not None
            
            # Compute pose quality
            mol_block = df['mol_block'].iloc[i]
            config = {'calculate_clashes': True, 'calculate_strain': True}
            result = analyze_single_pose(mol_block, protein_content, config)
            
            assert isinstance(result, dict)
            assert result['clashes'] >= 0
            assert result['strain_energy'] >= 0.0

    def test_ml_workflow_integration(self):
        """Test integration of ML workflow with mock data."""
        # Create mock molecular data
        np.random.seed(42)
        n_molecules = 15
        
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
        
        # Add grades to first 10 molecules
        grades = ['A', 'B', 'C'] * 3 + ['A']
        df.loc[:9, 'grade'] = grades
        
        # Train ML model
        model, metrics = ml_models.train_model(df)
        
        assert model is not None
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        
        # Generate predictions
        updated_df = ml_models.update_predictions(df, model, metrics)
        
        # Verify predictions
        predictions = updated_df['prediction'].dropna()
        assert len(predictions) > 0
        
        # Test grading workflow
        stats = get_grading_statistics(updated_df)
        assert stats['total_molecules'] == n_molecules
        assert stats['graded_count'] == 10
        
        # Test sorting strategies
        sorted_by_score = get_molecules_by_strategy(updated_df, 'Best Score')
        assert len(sorted_by_score) > 0
        

    def test_session_workflow_integration(self):
        """Test session save/load workflow integration."""
        # Create test data
        np.random.seed(42)
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['mol_1', 'mol_2', 'mol_3', 'mol_4', 'mol_5'],
            'smiles': ['CCO', 'CC', 'CCC', 'CCCC', 'CCCCC'],
            'score': [-2.5, -3.1, -1.8, -4.2, -2.9],
            'morgan_fp': [np.random.randint(0, 2, 1024).tolist() for _ in range(5)],
            'rdkit_fp': [np.random.randint(0, 2, 2048).tolist() for _ in range(5)],
            'grade': [None, 'A', None, 'B', None],
        }
        
        df = pd.DataFrame(data)
        metadata = {"test": "integration"}
        
        # Save session
        session_id = "test_integration_session"
        success = save_session(session_id, df, metadata)
        assert success is True
        
        # Load session
        result = load_session(session_id)
        assert result is not None
        
        loaded_df, loaded_metadata = result
        assert len(loaded_df) == len(df)
        assert loaded_metadata == metadata
        
        # Test grading on loaded data
        updated_df = add_grade(loaded_df, 1, 'A')
        assert updated_df.loc[updated_df['id'] == 1, 'grade'].iloc[0] == 'A'
        
        # Cleanup
        import shutil
        from pathlib import Path
        session_path = Path("data/sessions") / session_id
        if session_path.exists():
            shutil.rmtree(session_path)


if __name__ == '__main__':
    pytest.main([__file__])