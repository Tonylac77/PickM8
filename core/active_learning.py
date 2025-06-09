import numpy as np
import polars as pl
from core.fingerprints import FingerprintHandler
from core.models.base import ModelLoader
import json

class ActiveLearningStrategy:
    def __init__(self, model_type='ensemble', **model_kwargs):
        self.model = ModelLoader.load_builtin_model(model_type, **model_kwargs)
        self.fp_handler = FingerprintHandler()
        self.is_trained = False
    
    def prepare_features(self, molecules_df):
        features = []
        
        for row in molecules_df.iter_rows(named=True):
            ifp_dict = json.loads(row['ifp'])
            ifp_array = self.fp_handler.ifp_to_array(ifp_dict)
            
            if 'morgan_fp' in row and row['morgan_fp']:
                morgan_fp = np.array(row['morgan_fp'])
                combined_features = self.fp_handler.combine_fingerprints(morgan_fp, ifp_array)
            else:
                combined_features = ifp_array
            
            features.append(combined_features)
        
        return np.array(features)
    
    def train(self, molecules_df, grades_df):
        merged = molecules_df.join(
            grades_df.select(['mol_id', 'grade']), 
            left_on='id', 
            right_on='mol_id'
        )
        
        X = self.prepare_features(merged)
        y = merged['grade'].to_numpy()
        
        self.model.train(X, y)
        self.is_trained = True
        
        return {
            'n_samples': len(X),
            'unique_grades': np.unique(y).tolist(),
            'grade_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }
    
    def predict(self, molecules_df):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(molecules_df)
        predictions = self.model.predict(X)
        uncertainties = self.model.get_uncertainty(X)
        
        predictions_data = []
        for i, row in enumerate(molecules_df.iter_rows(named=True)):
            predictions_data.append({
                'mol_id': row['id'],
                'prediction': predictions[i],
                'uncertainty': uncertainties[i]
            })
        
        return pl.DataFrame(predictions_data)
    
    def get_next_molecules(self, molecules_df, grades_df, n_molecules=10, strategy='uncertainty'):
        ungraded_ids = set(molecules_df['id'].to_list())
        if grades_df is not None:
            graded_ids = set(grades_df['mol_id'].to_list())
            ungraded_ids = ungraded_ids - graded_ids
        
        ungraded_df = molecules_df.filter(pl.col('id').is_in(list(ungraded_ids)))
        
        if not self.is_trained or strategy == 'random':
            return ungraded_df.sample(min(n_molecules, len(ungraded_df)))
        
        predictions_df = self.predict(ungraded_df)
        
        if strategy == 'uncertainty':
            sorted_df = predictions_df.sort('uncertainty', descending=True)
        elif strategy == 'score':
            return ungraded_df.sort('score').head(n_molecules)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        top_mol_ids = sorted_df.head(n_molecules)['mol_id'].to_list()
        return molecules_df.filter(pl.col('id').is_in(top_mol_ids))