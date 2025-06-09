from rdkit import Chem
import numpy as np
import json

def mol_from_string(mol_string, format='mol'):
    if format == 'mol':
        return Chem.MolFromMolBlock(mol_string)
    elif format == 'smiles':
        return Chem.MolFromSmiles(mol_string)
    else:
        raise ValueError(f"Unknown format: {format}")

def prepare_molecule_batch(molecules, batch_size=50):
    batches = []
    for i in range(0, len(molecules), batch_size):
        batch = molecules[i:i+batch_size]
        batches.append(batch)
    return batches

def filter_molecules_by_grade_status(molecules_df, grades_df, mode='annotate'):
    if grades_df is None or grades_df.is_empty():
        return molecules_df
    
    graded_mol_ids = grades_df['mol_id'].unique().to_list()
    
    if mode == 'annotate':
        return molecules_df.filter(~molecules_df['id'].is_in(graded_mol_ids))
    else:
        return molecules_df

def sort_molecules(molecules_df, predictions_df=None, method='score'):
    if method == 'score':
        return molecules_df.sort('score')
    
    elif method in ['uncertainty', 'prediction'] and predictions_df is not None:
        merged = molecules_df.join(predictions_df, left_on='id', right_on='mol_id', how='left')
        
        if method == 'uncertainty':
            return merged.sort('uncertainty', descending=True)
        else:
            return merged.sort(['prediction', 'uncertainty'])
    
    return molecules_df

def calculate_interaction_summary(interactions):
    summary = {}
    for interaction in interactions.interactions:
        inter_type = interaction.type
        if inter_type not in summary:
            summary[inter_type] = 0
        summary[inter_type] += 1
    return summary