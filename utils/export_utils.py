"""
Export utilities using functional programming approach.
Handles exporting molecular data to various formats (SDF, CSV, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import json
from pathlib import Path
from io import StringIO

from rdkit import Chem
from rdkit.Chem import PandasTools
import csv

logger = logging.getLogger(__name__)


def prepare_dataframe_for_export(df: pd.DataFrame, include_predictions: bool = True,
                                include_fingerprints: bool = False,
                                include_pose_metrics: bool = True) -> pd.DataFrame:
    """
    Prepare DataFrame for export by cleaning and selecting columns.
    
    Args:
        df: Molecules DataFrame
        include_predictions: Whether to include ML predictions
        include_fingerprints: Whether to include fingerprint data
        include_pose_metrics: Whether to include pose quality metrics
        
    Returns:
        Cleaned DataFrame ready for export
    """
    export_df = df.copy()
    
    # Core columns to always include
    core_columns = ['id', 'name', 'smiles', 'score']
    
    # Add grade information
    grade_columns = []
    if 'grade' in export_df.columns:
        grade_columns.extend(['grade', 'grade_timestamp'])
    
    # Add prediction information
    prediction_columns = []
    if include_predictions and 'prediction' in export_df.columns:
        prediction_columns.extend(['prediction', 'prediction_uncertainty', 'prediction_timestamp'])
    
    # Add pose quality metrics
    pose_columns = []
    if include_pose_metrics:
        pose_columns.extend(['clashes', 'strain_energy'])
    
    # Add fingerprint information
    fp_columns = []
    if include_fingerprints:
        fp_columns.extend(['morgan_fp', 'rdkit_fp', 'interaction_fp', 'interactions', 'num_interactions'])
    
    # Add any SDF properties (columns starting with 'prop_' or other property columns)
    property_columns = [col for col in export_df.columns if col.startswith('prop_') or 
                       col in ['binding_affinity', 'docking_score', 'energy', 'rmsd']]
    
    # Select columns for export
    selected_columns = []
    for col_group in [core_columns, grade_columns, prediction_columns, pose_columns, 
                     fp_columns, property_columns]:
        for col in col_group:
            if col in export_df.columns and col not in selected_columns:
                selected_columns.append(col)
    
    # Keep only selected columns
    export_df = export_df[selected_columns].copy()
    
    # Clean up fingerprint data for export if included
    if include_fingerprints:
        for fp_col in ['morgan_fp', 'rdkit_fp']:
            if fp_col in export_df.columns:
                # Convert list fingerprints to string representation
                export_df[fp_col] = export_df[fp_col].apply(
                    lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x) if x is not None else ''
                )
    
    # Clean up timestamp columns
    for timestamp_col in ['grade_timestamp', 'prediction_timestamp']:
        if timestamp_col in export_df.columns:
            export_df[timestamp_col] = export_df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return export_df


def export_to_sdf(df: pd.DataFrame, output_path: str, include_predictions: bool = True,
                 include_pose_metrics: bool = True) -> None:
    """
    Export molecules DataFrame to SDF format using RDKit PandasTools.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for output SDF file
        include_predictions: Whether to include ML predictions as properties
        include_pose_metrics: Whether to include pose quality metrics
    """
    try:
        # Check if we have RDKit Mol objects
        if 'mol' not in df.columns or df['mol'].isna().all():
            logger.error("No RDKit Mol objects found in DataFrame - cannot export to SDF")
            raise ValueError("RDKit Mol objects required for SDF export")
        
        # Prepare DataFrame for export
        export_df = prepare_dataframe_for_export(
            df, 
            include_predictions=include_predictions,
            include_fingerprints=False,  # Don't include raw fingerprints in SDF
            include_pose_metrics=include_pose_metrics
        )
        
        # Ensure we have the mol column
        if 'mol' in df.columns:
            export_df['mol'] = df['mol']
        
        # Remove rows with invalid molecules
        export_df = export_df[export_df['mol'].notna()].copy()
        
        if len(export_df) == 0:
            logger.warning("No valid molecules to export")
            return
        
        # Use RDKit PandasTools to write SDF
        PandasTools.WriteSDF(export_df, output_path, molColName='mol', properties=list(export_df.columns))
        
        logger.info(f"Exported {len(export_df)} molecules to SDF: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to SDF: {e}")
        raise


def export_to_csv(df: pd.DataFrame, output_path: str, include_predictions: bool = True,
                 include_fingerprints: bool = False, include_pose_metrics: bool = True) -> None:
    """
    Export molecules DataFrame to CSV format.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for output CSV file
        include_predictions: Whether to include ML predictions
        include_fingerprints: Whether to include fingerprint data
        include_pose_metrics: Whether to include pose quality metrics
    """
    try:
        # Prepare DataFrame for export
        export_df = prepare_dataframe_for_export(
            df,
            include_predictions=include_predictions,
            include_fingerprints=include_fingerprints,
            include_pose_metrics=include_pose_metrics
        )
        
        if len(export_df) == 0:
            logger.warning("No data to export")
            return
        
        # Export to CSV
        export_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(export_df)} molecules to CSV: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise


def export_graded_molecules_only(df: pd.DataFrame, output_path: str, format: str = "csv") -> None:
    """
    Export only graded molecules.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for output file
        format: Export format ('csv' or 'sdf')
    """
    graded_df = df[df['grade'].notna()].copy()
    
    if len(graded_df) == 0:
        logger.warning("No graded molecules to export")
        return
    
    if format.lower() == "csv":
        export_to_csv(graded_df, output_path)
    elif format.lower() == "sdf":
        export_to_sdf(graded_df, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def export_predictions_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Export summary of ML predictions and uncertainties.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for output CSV file
    """
    try:
        # Get molecules with predictions
        pred_df = df[df['prediction'].notna()].copy()
        
        if len(pred_df) == 0:
            logger.warning("No predictions to export")
            return
        
        # Create summary DataFrame
        summary_df = pred_df[['id', 'name', 'smiles', 'score', 'prediction', 
                             'prediction_uncertainty', 'grade']].copy()
        
        # Add prediction confidence category
        summary_df['confidence'] = pd.cut(
            1 - summary_df['prediction_uncertainty'],  # Convert uncertainty to confidence
            bins=[0, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Sort by uncertainty (most uncertain first)
        summary_df = summary_df.sort_values('prediction_uncertainty', ascending=False)
        
        # Export to CSV
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported predictions summary for {len(summary_df)} molecules to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting predictions summary: {e}")
        raise


def export_pose_quality_report(df: pd.DataFrame, output_path: str) -> None:
    """
    Export pose quality analysis report.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for output CSV file
    """
    try:
        # Get molecules with pose quality data
        pose_df = df[(df['clashes'].notna()) | (df['strain_energy'].notna())].copy()
        
        if len(pose_df) == 0:
            logger.warning("No pose quality data to export")
            return
        
        # Create report DataFrame
        report_df = pose_df[['id', 'name', 'smiles', 'score', 'clashes', 'strain_energy', 'grade']].copy()
        
        # Add quality categories
        report_df['clash_category'] = pd.cut(
            report_df['clashes'].fillna(0),
            bins=[-1, 0, 2, 5, float('inf')],
            labels=['None', 'Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        report_df['strain_category'] = pd.cut(
            report_df['strain_energy'].fillna(0),
            bins=[-float('inf'), 5, 15, 30, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        # Sort by overall quality (clashes first, then strain energy)
        report_df = report_df.sort_values(['clashes', 'strain_energy'], ascending=[True, True])
        
        # Export to CSV
        report_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported pose quality report for {len(report_df)} molecules to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting pose quality report: {e}")
        raise


def export_interaction_analysis(df: pd.DataFrame, output_path: str) -> None:
    """
    Export interaction analysis report.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for output CSV file
    """
    try:
        # Get molecules with interaction data
        interaction_df = df[df['interactions'].notna()].copy()
        
        if len(interaction_df) == 0:
            logger.warning("No interaction data to export")
            return
        
        # Parse interaction details
        interaction_summaries = []
        
        for _, row in interaction_df.iterrows():
            try:
                interactions = json.loads(row['interactions'])
                
                if isinstance(interactions, list):
                    # Count interaction types
                    interaction_types = {}
                    for interaction in interactions:
                        if isinstance(interaction, dict):
                            interaction_type = interaction.get('type', 'unknown')
                            interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
                    
                    summary = {
                        'id': row['id'],
                        'name': row['name'],
                        'smiles': row['smiles'],
                        'score': row['score'],
                        'total_interactions': len(interactions),
                        'grade': row.get('grade')
                    }
                    
                    # Add interaction type counts
                    for interaction_type, count in interaction_types.items():
                        summary[f'{interaction_type}_count'] = count
                    
                    interaction_summaries.append(summary)
                    
            except (json.JSONDecodeError, TypeError):
                continue
        
        if not interaction_summaries:
            logger.warning("No valid interaction data found")
            return
        
        # Create DataFrame and export
        summary_df = pd.DataFrame(interaction_summaries)
        summary_df = summary_df.sort_values('total_interactions', ascending=False)
        
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Exported interaction analysis for {len(summary_df)} molecules to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error exporting interaction analysis: {e}")
        raise


def create_export_package(df: pd.DataFrame, output_dir: str, session_name: str = "pickm8_export") -> None:
    """
    Create a complete export package with multiple file formats and reports.
    
    Args:
        df: Molecules DataFrame
        output_dir: Directory for export package
        session_name: Name prefix for exported files
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating export package in {output_dir}")
        
        # Export all molecules to different formats
        if len(df) > 0:
            # Complete dataset
            export_to_csv(df, output_path / f"{session_name}_complete.csv", 
                         include_predictions=True, include_pose_metrics=True)
            
            # Try to export SDF if possible
            if 'mol' in df.columns and df['mol'].notna().any():
                try:
                    export_to_sdf(df, output_path / f"{session_name}_complete.sdf")
                except Exception as e:
                    logger.warning(f"Could not export SDF: {e}")
            
            # Graded molecules only
            graded_df = df[df['grade'].notna()]
            if len(graded_df) > 0:
                export_to_csv(graded_df, output_path / f"{session_name}_graded.csv")
                
                if 'mol' in graded_df.columns and graded_df['mol'].notna().any():
                    try:
                        export_to_sdf(graded_df, output_path / f"{session_name}_graded.sdf")
                    except Exception as e:
                        logger.warning(f"Could not export graded SDF: {e}")
            
            # Predictions summary
            if 'prediction' in df.columns and df['prediction'].notna().any():
                export_predictions_summary(df, output_path / f"{session_name}_predictions.csv")
            
            # Pose quality report
            if ('clashes' in df.columns and df['clashes'].notna().any()) or \
               ('strain_energy' in df.columns and df['strain_energy'].notna().any()):
                export_pose_quality_report(df, output_path / f"{session_name}_pose_quality.csv")
            
            # Interaction analysis
            if 'interactions' in df.columns and df['interactions'].notna().any():
                export_interaction_analysis(df, output_path / f"{session_name}_interactions.csv")
            
            # Create summary statistics
            create_export_summary(df, output_path / f"{session_name}_summary.txt")
        
        logger.info(f"Export package created successfully in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating export package: {e}")
        raise


def create_export_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Create a text summary of the exported data.
    
    Args:
        df: Molecules DataFrame
        output_path: Path for summary text file
    """
    try:
        with open(output_path, 'w') as f:
            f.write("PickM8 Export Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total molecules: {len(df)}\n")
            
            if 'grade' in df.columns:
                graded_count = df['grade'].notna().sum()
                f.write(f"Graded molecules: {graded_count} ({graded_count/len(df)*100:.1f}%)\n")
                
                if graded_count > 0:
                    grade_dist = df['grade'].value_counts()
                    f.write("Grade distribution:\n")
                    for grade, count in grade_dist.items():
                        f.write(f"  {grade}: {count} ({count/graded_count*100:.1f}%)\n")
            
            # Prediction statistics
            if 'prediction' in df.columns:
                pred_count = df['prediction'].notna().sum()
                f.write(f"\nMolecules with predictions: {pred_count}\n")
                
                if pred_count > 0 and 'prediction_uncertainty' in df.columns:
                    avg_uncertainty = df['prediction_uncertainty'].mean()
                    f.write(f"Average prediction uncertainty: {avg_uncertainty:.3f}\n")
            
            # Pose quality statistics
            if 'clashes' in df.columns:
                clash_data = df['clashes'].dropna()
                if len(clash_data) > 0:
                    f.write(f"\nPose quality metrics:\n")
                    f.write(f"Average clashes: {clash_data.mean():.1f}\n")
                    f.write(f"Molecules with no clashes: {(clash_data == 0).sum()}\n")
            
            if 'strain_energy' in df.columns:
                strain_data = df['strain_energy'].dropna()
                if len(strain_data) > 0:
                    f.write(f"Average strain energy: {strain_data.mean():.2f}\n")
            
            # Interaction statistics
            if 'num_interactions' in df.columns:
                interaction_data = df['num_interactions'].dropna()
                if len(interaction_data) > 0:
                    f.write(f"\nInteraction statistics:\n")
                    f.write(f"Average interactions per molecule: {interaction_data.mean():.1f}\n")
                    f.write(f"Molecules with interactions: {(interaction_data > 0).sum()}\n")
            
            f.write(f"\nExport created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Export summary created: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating export summary: {e}")
        raise


def validate_export_data(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate DataFrame before export and return potential issues.
    
    Args:
        df: Molecules DataFrame
        
    Returns:
        Dictionary mapping issue types to lists of issues
    """
    issues = {
        "warnings": [],
        "errors": []
    }
    
    if len(df) == 0:
        issues["errors"].append("DataFrame is empty")
        return issues
    
    # Check for required columns
    required_columns = ['id', 'name']
    for col in required_columns:
        if col not in df.columns:
            issues["errors"].append(f"Missing required column: {col}")
    
    # Check for missing critical data
    if 'smiles' in df.columns and df['smiles'].isna().sum() > len(df) * 0.5:
        issues["warnings"].append("More than 50% of molecules missing SMILES")
    
    if 'mol' in df.columns and df['mol'].isna().sum() > len(df) * 0.5:
        issues["warnings"].append("More than 50% of molecules missing RDKit Mol objects")
    
    # Check grade data
    if 'grade' in df.columns:
        valid_grades = {'A', 'B', 'C', 'D', 'F'}
        invalid_grades = set(df['grade'].dropna()) - valid_grades
        if invalid_grades:
            issues["warnings"].append(f"Invalid grade values found: {invalid_grades}")
    
    return issues