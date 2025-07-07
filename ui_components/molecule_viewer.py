"""3D and 2D molecule visualization components."""
import streamlit as st
import json
import tempfile
from pathlib import Path
import io
import base64
from rdkit import Chem
from rdkit.Chem import Draw

try:
    from streamlit_molstar.docking import st_molstar_docking
    MOLSTAR_AVAILABLE = True
except ImportError:
    MOLSTAR_AVAILABLE = False

class MoleculeVisualizer:
    """3D molecule visualization using MolStar."""
    
    def __init__(self):
        self.interaction_colors = {
            "Proximal": "#999999",
            "Hydrogen bond": "#4C4CFF",
            "Water-bridged hydrogen bond": "#BFBFFF",
            "Weak hydrogen bond": "#66B2B2",
            "Ionic": "#00FF00",
            "Salt bridge": "#339933",
            "Cation-pi": "#FF9999",
            "Hydrophobic": "#FF7F00",
            "Halogen bond": "#7FFFFF",
            "Pi-stacking": "#FF3333",
        }
    
    def show_complex(self, protein_content: str, ligand_mol_block: str, 
                    interactions_json: str, key: str = "mol_view"):
        """Display protein-ligand complex with interactions."""
        if not MOLSTAR_AVAILABLE:
            st.error("MolStar visualization not available")
            return
            
        with tempfile.TemporaryDirectory() as tmpdir:
            protein_path = Path(tmpdir) / "protein.pdb"
            ligand_path = Path(tmpdir) / "ligand.sdf"
            
            with open(protein_path, 'w') as f:
                f.write(protein_content)
            
            with open(ligand_path, 'w') as f:
                f.write(ligand_mol_block)
            
            st_molstar_docking(
                str(protein_path),
                str(ligand_path),
                key=key,
                height=500,
                options = {"defaultPolymerReprType": "cartoon", "sizeFactor": 0.35}
            )
    
    def show_2d_structure(self, mol_block: str, size: tuple = (300, 300)):
        """Display 2D structure of molecule with proper flattening."""
        try:
            mol = Chem.MolFromMolBlock(mol_block)
            if mol is None:
                st.warning("Could not parse molecule structure")
                return
            
            # Generate 2D coordinates to flatten the molecule
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)
            
            # Generate 2D image
            img = Draw.MolToImage(mol, size=size)
            
            # Convert to base64 for display
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Display image
            st.markdown(
                f'<img src="data:image/png;base64,{img_str}" style="max-width: 100%; height: auto;">',
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Error generating 2D structure: {str(e)}")
    
    def show_molecule_info(self, mol_data: dict):
        """Display molecule information in a clean format."""
        st.markdown("### 📊 Molecule Properties")
        
        # Basic properties
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Name", mol_data.get('name', 'Unknown'))
            if 'score' in mol_data:
                st.metric("Score", f"{mol_data['score']:.3f}")
        
        with col2:
            if mol_data.get('clashes') is not None:
                st.metric("Clashes", mol_data['clashes'])
            if mol_data.get('num_interactions') is not None:
                st.metric("Interactions", mol_data['num_interactions'])
        
        # Additional properties if available
        if 'molecular_weight' in mol_data:
            st.metric("Molecular Weight", f"{mol_data['molecular_weight']:.2f}")
        if 'logp' in mol_data:
            st.metric("LogP", f"{mol_data['logp']:.2f}")
    
    def show_compact_molecule_info(self, mol_data: dict):
        """Display molecule information in ultra-compact format."""
        st.markdown(f"**{mol_data.get('name', 'Unknown')}**")
        
        # Key metrics in minimal space
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Score: {mol_data['score']:.3f}")
            if mol_data.get('clashes') is not None:
                st.text(f"Clashes: {mol_data['clashes']}")
        
        with col2:
            if mol_data.get('num_interactions') is not None:
                st.text(f"Interactions: {mol_data['num_interactions']}")
            if 'molecular_weight' in mol_data:
                st.text(f"MW: {mol_data['molecular_weight']:.1f}")
        
        # ML Predictions section
        if mol_data.get('prediction') is not None:
            st.divider()
            st.markdown("**🤖 ML Prediction**")
            
            # Get prediction (now already decoded to grade string)
            prediction_value = mol_data['prediction']
            
            # Handle both legacy numeric predictions and new grade string predictions
            if isinstance(prediction_value, str) and prediction_value in ['A', 'B', 'C', 'D']:
                # New system: prediction is already a grade string
                prediction_display = prediction_value
            else:
                # Legacy system: try to convert using label mapping
                label_mapping = None
                if hasattr(st.session_state, 'metadata') and st.session_state.metadata:
                    label_mapping = st.session_state.metadata.get('label_mapping')
                
                if label_mapping and isinstance(label_mapping, dict) and len(label_mapping) > 0:
                    try:
                        pred_int = int(float(prediction_value))
                        reverse_mapping = {v: k for k, v in label_mapping.items()}
                        prediction_display = reverse_mapping.get(pred_int, f"Unknown({pred_int})")
                    except (ValueError, TypeError):
                        prediction_display = f"Error({prediction_value})"
                else:
                    prediction_display = f"Raw({prediction_value})"
            
            # Prediction with color coding
            
            grade_colors = {
                'A': '🟢', 'B': '🔵', 'C': '🟡', 'D': '🟠'
            }
            icon = grade_colors.get(str(prediction_display), '⚪')
            
            st.markdown(f"{icon} **Grade: {prediction_display}**")
            
            # Uncertainty with color coding
            if mol_data.get('prediction_uncertainty') is not None:
                uncertainty = mol_data['prediction_uncertainty']
                
                # Color code uncertainty: low=green, medium=yellow, high=red
                if uncertainty < 0.3:
                    uncertainty_color = "🟢"
                    uncertainty_text = "Low"
                elif uncertainty < 0.6:
                    uncertainty_color = "🟡"
                    uncertainty_text = "Medium"
                else:
                    uncertainty_color = "🔴"
                    uncertainty_text = "High"
                
                st.text(f"{uncertainty_color} Uncertainty: {uncertainty_text} ({uncertainty:.2f})")
            
            # Timestamp
            if mol_data.get('prediction_timestamp') is not None:
                timestamp = mol_data['prediction_timestamp']
                if hasattr(timestamp, 'strftime'):
                    time_str = timestamp.strftime('%H:%M:%S')
                    st.text(f"⏰ Predicted at {time_str}")