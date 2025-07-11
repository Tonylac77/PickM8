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
        st.markdown(f"#### {mol_data.get('name', 'Unknown')}")
        
        # Key metrics in minimal space
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"##### Score: {mol_data['score']:.3f}")
            if mol_data.get('clashes') is not None:
                st.markdown(f"##### Clashes: {mol_data['clashes']}")
            if mol_data.get('strain_energy') is not None:
                st.markdown(f"##### Strain Energy: {mol_data['strain_energy']:.3f}")
                
                # Calculate strain energy per heavy atom if molecule is available
                if mol_data.get('mol') is not None:
                    try:
                        heavy_atom_count = mol_data['mol'].GetNumHeavyAtoms()
                        if heavy_atom_count > 0:
                            strain_per_heavy_atom = mol_data['strain_energy'] / heavy_atom_count
                            st.markdown(f"##### Strain/Heavy Atom: {strain_per_heavy_atom:.3f}")
                    except Exception:
                        pass  # Skip if calculation fails

        with col2:
            if mol_data.get('num_interactions') is not None:
                st.markdown(f"##### Interactions: {mol_data['num_interactions']}")
            if 'molecular_weight' in mol_data:
                st.markdown(f"##### MW: {mol_data['molecular_weight']:.1f}")

        # ML Predictions section
        if mol_data.get('prediction') is not None:
            st.markdown("#### 🤖 ML Prediction")
            
            # Get prediction (now already decoded to grade string)
            prediction_value = mol_data['prediction']
            
            prediction_display = prediction_value
            
            # Prediction with color coding
            
            grade_colors = {
                'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'
            }
            icon = grade_colors.get(str(prediction_display), '⚪')
            
            st.markdown(f"##### {icon} Grade: {prediction_display}")
