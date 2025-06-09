from streamlit_molstar.docking import st_molstar_docking
import streamlit as st
import json
import tempfile
from pathlib import Path

class MoleculeVisualizer:
    def __init__(self):
        self.interaction_colors = {
            "Proximal": "#999999",
            "Hydrogen bond": "#4C4CFF",
            "Water-bridged hydrogen bond": "#BFBFFF",
            "Weak hydrogen bond": "#66B2B2",
            "Ionic": "#00FF00",
            "Salt bridge": "#339933",
            "Cation-pi": "#FF9999",
            "Amide-aromatic stacking": "#B24C66",
            "Hydrophobic": "#FF7F00",
            "Halogen bond": "#7FFFFF",
            "Halogen-pi": "#7FFFFF",
            "Chalcogen bond": "#FFCC7F",
            "Chalcogen-pi": "#FFCC7F",
            "Repulsive": "#8C3F99",
            "Covalent bond": "#000000",
            "Atom overlap": "#666666",
            "Van der Waals clash": "#E5E5E5",
            "Van der Waals": "#7F7F7F",
            "Orthogonal multipolar": "#FFFF7F",
            "Parallel multipolar": "#FFFF7F",
            "Antiparallel multipolar": "#FFFF7F",
            "Pi-stacking": "#FF3333",
            "Face-to-face pi-stacking": "#FF3333",
            "Face-to-edge pi-stacking": "#FF3333",
            "Edge-to-edge pi-stacking": "#FF3333"
        }
    
    def show_complex(self, protein_content, ligand_mol_block, interactions_json, key="mol_view"):
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
                height=600,
                options = {"defaultPolymerReprType": "cartoon", "sizeFacotr": 0.35}
            )
    
    def get_interaction_summary(self, interactions_json):
        interactions = json.loads(interactions_json)
        summary = {}
        
        for inter in interactions:
            inter_type = inter.get('type', 'Unknown')
            if inter_type not in summary:
                summary[inter_type] = {
                    'count': 0,
                    'color': self.interaction_colors.get(inter_type, '#CCCCCC')
                }
            summary[inter_type]['count'] += 1
        
        return summary
    
    def show_interaction_legend(self, interaction_summary):
        if not interaction_summary:
            return
        
        cols = st.columns(min(len(interaction_summary), 4))
        
        for i, (inter_type, info) in enumerate(interaction_summary.items()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                color = info['color']
                count = info['count']
                st.markdown(
                    f'<div style="display: flex; align-items: center;">'
                    f'<div style="width: 20px; height: 20px; background-color: {color}; '
                    f'border-radius: 3px; margin-right: 8px;"></div>'
                    f'<span>{inter_type} ({count})</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )