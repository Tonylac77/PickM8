import streamlit as st
from utils.io_handlers import DataHandler
from utils.visualization import MoleculeVisualizer
from utils.molecule_utils import filter_molecules_by_grade_status, sort_molecules
import polars as pl
from datetime import datetime
import json

st.set_page_config(page_title="Active Learning - PickM8", page_icon="🎯", layout="wide")

def init_page_state():
    if 'current_mol_idx' not in st.session_state:
        st.session_state.current_mol_idx = 0
    if 'mode' not in st.session_state:
        st.session_state.mode = 'annotate'
    if 'sort_method' not in st.session_state:
        st.session_state.sort_method = 'score'
    if 'history' not in st.session_state:
        st.session_state.history = []

def save_grade(mol_id, grade):
    data_handler = DataHandler(st.session_state.session_id)
    
    new_grade = {
        'mol_id': mol_id,
        'grade': grade,
        'timestamp': datetime.now(),
        'session_id': st.session_state.session_id
    }
    
    existing_grades = data_handler.load_grades()
    if existing_grades is not None:
        grades_list = existing_grades.to_dicts()
        grades_list.append(new_grade)
        data_handler.save_grades(grades_list)
    else:
        data_handler.save_grades([new_grade])
    
    if mol_id not in st.session_state.get('grades', {}):
        st.session_state.grades[mol_id] = grade

def main():
    st.title("🎯 Active Learning Interface")
    init_page_state()
    
    data_handler = DataHandler(st.session_state.session_id)
    visualizer = MoleculeVisualizer()
    
    molecules_df = data_handler.load_molecules()
    grades_df = data_handler.load_grades()
    predictions_df = data_handler.load_predictions()
    session_state = data_handler.load_session_state()
    
    if molecules_df is None or molecules_df.is_empty():
        st.error("No molecules loaded. Please upload a screen first.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Controls")
        
        mode = st.selectbox(
            "Mode",
            ["annotate", "review"],
            index=0 if st.session_state.mode == "annotate" else 1
        )
        st.session_state.mode = mode
        
        sort_options = ["score"]
        if predictions_df is not None and not predictions_df.is_empty():
            sort_options.extend(["uncertainty", "prediction"])
        
        sort_method = st.selectbox(
            "Sort by",
            sort_options,
            index=sort_options.index(st.session_state.sort_method) if st.session_state.sort_method in sort_options else 0
        )
        st.session_state.sort_method = sort_method
        
        st.divider()
        
        if grades_df is not None:
            st.metric("Graded", len(grades_df))
        st.metric("Total", len(molecules_df))
        
        if st.button("Train Model", type="primary", disabled=grades_df is None or len(grades_df) < 10):
            st.switch_page("pages/5_⚙️_Settings.py")
    
    # Filter and sort molecules
    filtered_df = filter_molecules_by_grade_status(molecules_df, grades_df, mode)
    sorted_df = sort_molecules(filtered_df, predictions_df, sort_method)
    
    if sorted_df.is_empty():
        st.info("No molecules to show in current mode.")
        return
    
    # Main content
    mol_list = sorted_df.to_dicts()
    current_mol = mol_list[st.session_state.current_mol_idx]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Molecule: {current_mol['name']}")
        st.caption(f"Score: {current_mol['score']:.3f}")
        
        if session_state and 'protein_content' in session_state:
            visualizer.show_complex(
                session_state['protein_content'],
                current_mol['mol_block'],
                current_mol['interactions'],
                key=f"mol_{current_mol['id']}"
            )
        
        interaction_summary = visualizer.get_interaction_summary(current_mol['interactions'])
        with st.expander("Interaction Summary", expanded=True):
            visualizer.show_interaction_legend(interaction_summary)
    
    with col2:
        st.subheader("Grade")
        
        grade_options = ['A', 'B', 'C', 'D', 'F']
        current_grade = None
        
        if grades_df is not None:
            grade_row = grades_df.filter(pl.col('mol_id') == current_mol['id'])
            if not grade_row.is_empty():
                current_grade = grade_row['grade'][0]
        
        selected_grade = st.radio(
            "Select grade",
            grade_options,
            index=grade_options.index(current_grade) if current_grade else None,
            key=f"grade_{current_mol['id']}"
        )
        
        if st.button("Submit", type="primary", disabled=selected_grade is None):
            save_grade(current_mol['id'], selected_grade)
            st.success(f"Saved grade: {selected_grade}")
            
            if st.session_state.current_mol_idx < len(mol_list) - 1:
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        st.divider()
        
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Previous", disabled=st.session_state.current_mol_idx == 0):
                st.session_state.current_mol_idx -= 1
                st.rerun()
        
        with col_next:
            if st.button("Next →", disabled=st.session_state.current_mol_idx >= len(mol_list) - 1):
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        st.divider()
        st.caption(f"Molecule {st.session_state.current_mol_idx + 1} of {len(mol_list)}")
        
        # Molecule list
        st.subheader("Molecule List")
        for i, mol in enumerate(mol_list[:10]):
            prefix = "✓ " if grades_df and not grades_df.filter(pl.col('mol_id') == mol['id']).is_empty() else "  "
            if st.button(f"{prefix}{mol['name']}", key=f"mol_btn_{mol['id']}"):
                st.session_state.current_mol_idx = i
                st.rerun()

if __name__ == "__main__":
    main()