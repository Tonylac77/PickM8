import streamlit as st
from utils.io_handlers import DataHandler
from utils.visualization import MoleculeVisualizer
from utils.molecule_utils import filter_molecules_by_grade_status, sort_molecules
from utils.posecheck_utils import PoseCheckAnalyzer
from utils.poseview_utils import PoseViewAPI
import polars as pl
from datetime import datetime
import json
import base64

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
    if 'posecheck_data' not in st.session_state:
        st.session_state.posecheck_data = {}
    if 'poseview_cache' not in st.session_state:
        st.session_state.poseview_cache = {}
    if 'poseview_api' not in st.session_state:
        st.session_state.poseview_api = PoseViewAPI()

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

def generate_poseview_diagram(session_state, current_mol):
    """Generate 2D interaction diagram using PoseView API"""
    mol_id = current_mol['id']
    
    # Check cache first
    if mol_id in st.session_state.poseview_cache:
        return st.session_state.poseview_cache[mol_id]
    
    if not session_state or 'protein_content' not in session_state:
        return None
    
    try:
        with st.spinner("Generating 2D interaction diagram..."):
            result = st.session_state.poseview_api.generate_interaction_diagram(
                session_state['protein_content'],
                current_mol['mol_block'],
                current_mol['name']
            )
            
            if result:
                # Cache the result
                st.session_state.poseview_cache[mol_id] = result
                return result
            else:
                st.warning("Could not generate 2D interaction diagram")
                return None
                
    except Exception as e:
        st.error(f"Error generating diagram: {str(e)}")
        return None

def create_molecule_table(molecules_df, grades_df, posecheck_data):
    """Create a table with molecule properties"""
    mol_data = []
    
    for mol in molecules_df.to_dicts():
        mol_id = mol['id']
        
        # Check if graded
        is_graded = False
        grade = ""
        if grades_df is not None:
            grade_row = grades_df.filter(pl.col('mol_id') == mol_id)
            if not grade_row.is_empty():
                is_graded = True
                grade = grade_row['grade'][0]
        
        # Get PoseCheck data
        pc_data = posecheck_data.get(mol_id, {'clashes': 0, 'strain_energy': 0.0})
        
        mol_data.append({
            'Name': mol['name'],
            'Score': f"{mol['score']:.3f}",
            'Clashes': pc_data['clashes'],
            'Strain Energy': f"{pc_data['strain_energy']:.2f}",
            'Grade': grade if is_graded else "Not graded",
            'Status': "✅ Graded" if is_graded else "⏳ Pending",
            'mol_idx': len(mol_data)  # Store index for selection
        })
    
    return mol_data

def calculate_posecheck_metrics(session_state, molecules_df):
    """Calculate PoseCheck metrics for molecules if not already calculated"""
    if 'posecheck_analyzer' not in st.session_state:
        if session_state and 'protein_content' in session_state:
            st.session_state.posecheck_analyzer = PoseCheckAnalyzer()
            st.session_state.posecheck_analyzer.load_protein_from_content(session_state['protein_content'])
    
    posecheck_data = st.session_state.get('posecheck_data', {})
    
    # Calculate metrics for molecules that don't have them yet
    for mol in molecules_df.to_dicts():
        mol_id = mol['id']
        if mol_id not in posecheck_data:
            try:
                if 'posecheck_analyzer' in st.session_state:
                    clashes, strain = st.session_state.posecheck_analyzer.analyze_molecule(mol['mol_block'])
                    posecheck_data[mol_id] = {
                        'clashes': clashes,
                        'strain_energy': strain
                    }
                else:
                    posecheck_data[mol_id] = {
                        'clashes': 0,
                        'strain_energy': 0.0
                    }
            except Exception as e:
                st.warning(f"Could not calculate PoseCheck metrics for {mol['name']}: {str(e)}")
                posecheck_data[mol_id] = {
                    'clashes': 0,
                    'strain_energy': 0.0
                }
    
    st.session_state.posecheck_data = posecheck_data
    return posecheck_data

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
    
    # Calculate PoseCheck metrics
    with st.spinner("Calculating pose quality metrics..."):
        posecheck_data = calculate_posecheck_metrics(session_state, molecules_df)
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Controls")
        
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
        
        # Enhanced metrics with colors
        col1, col2 = st.columns(2)
        with col1:
            graded_count = len(grades_df) if grades_df is not None else 0
            st.metric("✅ Graded", graded_count)
        with col2:
            st.metric("📊 Total", len(molecules_df))
        
        # Progress bar
        if grades_df is not None:
            progress = len(grades_df) / len(molecules_df)
            st.progress(progress)
            st.caption(f"{progress:.1%} Complete")
        
        st.divider()
        
        if st.button("🤖 Train Model", type="primary", disabled=grades_df is None or len(grades_df) < 10):
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
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Controls")
        
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
        
        # Enhanced metrics with colors
        col1, col2 = st.columns(2)
        with col1:
            graded_count = len(grades_df) if grades_df is not None else 0
            st.metric("✅ Graded", graded_count)
        with col2:
            st.metric("📊 Total", len(molecules_df))
        
        # Progress bar
        if grades_df is not None:
            progress = len(grades_df) / len(molecules_df)
            st.progress(progress)
            st.caption(f"{progress:.1%} Complete")
        
        st.divider()
        
        if st.button("🤖 Train Model", type="primary", disabled=grades_df is None or len(grades_df) < 10):
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
    
    # Enhanced layout with 2D diagram
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"🧬 Molecule: {current_mol['name']}")
        
        # Enhanced molecule info with colors
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("📊 Score", f"{current_mol['score']:.3f}")
        with info_cols[1]:
            pc_data = posecheck_data.get(current_mol['id'], {'clashes': 0, 'strain_energy': 0.0})
            st.metric("⚠️ Clashes", pc_data['clashes'])
        with info_cols[2]:
            st.metric("⚡ Strain Energy", f"{pc_data['strain_energy']:.2f}")
        with info_cols[3]:
            st.metric("🔬 Interactions", current_mol.get('num_interactions', 0))
        
        if session_state and 'protein_content' in session_state:
            visualizer.show_complex(
                session_state['protein_content'],
                current_mol['mol_block'],
                current_mol['interactions'],
                key=f"mol_{current_mol['id']}"
            )
        
        interaction_summary = visualizer.get_interaction_summary(current_mol['interactions'])
        with st.expander("🔬 Interaction Summary", expanded=True):
            visualizer.show_interaction_legend(interaction_summary)
    
    with col2:
        st.subheader("📋 2D Interaction Pattern")
        
        # Generate and display PoseView diagram
        poseview_result = generate_poseview_diagram(session_state, current_mol)
        
        if poseview_result:
            try:
                # Try to get PNG image
                png_data = st.session_state.poseview_api.get_png_image_data(poseview_result)
                if png_data:
                    st.image(png_data, caption="2D Interaction Diagram", use_column_width=True)
                else:
                    # Try SVG as fallback
                    svg_data = st.session_state.poseview_api.get_svg_image_data(poseview_result)
                    if svg_data:
                        st.components.v1.html(
                            f'<div style="display: flex; justify-content: center;">{svg_data}</div>',
                            height=400
                        )
                    else:
                        st.info("2D diagram generated but image data not available")
            except Exception as e:
                st.error(f"Error displaying diagram: {str(e)}")
        else:
            st.info("Click 'Generate 2D Diagram' to create interaction pattern")
            if st.button("🎨 Generate 2D Diagram", type="secondary"):
                st.rerun()
    
    with col3:
        # Enhanced grading interface with colors and bigger buttons
        st.markdown("""
        <style>
        .grade-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            margin: 10px 0;
        }
        .grade-title {
            color: white;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="grade-container">', unsafe_allow_html=True)
        st.markdown('<div class="grade-title">⭐ Grade Molecule</div>', unsafe_allow_html=True)
        
        grade_options = ['A', 'B', 'C', 'D', 'F']
        grade_colors = {
            'A': '🟢',
            'B': '🔵', 
            'C': '🟡',
            'D': '🟠',
            'F': '🔴'
        }
        
        current_grade = None
        
        if grades_df is not None:
            grade_row = grades_df.filter(pl.col('mol_id') == current_mol['id'])
            if not grade_row.is_empty():
                current_grade = grade_row['grade'][0]
        
        # Enhanced grade selection with colors and descriptions
        grade_descriptions = {
            'A': 'Excellent - High quality pose',
            'B': 'Good - Acceptable pose',
            'C': 'Average - Moderate issues',
            'D': 'Poor - Significant problems',
            'F': 'Fail - Unacceptable pose'
        }
        
        selected_grade = None
        for grade in grade_options:
            color = grade_colors[grade]
            desc = grade_descriptions[grade]
            is_selected = current_grade == grade
            
            if st.button(
                f"{color} {grade} - {desc}",
                key=f"grade_{grade}_{current_mol['id']}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                selected_grade = grade
        
        if selected_grade:
            save_grade(current_mol['id'], selected_grade)
            st.success(f"✅ Saved grade: {selected_grade}")
            
            if st.session_state.current_mol_idx < len(mol_list) - 1:
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Enhanced navigation
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.current_mol_idx == 0, use_container_width=True):
                st.session_state.current_mol_idx -= 1
                st.rerun()
        
        with col_next:
            if st.button("➡️ Next", disabled=st.session_state.current_mol_idx >= len(mol_list) - 1, use_container_width=True):
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        st.caption(f"📍 Molecule {st.session_state.current_mol_idx + 1} of {len(mol_list)}")
    
    # Enhanced molecule table
    st.divider()
    st.subheader("📋 Molecule Table")
    
    # Create molecule table data
    table_data = create_molecule_table(sorted_df, grades_df, posecheck_data)
    
    # Display as interactive dataframe
    if table_data:
        # Convert to DataFrame for better display
        import pandas as pd
        df_display = pd.DataFrame(table_data)
        
        # Style the dataframe
        def color_grade(val):
            if val == "✅ Graded":
                return 'background-color: #90EE90'
            elif val == "⏳ Pending":
                return 'background-color: #FFE4B5'
            return ''
        
        def color_clashes(val):
            if val > 5:
                return 'background-color: #FFB6C1'
            elif val > 2:
                return 'background-color: #FFFFE0'
            return 'background-color: #F0FFF0'
        
        # Display with pagination
        rows_per_page = 20
        total_rows = len(df_display)
        total_pages = (total_rows - 1) // rows_per_page + 1
        
        if 'table_page' not in st.session_state:
            st.session_state.table_page = 0
        
        # Page navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Prev Page", disabled=st.session_state.table_page == 0):
                st.session_state.table_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.table_page + 1} of {total_pages}")
        with col3:
            if st.button("Next Page ➡️", disabled=st.session_state.table_page >= total_pages - 1):
                st.session_state.table_page += 1
                st.rerun()
        
        # Show current page
        start_idx = st.session_state.table_page * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        page_data = df_display.iloc[start_idx:end_idx]
        
        # Display table
        st.dataframe(
            page_data.drop('mol_idx', axis=1),  # Don't show the index column
            use_container_width=True,
            hide_index=True
        )
        
        # Add molecule selection below the table
        st.subheader("🎯 Quick Select")
        cols = st.columns(min(5, len(page_data)))
        
        for idx, (_, row) in enumerate(page_data.iterrows()):
            col_idx = idx % len(cols)
            with cols[col_idx]:
                actual_mol_idx = start_idx + idx
                button_text = f"{row['Name'][:10]}..."
                if len(row['Name']) <= 10:
                    button_text = row['Name']
                
                # Color code the button based on grade status
                button_type = "primary" if row['Status'] == "✅ Graded" else "secondary"
                
                if st.button(
                    button_text,
                    key=f"select_mol_{actual_mol_idx}",
                    type=button_type,
                    help=f"Score: {row['Score']}, Clashes: {row['Clashes']}"
                ):
                    st.session_state.current_mol_idx = actual_mol_idx
                    st.rerun()

if __name__ == "__main__":
    main()