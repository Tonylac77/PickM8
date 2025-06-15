import streamlit as st
from utils.io_handlers import DataHandler
from utils.visualization import MoleculeVisualizer
from utils.molecule_utils import filter_molecules_by_grade_status, sort_molecules
from utils.posecheck_utils import PoseCheckAnalyzer
import polars as pl
from datetime import datetime

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

def format_progress_percentage(current, total):
    """Format progress as a percentage string"""
    if total == 0:
        return "0%"
    percentage = (current / total) * 100
    return f"{percentage:.1f}%"

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

def calculate_posecheck_metrics(session_state, molecules_df, progress_bar=None, status_text=None):
    """Calculate PoseCheck metrics for molecules if not already calculated"""
    import time
    
    if 'posecheck_analyzer' not in st.session_state:
        if session_state and 'protein_content' in session_state:
            if status_text:
                status_text.text("🔧 Initializing PoseCheck analyzer...")
            st.session_state.posecheck_analyzer = PoseCheckAnalyzer()
            st.session_state.posecheck_analyzer.load_protein_from_content(session_state['protein_content'])
    
    posecheck_data = st.session_state.get('posecheck_data', {})
    
    # Get molecules that need calculation
    molecules_to_calculate = []
    mol_ids_to_calculate = []
    
    for mol in molecules_df.to_dicts():
        mol_id = mol['id']
        if mol_id not in posecheck_data:
            molecules_to_calculate.append(mol['mol_block'])
            mol_ids_to_calculate.append(mol_id)
    
    total_molecules = len(mol_ids_to_calculate)
    
    # If we have molecules to calculate and a valid analyzer
    if molecules_to_calculate and 'posecheck_analyzer' in st.session_state:
        try:
            start_time = time.time()
            
            if status_text:
                status_text.text(f"🧪 Starting analysis of {total_molecules} molecules...")
            
            # For better progress tracking, we'll process in chunks
            chunk_size = max(1, min(10, total_molecules // 5))  # Process in chunks for progress updates
            
            for i in range(0, total_molecules, chunk_size):
                chunk_start_time = time.time()
                
                chunk_end = min(i + chunk_size, total_molecules)
                chunk_molecules = molecules_to_calculate[i:chunk_end]
                chunk_ids = mol_ids_to_calculate[i:chunk_end]
                
                # Estimate remaining time
                if i > 0:
                    elapsed_time = time.time() - start_time
                    rate = i / elapsed_time  # molecules per second
                    remaining_molecules = total_molecules - chunk_end
                    estimated_remaining = remaining_molecules / rate if rate > 0 else 0
                    
                    if status_text:
                        if estimated_remaining > 60:
                            time_str = f"{estimated_remaining/60:.1f} min"
                        else:
                            time_str = f"{estimated_remaining:.0f} sec"
                        status_text.text(f"🧪 Processing molecules {i+1}-{chunk_end} of {total_molecules} (≈{time_str} remaining)")
                else:
                    if status_text:
                        status_text.text(f"🧪 Processing molecules {i+1}-{chunk_end} of {total_molecules}...")
                
                # Use parallel processing for batch calculation
                clashes_list, strain_list = st.session_state.posecheck_analyzer.analyze_multiple_molecules_smart(
                    chunk_molecules
                )
                
                # Store results for this chunk
                for j, mol_id in enumerate(chunk_ids):
                    posecheck_data[mol_id] = {
                        'clashes': clashes_list[j],
                        'strain_energy': strain_list[j]
                    }
                
                # Update progress bar
                if progress_bar:
                    progress = (chunk_end) / total_molecules
                    progress_percentage = format_progress_percentage(chunk_end, total_molecules)
                    progress_bar.progress(progress, text=f"Progress: {progress_percentage} ({chunk_end}/{total_molecules} molecules)")
            
            total_time = time.time() - start_time
            if status_text:
                if total_time > 60:
                    time_str = f"{total_time/60:.1f} minutes"
                else:
                    time_str = f"{total_time:.1f} seconds"
                status_text.text(f"✅ Completed analysis of {total_molecules} molecules in {time_str}!")
            
        except Exception as e:
            if status_text:
                status_text.text(f"⚠️ Error in PoseCheck analysis: {str(e)}")
            st.warning(f"Could not calculate PoseCheck metrics: {str(e)}")
            # Fallback to default values
            for mol_id in mol_ids_to_calculate:
                posecheck_data[mol_id] = {
                    'clashes': 0,
                    'strain_energy': 0.0
                }
                
            # Update progress to complete even on error
            if progress_bar:
                progress_bar.progress(1.0, text="Error occurred - falling back to default values")
    else:
        # No analyzer available, use default values
        if status_text:
            if not molecules_to_calculate:
                status_text.text("✅ All molecules already analyzed!")
            else:
                status_text.text("⚠️ PoseCheck analyzer not available, using default values...")
        
        for mol_id in mol_ids_to_calculate:
            posecheck_data[mol_id] = {
                'clashes': 0,
                'strain_energy': 0.0
            }
        
        # Update progress to complete
        if progress_bar:
            if not molecules_to_calculate:
                progress_bar.progress(1.0, text="All molecules already analyzed!")
            else:
                progress_bar.progress(1.0, text="Using default values (PoseCheck not available)")
    
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
    
    # Calculate PoseCheck metrics for all molecules upfront with progress bar
    molecules_needing_calculation = []
    existing_posecheck_data = st.session_state.get('posecheck_data', {})
    
    for mol in molecules_df.to_dicts():
        mol_id = mol['id']
        if mol_id not in existing_posecheck_data:
            molecules_needing_calculation.append(mol_id)
    
    if molecules_needing_calculation:
        # Create an expandable section for the progress tracking
        with st.expander(f"🧪 Calculating Pose Quality Metrics ({len(molecules_needing_calculation)} molecules)", expanded=True):
            st.markdown("""
            **PoseCheck Analysis:** Evaluating molecular poses for clashes and strain energy.
            This helps identify problematic poses that might need attention.
            """)
            
            # Create progress metrics in columns
            progress_cols = st.columns(3)
            with progress_cols[0]:
                total_metric = st.metric("Total Molecules", len(molecules_df))
            with progress_cols[1]:
                remaining_metric = st.metric("To Analyze", len(molecules_needing_calculation))
            with progress_cols[2]:
                completed_metric = st.metric("Already Done", len(molecules_df) - len(molecules_needing_calculation))
            
            st.divider()
            
            # Create progress bar and status text containers
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                
            with status_container:
                status_text = st.empty()
                
            # Calculate metrics with progress tracking
            posecheck_data = calculate_posecheck_metrics(
                session_state, 
                molecules_df, 
                progress_bar=progress_bar, 
                status_text=status_text
            )
            
            # Clean up progress indicators after completion
            import time
            time.sleep(1.5)  # Brief pause to show completion
            progress_container.empty()
            
            # Show final summary
            num_calculated = len(molecules_needing_calculation)
            if num_calculated > 0:
                st.success(f"✅ Successfully calculated pose quality metrics for {num_calculated} molecules!")
            
            # Clear status after showing final message
            time.sleep(1)
            status_container.empty()
            
    else:
        # All molecules already have metrics calculated
        posecheck_data = calculate_posecheck_metrics(session_state, molecules_df)
    
    # Initialize any missing data
    for mol in molecules_df.to_dicts():
        mol_id = mol['id']
        if mol_id not in posecheck_data:
            posecheck_data[mol_id] = {'clashes': 0, 'strain_energy': 0.0}
    
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
        
        st.divider()
        
        if st.button("🏠 Main Entry", type="secondary"):
            st.switch_page("main_entry.py")
    
    # Filter and sort molecules
    filtered_df = filter_molecules_by_grade_status(molecules_df, grades_df, mode)
    sorted_df = sort_molecules(filtered_df, predictions_df, sort_method)
    
    if sorted_df.is_empty():
        st.info("No molecules to show in current mode.")
        return
    
    # Main content
    mol_list = sorted_df.to_dicts()
    current_mol = mol_list[st.session_state.current_mol_idx]
    
    # Three-column layout: 2D view, 3D visualization, and grading interface
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # 2D Ligand Structure
        st.subheader("2D Structure")
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            import io
            from PIL import Image
            
            # Convert mol_block to RDKit molecule
            mol = Chem.MolFromMolBlock(current_mol['mol_block'])
            if mol is not None:
                # Generate 2D coordinates if not present
                from rdkit.Chem import rdDepictor
                rdDepictor.Compute2DCoords(mol)
                
                # Create 2D image
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption=f"{current_mol['name']}", use_container_width=True)
            else:
                st.error("Could not parse molecule structure")
        except Exception as e:
            st.error(f"Error generating 2D structure: {str(e)}")
        
        # Compact molecule data under 2D diagram
        st.markdown("**📊 Data**")
        pc_data = posecheck_data.get(current_mol['id'], {'clashes': 0, 'strain_energy': 0.0})
        
        # Much smaller data display
        st.markdown(f"""
        <div style="font-size: 12px; line-height: 1.2;">
        <strong>Score:</strong> {current_mol['score']:.3f}<br>
        <strong>Clashes:</strong> {pc_data['clashes']}<br>
        <strong>Strain:</strong> {pc_data['strain_energy']:.2f}<br>
        <strong>Interactions:</strong> {current_mol.get('num_interactions', 0)}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Main 3D visualization in center
        st.subheader(f"🧬 {current_mol['name']}")
        
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
    
    with col3:
        # Grading interface (without gradient background)
        st.subheader("⭐ Grade")
        
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
        
        # Grade selection buttons
        selected_grade = None
        for grade in grade_options:
            color = grade_colors[grade]
            is_selected = current_grade == grade
            
            if st.button(
                f"{color} {grade}",
                key=f"grade_{grade}_{current_mol['id']}",
                type="primary" if is_selected else "secondary",
                use_container_width=True,
                help=f"Grade {grade}"
            ):
                selected_grade = grade
        
        if selected_grade:
            save_grade(current_mol['id'], selected_grade)
            st.success(f"✅ {selected_grade}")
            
            if st.session_state.current_mol_idx < len(mol_list) - 1:
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        st.divider()
        
        # Navigation
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️", disabled=st.session_state.current_mol_idx == 0, use_container_width=True, help="Previous molecule"):
                st.session_state.current_mol_idx -= 1
                st.rerun()
        
        with col_next:
            if st.button("➡️", disabled=st.session_state.current_mol_idx >= len(mol_list) - 1, use_container_width=True, help="Next molecule"):
                st.session_state.current_mol_idx += 1
                st.rerun()
        
        st.caption(f"{st.session_state.current_mol_idx + 1}/{len(mol_list)}")
    
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
        
        # Compact page navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("⬅️", disabled=st.session_state.table_page == 0, help="Previous page"):
                st.session_state.table_page -= 1
                st.rerun()
        with col2:
            st.caption(f"Page {st.session_state.table_page + 1}/{total_pages}")
        with col3:
            if st.button("➡️", disabled=st.session_state.table_page >= total_pages - 1, help="Next page"):
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