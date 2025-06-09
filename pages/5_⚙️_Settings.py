import streamlit as st
from utils.io_handlers import DataHandler
from core.active_learning import ActiveLearningStrategy
import yaml
import time

st.set_page_config(page_title="Settings - PickM8", page_icon="⚙️", layout="wide")

def train_model(molecules_df, grades_df, model_config):
    progress_bar = st.progress(0)
    status = st.empty()
    
    status.text("Initializing model...")
    strategy = ActiveLearningStrategy(
        model_type=model_config['model_type'],
        **model_config.get('model_params', {})
    )
    progress_bar.progress(0.2)
    
    status.text("Training model...")
    train_info = strategy.train(molecules_df, grades_df)
    progress_bar.progress(0.6)
    
    status.text("Generating predictions...")
    predictions_df = strategy.predict(molecules_df)
    progress_bar.progress(0.8)
    
    status.text("Saving results...")
    data_handler = DataHandler(st.session_state.session_id)
    data_handler.save_predictions(predictions_df.to_dicts())
    
    model_path = data_handler.session_path / "model.pkl"
    strategy.model.save(model_path)
    progress_bar.progress(1.0)
    
    status.text("Training complete!")
    time.sleep(1)
    
    return train_info, predictions_df

def main():
    st.title("⚙️ Model Training & Settings")
    
    if 'session_id' not in st.session_state:
        st.error("No active session.")
        return
    
    data_handler = DataHandler(st.session_state.session_id)
    molecules_df = data_handler.load_molecules()
    grades_df = data_handler.load_grades()
    
    if grades_df is None or grades_df.is_empty():
        st.error("No grades available. Please grade some molecules first.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Train Model", "Model Settings", "Fingerprinting", "Custom Models"])
    
    with tab1:
        st.subheader("Model Training")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Graded Molecules", len(grades_df))
            st.metric("Total Molecules", len(molecules_df))
        
        with col2:
            grade_counts = grades_df.group_by('grade').count().sort('grade')
            st.bar_chart(grade_counts, x='grade', y='count', height=200)
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training in progress..."):
                model_config = {
                    'model_type': st.session_state.get('model_type', 'ensemble'),
                    'model_params': st.session_state.get('model_params', {})
                }
                
                train_info, predictions_df = train_model(molecules_df, grades_df, model_config)
                
                st.success("Model trained successfully!")
                st.json(train_info)
                
                st.session_state.model_trained = True
                
                if st.button("Go to Active Learning"):
                    st.switch_page("pages/3_🎯_Active_Learning.py")
    
    with tab2:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Model Type",
            ["ensemble", "random_forest", "svm"],
            index=0
        )
        st.session_state.model_type = model_type
        
        st.divider()
        
        if model_type == "ensemble":
            n_members = st.slider("Number of ensemble members", 2, 10, 3)
            base_model = st.selectbox("Base model", ["random_forest", "svm"])
            st.session_state.model_params = {
                'n_members': n_members,
                'base_model': base_model
            }
        
        elif model_type == "random_forest":
            n_estimators = st.slider("Number of trees", 50, 500, 100)
            st.session_state.model_params = {
                'n_estimators': n_estimators
            }
        
        elif model_type == "svm":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])
            C = st.slider("C parameter", 0.1, 10.0, 1.0)
            st.session_state.model_params = {
                'kernel': kernel,
                'C': C
            }
        
        if st.button("Save Configuration"):
            st.success("Configuration saved!")
    
    with tab3:
        st.subheader("Fingerprinting Configuration")
        
        # Load current config
        try:
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}
        
        fp_config = config.get('fingerprinting', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Interaction Fingerprints**")
            interaction_fp_type = st.selectbox(
                "Interaction Fingerprint Type",
                ["PLIP", "PLIF", "LUNA"],
                index=["PLIP", "PLIF", "LUNA"].index(fp_config.get('default_type', 'LUNA'))
            )
            
            st.info(f"""
            **{interaction_fp_type}** - {
                "Protein-Ligand Interaction Profiler" if interaction_fp_type == "PLIP" else
                "Protein-Ligand Interaction Fingerprints" if interaction_fp_type == "PLIF" else
                "LUNA interaction fingerprints"
            }
            """)
        
        with col2:
            st.write("**Molecule Fingerprints**")
            molecule_fp_type = st.selectbox(
                "Molecule Fingerprint Type",
                ["morgan", "rdkit"],
                index=["morgan", "rdkit"].index(fp_config.get('molecule_fp_type', 'morgan'))
            )
            
            if molecule_fp_type == "morgan":
                radius = st.slider("Morgan Radius", 1, 4, fp_config.get('molecule_fp_radius', 2))
                fp_size = st.selectbox("Fingerprint Size", [1024, 2048, 4096], 
                                     index=[1024, 2048, 4096].index(fp_config.get('molecule_fp_size', 2048)))
            else:
                radius = 2  # RDKit doesn't use radius
                fp_size = st.selectbox("Fingerprint Size", [1024, 2048, 4096],
                                     index=[1024, 2048, 4096].index(fp_config.get('molecule_fp_size', 2048)))
        
        if st.button("Save Fingerprinting Settings", type="primary"):
            # Update config
            if 'fingerprinting' not in config:
                config['fingerprinting'] = {}
            
            config['fingerprinting'].update({
                'default_type': interaction_fp_type,
                'molecule_fp_type': molecule_fp_type,
                'molecule_fp_size': fp_size,
                'molecule_fp_radius': radius
            })
            
            # Save to file
            with open("config.yaml", 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            st.success(f"✅ Fingerprinting settings saved! Using {interaction_fp_type} for interactions and {molecule_fp_type} for molecules.")
            st.session_state.fingerprint_config_changed = True
    
    with tab4:
        st.subheader("Custom Models")
        st.info("Upload a Python file containing a custom model class that inherits from BaseModel")
        
        uploaded_file = st.file_uploader("Choose a Python file", type=['py'])
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            st.code(content, language='python')
            
            if st.button("Load Custom Model"):
                model_path = data_handler.session_path / "custom_model.py"
                with open(model_path, 'w') as f:
                    f.write(content)
                
                st.success("Custom model loaded! You can now use it for training.")
                st.session_state.model_type = 'custom'
                st.session_state.custom_model_path = str(model_path)

if __name__ == "__main__":
    main()