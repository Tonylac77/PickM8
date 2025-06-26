"""PickM8 Main Application - Landing Page"""
import streamlit as st

st.set_page_config(
    page_title="PickM8 - Molecular Screening",
    page_icon="media/pickm8_white_logoonly.png", 
    layout="wide"
)

def main():
    """Main landing page."""
    # Add logo to app and sidebar
    st.logo(
        image="media/pickm8_white.png",
        size="large",
        icon_image="media/pickm8_white_logoonly.png"
    )
    
    # Welcome page
    st.title("PickM8 - Molecular Screening")
    
    st.markdown("""
    Welcome to **PickM8**, a powerful molecular screening and active learning platform for drug discovery.
    
    ## Features
    - 🔬 **Molecular Analysis**: Compute fingerprints and analyze protein-ligand interactions
    - 🎯 **Active Learning**: Train ML models to guide compound selection
    - 📊 **Results Visualization**: Interactive plots and molecular viewers
    - 💾 **Session Management**: Save and resume your screening workflows
    
    ## Getting Started
    1. **Setup**: Create a new session or load an existing one
    2. **Active Learning**: Train models and select compounds for validation
    3. **Results**: Analyze your screening results and export data
    """)
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Setup Session", use_container_width=True):
            st.switch_page("pages/1_🔧_Setup.py")
    
    with col2:
        if st.button("🎯 Active Learning", use_container_width=True):
            st.switch_page("pages/3_🎯_Active_Learning.py")
    
    with col3:
        if st.button("📊 View Results", use_container_width=True):
            st.switch_page("pages/4_📊_Results.py")
    
    # Quick stats if session exists
    if 'session_id' in st.session_state and st.session_state.session_id:
        st.markdown("---")
        st.subheader("Current Session")
        
        if 'molecules_df' in st.session_state:
            df = st.session_state.molecules_df
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Molecules", len(df))
            with col2:
                graded = df['grade'].notna().sum() if 'grade' in df.columns else 0
                st.metric("Graded Molecules", graded)
            with col3:
                predicted = df['prediction'].notna().sum() if 'prediction' in df.columns else 0
                st.metric("Predicted Molecules", predicted)

if __name__ == "__main__":
    main()