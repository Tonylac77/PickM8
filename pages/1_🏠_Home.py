import streamlit as st
from utils.io_handlers import DataHandler
import polars as pl

st.set_page_config(page_title="Home - PickM8", page_icon="🏠", layout="wide")

def main():
    st.title("🏠 Home")
    
    if 'session_id' not in st.session_state:
        st.error("No active session. Please start from the main page.")
        return
    
    data_handler = DataHandler(st.session_state.session_id)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Session Status")
        
        molecules_df = data_handler.load_molecules()
        grades_df = data_handler.load_grades()
        predictions_df = data_handler.load_predictions()
        
        st.metric("Total Molecules", len(molecules_df) if molecules_df else 0)
        st.metric("Graded Molecules", len(grades_df) if grades_df else 0)
        st.metric("Predictions Available", len(predictions_df) if predictions_df else 0)
        
        if st.button("Clear Session", type="secondary"):
            st.session_state.clear()
            st.rerun()
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("📤 Upload New Screen", type="primary"):
            st.switch_page("pages/2_📤_Upload_Screen.py")
        
        if molecules_df and not molecules_df.is_empty():
            if st.button("🎯 Continue Grading", type="primary"):
                st.switch_page("pages/3_🎯_Active_Learning.py")
        
        if grades_df and not grades_df.is_empty():
            if st.button("📊 View Results", type="primary"):
                st.switch_page("pages/4_📊_Results.py")
    
    if grades_df and not grades_df.is_empty():
        st.subheader("Grade Distribution")
        grade_counts = grades_df.group_by('grade').count().sort('grade')
        st.bar_chart(grade_counts, x='grade', y='count')

if __name__ == "__main__":
    main()