import streamlit as st
from home import show_home
from analysis import show_analysis
from modelling import show_modeling
from correlation import show_correlation  # Optional

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Analysis", "🔍 Modeling", "📈 Correlation Matrix"])

if page == "🏠 Home":
    show_home()
elif page == "📊 Analysis":
    show_analysis()
elif page == "🔍 Modeling":
    show_modeling()
elif page == "📈 Correlation Matrix":
    show_correlation()