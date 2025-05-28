import streamlit as st
from pages.home import show_home # Updated import
from pages.analysis import show_analysis # Updated import
from pages.modelling import show_modeling # Updated import
from pages.correlation import show_correlation # Updated import
from pages.regression import show_regression # Updated import

# Sidebar navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Analysis", "🔍 Modeling", "📈 Correlation Analysis", "📉 Regression Analysis"])

if page == "🏠 Home":
    show_home()
elif page == "📊 Analysis":
    show_analysis()
elif page == "🔍 Modeling":
    show_modeling()
elif page == "📈 Correlation Analysis":
    show_correlation()
elif page == "📉 Regression Analysis":
    show_regression()