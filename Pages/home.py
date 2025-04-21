import streamlit as st

st.set_page_config(page_title="Stock Analysis App", layout="wide")
st.title("📈 Stock Analysis App")

def show_home():
    st.title("Welcome to the Stock Volatility App")
    st.write("Use the sidebar to navigate through the analysis, modeling, and forecasting tools.")
    st.write("This app allows you to analyze stock returns, model volatility using GARCH and EWMA methods, and visualize the results.")
    st.write("You can also explore the correlation matrix of multiple stocks.")