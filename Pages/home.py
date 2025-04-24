import streamlit as st
from datetime import datetime

st.set_page_config(page_title="QuantMetrics", layout="wide")

def show_home():
    st.title("📊 QuantMetrics")
    st.markdown("""
    <h4 style='color:#4F8BF9;'>Empowering Quantitative Insights, One Metric at a Time</h4>
    """, unsafe_allow_html=True)

    st.image("mlreg_model.jpg", width=100)

    st.markdown("""
    Welcome to **QuantMetrics**, a powerful, multi-page analytics platform designed for quant finance enthusiasts, students, and professionals.
        
    Explore market behavior, analyze volatility, perform regression diagnostics, and investigate relationships between financial assets.

    #### 🚀 Features:
    - GARCH and EWMA volatility modeling with forecasting
    - Correlation matrix analysis for custom portfolios
    - Regression analysis with robust diagnostics and rolling predictions
    - Interactive charts and intuitive layout
    """)

    st.markdown("---")
    st.markdown(f"📅 *Session initialized: {datetime.now().strftime('%B %d, %Y – %H:%M:%S')}*")
    st.markdown("<small>Developed with ❤️ for quant learners & professionals</small>", unsafe_allow_html=True)