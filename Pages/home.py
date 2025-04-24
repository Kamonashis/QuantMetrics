import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="QuantMetrics", layout="wide")

def show_home():
    st.title("📊 QuantMetrics")
    st.markdown("""
    <h4 style='color:#4F8BF9;'>Empowering Quantitative Insights, One Metric at a Time</h4>
    """, unsafe_allow_html=True)

    st.markdown("""
    Welcome to **QuantMetrics**, a powerful, multi-page analytics platform designed for quant finance enthusiasts, students, and professionals.
        
    Explore market behavior, analyze volatility, perform regression diagnostics, and investigate relationships between financial assets.

    #### 🚀 Features:
    - GARCH and EWMA volatility modeling with forecasting
    - Correlation matrix analysis for custom portfolios
    - Regression analysis with robust diagnostics and rolling predictions
    - Interactive charts and intuitive layout
    """)
    df = pd.DataFrame(np.random.randn(150, 3), columns=(["A", "B", "C"]))
    my_data_element = st.line_chart(df, use_container_width=True, height=300)
    st.markdown("---")

    for tick in range(10):
        time.sleep(.5)
        add_df = pd.DataFrame(np.random.randn(1, 3), columns=(["A", "B", "C"]))
        my_data_element.add_rows(add_df)
    st.button("Regenerate")

    st.markdown("---")
    st.markdown(f"📅 *Session initialized: {datetime.now().strftime('%B %d, %Y – %H:%M:%S')}*")
    st.markdown("<small>Developed with ❤️ for quant learners & professionals</small>", unsafe_allow_html=True)