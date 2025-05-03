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
    - **Financial Analysis:** Visualize historical price and returns data, analyze time series properties with ACF/PACF and Seasonal Decomposition, and test for stationarity using the ADF test.
    - **Volatility Modeling:** Model volatility using GARCH, EGARCH, HARCH, and RiskMetrics (EWMA) models. Compare volatility estimates and forecast future volatility.
    - **Correlation Analysis:** Calculate and visualize the static correlation matrix between multiple assets based on either Close Prices or Returns, using Pearson, Spearman, or Kendall methods. View p-values for Pearson correlation and scatter plots for ticker pairs.
    - **Regression Analysis:** Perform regression analysis on asset returns using Standard OLS, Robust Regression, or Rolling Regression. Analyze model summaries, check for multicollinearity with VIF, examine residuals, and forecast the next day's return.
    """)

    # Keeping the random number generator example as it was in the original home.py
    st.markdown("---")
    st.subheader("Live Data Example")
    st.markdown("Below is an example of a live updating chart using random numbers.")

    df = pd.DataFrame(np.random.randn(150, 3), columns=(["A", "B", "C"]))
    my_data_element = st.line_chart(df)

    # This loop will cause the page to rerun and add rows if not commented out or handled differently
    # For a simple home page, you might prefer to just display a static chart or
    # handle the update with a button click and session state if needed.
    # Keeping the original loop structure for now, but be aware it causes reruns.
    # for tick in range(10):
    #     time.sleep(1.5)
    #     add_df = pd.DataFrame(np.random.randn(1, 3), columns=(["A", "B", "C"]))
    #     my_data_element.add_rows(add_df)

    # A button to trigger a manual update of the random data
    if st.button("Regenerate Random Data"):
        df = pd.DataFrame(np.random.randn(150, 3), columns=(["A", "B", "C"]))
        my_data_element.line_chart(df) # Replace the chart with new data


    st.markdown("---")
    st.markdown(f"📅 *Session initialized: {datetime.now().strftime('%B %d, %Y – %H:%M:%S')}*")
    st.markdown("<small>Developed with ❤️ for quant learners & professionals</small>", unsafe_allow_html=True)