from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
import warnings

# Data loader
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

# Main function
def show_analysis():
    st.title("📊 Financial Analysis")
    st.markdown("Visualize historical data, assess volatility")

    ticker = st.text_input("Enter stock ticker (Yahoo format):", "^NSEI")
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365 * 10))
    end_date = st.date_input("End Date", datetime.today())
    period = st.number_input("Seasonal Decomposition Period", min_value=1, value=30, step=1)
    model = st.selectbox("Decomposition Model", ["Additive", "Multiplicative"])

    # Function to perform seasonal decomposition and create a Plotly figure
    def plot_seasonal_decomposition_interactive(data, model='multiplicative', period=30):
        decomposition = seasonal_decompose(data['Close'], model=model, period=period, extrapolate_trend='freq')
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))

        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'), row=4, col=1)

        fig.update_layout(title='Seasonal Decomposition', height=2000)
        return decomposition, fig
        
    # Function to perform ADF test on residuals
    def perform_adf_test(residuals):
        st.subheader("🔎 Augmented Dickey-Fuller (ADF) Test on Residuals")
        if residuals is not None:
            adf_result = adfuller(residuals.dropna())
            st.write(f"ADF Statistic: {adf_result[0]:.3f}")
            st.write(f"p-value: {adf_result[1]:.3f}")
            st.write("Critical Values:")
            for key, value in adf_result[4].items():
                st.write(f"   {key}: {value:.3f}")

            if adf_result[1] <= 0.05:
                st.success("The residuals are likely stationary (p-value <= 0.05).")
            else:
                st.warning("The residuals are likely non-stationary (p-value > 0.05).")
        else:
            st.warning("Residuals could not be calculated for the ADF test.")

    if st.button("Run Analysis"):
        if not ticker:
            st.error("Please enter a valid ticker.")
            return

        data = load_data(ticker, start_date, end_date)
        if data.empty:
            st.error("No data retrieved.")
            return

        data['Return'] = data['Close'].pct_change()  # Calculate daily returns
        data['Return'] = data['Return'].replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
        data['Return'] = data['Return'].fillna(0)  # Fill NaN values with 0 for returns
        data.dropna(inplace=True)

        st.subheader("📈 Price and Returns")
        st.write(f"Data from {start_date} to {end_date}")
        st.write(f"Data shape: {data.shape}")
        st.line_chart(data['Close'], x_label='Date', use_container_width=True)
        st.write("Daily Returns")
        st.line_chart(data['Return'], x_label='Date', use_container_width=True)

        st.subheader("📊 ACF and PACF of Squared Returns")
        squared_returns = data['Return'] ** 2
        col1, col2 = st.columns(2)
        with col1:
            fig_acf, ax_acf = plt.subplots()
            plot_acf(squared_returns, ax=ax_acf, lags=25)
            ax_acf.set_title("ACF of Squared Returns")
            st.pyplot(fig_acf)
        with col2:
            fig_pacf, ax_pacf = plt.subplots()
            plot_pacf(squared_returns, ax=ax_pacf, lags=25, method='ywm')
            ax_pacf.set_title("PACF of Squared Returns")
            st.pyplot(fig_pacf)
        st.write("ACF and PACF plots help identify the order of ARIMA models.")
        
        st.subheader("📉 Interactive Seasonal Decomposition")
        st.write("Decomposing the time series into trend, seasonal, and residual components.")
        decomposition_result, interactive_decomp_fig = plot_seasonal_decomposition_interactive(data)
        st.plotly_chart(interactive_decomp_fig, use_container_width=True)
        st.write("The seasonal decomposition helps to understand the underlying patterns in the data.")
        st.write("Trend: Long-term movement in the data.") 
        st.write("Seasonal: Regular pattern that repeats over a fixed period.")
        st.write("Residual: The remaining variation after removing trend and seasonal components.")

        # Perform ADF test on residuals
        perform_adf_test(decomposition_result.resid)