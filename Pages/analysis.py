from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Load historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data.dropna(inplace=True)
    return data

def monte_carlo_simulation(last_price, mu, sigma, T, N, paths):
    dt = T / N
    prices = np.zeros((N + 1, paths))
    prices[0] = last_price
    for t in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return prices

def show_analysis():
    st.title("📊 Financial Analysis")
    st.markdown("Use this page to visualize historical data, assess volatility, and predict returns.")

    ticker = st.text_input("Enter stock ticker as apecified in Yahoo Finance (e.g., ^IXIC):", "^NSEI")
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365 * 10))
    end_date = st.date_input("End Date", datetime.today())
    st.markdown("**Note:** The date range is inclusive of the start and end dates.")

    if st.button("Run Analysis"):
        if not ticker:
            st.error("Please enter a valid stock ticker.")
            return
        data = load_data(ticker, start_date, end_date)
        if data.empty:
            st.error("No data found for the specified ticker and date range.")
            return
        data.dropna(inplace=True)
        data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)
        squared_returns = data['Return'] ** 2

        st.header("📈 Price & Return")
        fig1, ax1 = plt.subplots()
        ax1.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax1.set_ylabel('Price', color='blue')
        ax2 = ax1.twinx()
        ax2.plot(data.index, data['Return'], label='Log Returns', color='orange', alpha=0.6)
        ax2.set_ylabel('Return', color='orange')
        st.pyplot(fig1)
        st.markdown("**Note:** The blue line represents the closing price, while the orange line represents the log returns.")
        st.markdown("**Note:** Log returns are calculated as the natural logarithm of the ratio of consecutive closing prices.")
        
        st.divider()
        st.subheader("🔍 Time Series Diagnostics")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 ACF Plot")
            fig_acf, ax_acf = plt.subplots()
            plot_acf(squared_returns, ax=ax_acf, lags=25)
            ax_acf.set_title("ACF of Squared Returns")
            st.pyplot(fig_acf)
            st.markdown("**Note:** ACF plot shows the correlation of the series with its own lagged values.")

        with col2:
            st.subheader("📊 PACF Plot")
            fig_pacf, ax_pacf = plt.subplots()
            plot_pacf(squared_returns, ax=ax_pacf, lags=25, method='ywm')
            ax_pacf.set_title("PACF of Squared Returns")
            st.pyplot(fig_pacf)
            st.markdown("**Note:** PACF plot shows the correlation of the series with its own lagged values after removing the effects of intermediate lags.")
        st.markdown("**Note:** ACF and PACF plots help identify the order of ARIMA/SARIMA models.")
        st.markdown("**Note:** ACF shows the correlation of the series with its own lagged values, while PACF shows the correlation after removing the effects of intermediate lags.")

        