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