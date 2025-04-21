import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

def show_analysis():
    st.title("📊 Stock Return Analysis")

    ticker = st.sidebar.text_input("Ticker Symbol", "^NSEI").upper()
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=3650))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    if st.sidebar.button("Fetch Data"):
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data retrieved.")
            return

        data['Return'] = data['Close'].pct_change() * 100
        returns = data['Return'].dropna()
        returns = returns[returns != 0]
        returns = returns[returns != np.inf]
        st.subheader(f"Data for **{ticker}** from {start_date} to {end_date}")
        st.line_chart(data['Close'])
        st.subheader("Returns")
        st.write(f"Mean Return: {returns.mean():.2f}%")
        st.line_chart(returns)

        st.subheader("ACF of Squared Returns")
        st.write("The ACF of squared returns can help identify the presence of volatility clustering.")
        st.write("A significant spike at lag 1 indicates volatility clustering, which is common in financial time series.")
        fig1, ax1 = plt.subplots()
        plot_acf(returns**2, ax=ax1, lags=25)
        st.pyplot(fig1)

        st.subheader("PACF of Squared Returns")
        st.write("The PACF of squared returns can help identify the order of GARCH models.")
        st.write("A significant spike at lag 1 indicates the presence of ARCH effects.")
        st.write("This can help in determining the order of the GARCH model.")
        fig2, ax2 = plt.subplots()
        plot_pacf(returns**2, ax=ax2, lags=25)
        st.pyplot(fig2)

        st.session_state['returns'] = returns
        st.session_state['ticker'] = ticker
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date