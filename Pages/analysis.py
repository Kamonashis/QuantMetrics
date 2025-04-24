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

def monte_carlo_simulation(last_price, mu, sigma, T, N, paths):
    dt = T / N
    prices = np.zeros((N + 1, paths))
    prices[0] = last_price
    for t in range(1, N + 1):
        Z = np.random.standard_normal(paths)
        prices[t] = prices[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    return prices

def show_analysis():
    st.set_page_config(page_title="QuantMetrics - Analysis", layout="wide")

    st.title("📊 Financial Analysis")
    st.markdown("Use this page to visualize historical data, assess volatility, and predict returns.")

    ticker = st.text_input("Enter stock ticker (e.g., AAPL):", "AAPL")
    start_date = datetime.today() - timedelta(days=365 * 3)
    end_date = datetime.today()

    if ticker:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.dropna(inplace=True)
        data['Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Price & Return")
            fig1, ax1 = plt.subplots()
            ax1.plot(data.index, data['Adj Close'], label='Adj Close Price', color='blue')
            ax1.set_ylabel('Price', color='blue')
            ax2 = ax1.twinx()
            ax2.plot(data.index, data['Return'], label='Log Returns', color='orange', alpha=0.6)
            ax2.set_ylabel('Return', color='orange')
            st.pyplot(fig1)

        with col2:
            st.subheader("📉 Rolling Volatility")
            rolling_vol = data['Return'].rolling(window=21).std() * np.sqrt(252)
            fig2, ax = plt.subplots()
            ax.plot(data.index, rolling_vol, label='Rolling Volatility (21d)', color='red')
            ax.set_ylabel('Annualized Volatility')
            st.pyplot(fig2)

        st.divider()
        st.subheader("🔍 Time Series Diagnostics")

        fig_acf, ax_acf = plt.subplots()
        plot_acf(data['Return'], ax=ax_acf, lags=40)
        st.pyplot(fig_acf)

        fig_pacf, ax_pacf = plt.subplots()
        plot_pacf(data['Return'], ax=ax_pacf, lags=40, method='ywm')
        st.pyplot(fig_pacf)

        st.divider()
        st.subheader("🔮 Return Prediction via Monte Carlo Simulation")

        model_choice = st.selectbox("Choose Forecasting Model", ["ARIMA", "SARIMA", "Exponential Smoothing"])
        n_days = st.slider("Forecast Horizon (days)", 10, 100, 30)
        n_paths = st.slider("Monte Carlo Paths", 100, 2000, 500)

        train_data = data['Return'].dropna()

        if model_choice == "ARIMA":
            model = ARIMA(train_data, order=(1, 0, 1)).fit()
        elif model_choice == "SARIMA":
            model = SARIMAX(train_data, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12)).fit()
        else:
            model = ExponentialSmoothing(train_data, trend='add', seasonal=None).fit()

        forecast_returns = model.forecast(n_days)
        mu = forecast_returns.mean()
        sigma = forecast_returns.std()
        last_price = data['Adj Close'][-1]

        st.write(f"**Forecast Mean Return:** `{mu:.5f}` | **Std Dev:** `{sigma:.5f}`")

        sim_data = monte_carlo_simulation(last_price, mu, sigma, T=n_days / 252, N=n_days, paths=n_paths)

        st.subheader("📊 Monte Carlo Forecast: Fan Chart")
        fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
        ax_mc.plot(sim_data, color='lightgray', alpha=0.1)
        ax_mc.plot(sim_data.mean(axis=1), color='blue', label='Mean Forecast')
        ax_mc.fill_between(range(n_days + 1),
                           np.percentile(sim_data, 5, axis=1),
                           np.percentile(sim_data, 95, axis=1),
                           color='skyblue', alpha=0.3, label='90% CI')
        ax_mc.set_title("Simulated Price Paths")
        ax_mc.legend()
        st.pyplot(fig_mc)

        st.subheader("📈 Histogram of Simulated Returns")
        final_returns = np.log(sim_data[-1] / last_price)
        fig_hist, ax_hist = plt.subplots()
        ax_hist.hist(final_returns, bins=50, color='purple', alpha=0.7)
        ax_hist.set_title("Distribution of Final Simulated Returns")
        st.pyplot(fig_hist)

        st.subheader("🔁 Backtest Model Prediction vs Actual")
        forecast_backtest = model.get_prediction(start=-n_days)
        pred_mean = forecast_backtest.predicted_mean
        true_vals = train_data[-n_days:]

        fig_bt, ax_bt = plt.subplots()
        ax_bt.plot(true_vals.index, true_vals.values, label="Actual", color='black')
        ax_bt.plot(pred_mean.index, pred_mean.values, label="Predicted", color='green')
        ax_bt.set_title("Backtest: Actual vs Predicted Returns")
        ax_bt.legend()
        st.pyplot(fig_bt)