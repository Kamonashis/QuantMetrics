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

# Function to perform seasonal decomposition and create a Plotly figure
def plot_seasonal_decomposition_interactive(data, model='multiplicative', period=30):
    # Ensure the data used for decomposition has a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        st.error("Data for seasonal decomposition must have a DatetimeIndex.")
        return None, None # Return None if index is not DatetimeIndex

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
    st.subheader("博 Augmented Dickey-Fuller (ADF) Test on Residuals")
    if residuals is not None:
        # Ensure residuals are a pandas Series and drop NaNs before testing
        residuals_series = pd.Series(residuals).dropna()
        if not residuals_series.empty:
            adf_result = adfuller(residuals_series)
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
             st.warning("Residuals data is empty after dropping NaNs. Cannot perform ADF test.")
    else:
        st.warning("Residuals could not be calculated for the ADF test.")


# Main function
def show_analysis():
    st.title("投 Financial Analysis")
    st.markdown("Visualize historical data, assess volatility")

    # --- Input Section ---
    st.header("Input Parameters")
    # Use session state to persist input values
    if 'analysis_ticker' not in st.session_state:
        st.session_state['analysis_ticker'] = "^NSEI"
    ticker = st.text_input("Enter stock ticker (Yahoo format):", value=st.session_state['analysis_ticker'], key='analysis_ticker_widget')
    st.session_state['analysis_ticker'] = ticker # Update session state

    col1, col2 = st.columns(2)
    with col1:
        if 'analysis_start_date' not in st.session_state:
            st.session_state['analysis_start_date'] = datetime.today() - timedelta(days=365 * 10)
        start_date = st.date_input("Start Date", value=st.session_state['analysis_start_date'], key='analysis_start_date_widget')
        st.session_state['analysis_start_date'] = start_date # Update session state
    with col2:
        if 'analysis_end_date' not in st.session_state:
            st.session_state['analysis_end_date'] = datetime.today()
        end_date = st.date_input("End Date", value=st.session_state['analysis_end_date'], key='analysis_end_date_widget')
        st.session_state['analysis_end_date'] = end_date # Update session state

    col3, col4 = st.columns(2)
    with col3:
        if 'analysis_period' not in st.session_state:
            st.session_state['analysis_period'] = 30
        period = st.number_input("Seasonal Decomposition Period", min_value=1, value=st.session_state['analysis_period'], step=1, key='analysis_period_widget')
        st.session_state['analysis_period'] = period # Update session state
    with col4:
        if 'analysis_model' not in st.session_state:
            st.session_state['analysis_model'] = "Additive"
        model = st.selectbox("Decomposition Model", ["Additive", "Multiplicative"], index=["Additive", "Multiplicative"].index(st.session_state['analysis_model']), key='analysis_model_widget')
        st.session_state['analysis_model'] = model # Update session state


    # --- Run Analysis Button ---
    if st.button("Run Analysis", key='run_analysis_button'):
        if not ticker:
            st.error("Please enter a valid ticker.")
            # Clear previous results if button is clicked with no ticker
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']
            return

        try:
            # Load data
            data = load_data(ticker, start_date, end_date)
            if data.empty:
                st.error("No data retrieved.")
                if 'analysis_results' in st.session_state:
                    del st.session_state['analysis_results']
                return

            # --- Store the original data with DatetimeIndex before resetting index ---
            original_data_with_datetime_index = data.copy()
            # --- End of storing original data ---

            # Process data (calculate returns and reset index for display convenience)
            data['Return'] = data['Close'].pct_change()
            data['Return'] = data['Return'].replace([np.inf, -np.inf], np.nan)
            data['Return'] = data['Return'].fillna(0)
            data.dropna(inplace=True)
            data.reset_index(inplace=True) # Reset index for easier plotting with Streamlit's line_chart

            # Perform seasonal decomposition using the original data with DatetimeIndex
            decomposition_result, interactive_decomp_fig = plot_seasonal_decomposition_interactive(original_data_with_datetime_index, model=model.lower(), period=period)

            # Store results in session state
            st.session_state['analysis_results'] = {
                'data': data, # Data with integer index (for line_chart)
                'original_data_with_datetime_index': original_data_with_datetime_index, # Original data with DatetimeIndex
                'returns': data['Return'], # Returns (from data with integer index)
                'ticker': ticker,
                'decomposition_result': decomposition_result,
                'interactive_decomp_fig': interactive_decomp_fig
            }
            st.success("Analysis complete!")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            # Clear results on error
            if 'analysis_results' in st.session_state:
                del st.session_state['analysis_results']

    # --- Display Results (if available in session state) ---
    # Check if analysis_results exists and is not empty
    if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
        results = st.session_state['analysis_results']

        # --- Robustly retrieve data from the results dictionary ---
        data = results.get('data') # Data with integer index
        original_data_with_datetime_index = results.get('original_data_with_datetime_index') # Original data with DatetimeIndex
        returns = results.get('returns')
        ticker = results.get('ticker')
        decomposition_result = results.get('decomposition_result')
        interactive_decomp_fig = results.get('interactive_decomp_fig')

        # Check if all essential data is present
        if data is None or original_data_with_datetime_index is None or returns is None or ticker is None or decomposition_result is None or interactive_decomp_fig is None:
             st.warning("Incomplete analysis results found in session state. Please run the analysis again to ensure all data is loaded.")
             # Optionally clear the incomplete state to prevent future errors
             # del st.session_state['analysis_results']
             return
        # --- End of robust retrieval ---


        st.subheader("嶋 Price and Returns")
        # Ensure 'Date' column exists in data before accessing min/max
        if 'Date' in data.columns:
            st.write(f"Data from {data['Date'].min().date()} to {data['Date'].max().date()}")
        else:
             st.write("Data date range information not available.")

        st.write(f"Data shape: {data.shape}")
        st.line_chart(data['Close'], x_label='Date', use_container_width=True)
        st.write("Daily Returns")
        st.line_chart(returns, x_label='Date', use_container_width=True)

        st.subheader("投 ACF and PACF of Squared Returns")
        squared_returns = returns ** 2
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

        st.subheader("悼 Interactive Seasonal Decomposition")
        st.write("Decomposing the time series into trend, seasonal, and residual components.")
        st.plotly_chart(interactive_decomp_fig, use_container_width=True)
        st.write("The seasonal decomposition helps to understand the underlying patterns in the data.")
        st.write("Trend: Long-term movement in the data.")
        st.write("Seasonal: Regular pattern that repeats over a fixed period.")
        st.write("Residual: The remaining variation after removing trend and seasonal components.")

        # Perform ADF test on residuals using the stored decomposition result
        perform_adf_test(decomposition_result.resid)