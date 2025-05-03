import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from datetime import datetime, timedelta # Import timedelta for date calculations

def show_modeling():
    st.title("📊 GARCH & RiskMetrics (EWMA) Modeling")

    # Check if analysis results are available from the analysis page
    if 'analysis_results' not in st.session_state or 'returns' not in st.session_state['analysis_results']:
        st.warning("No return data found. Please run the analysis first.")
        # Provide a button to go back to analysis if needed (optional, as navigation handles this)
        # if st.button("Back to Analysis"):
        #     st.session_state.page = "Analysis" # Assuming your navigation uses this key
        #     st.rerun()
        return

    # Retrieve returns and ticker from analysis results in session state
    returns = st.session_state['analysis_results']['returns']
    ticker = st.session_state['analysis_results']['ticker']
    # Retrieve the original data DataFrame for accessing the DatetimeIndex
    original_data = st.session_state['analysis_results'].get('data')

    if original_data is None or original_data.empty:
        st.warning("Original data not found or is empty in analysis results. Cannot perform modeling.")
        return


    st.write(f"Modeling volatility for **{ticker}**")

    # --- Input Section (Moved from Sidebar and Persisted) ---
    st.header("Model Parameters")

    st.subheader("GARCH Parameters")
    col1, col2 = st.columns(2)
    with col1:
        if 'garch_p' not in st.session_state:
            st.session_state['garch_p'] = 1
        # Changed min_value to 0 for p and q as GARCH(0,0) is white noise
        garch_p = st.slider("AR term (p)", 0, 15, value=st.session_state['garch_p'], key='garch_p_widget')
        st.session_state['garch_p'] = garch_p # Update session state

        if 'vol_type' not in st.session_state:
            st.session_state['vol_type'] = "GARCH"
        vol_type = st.selectbox("Volatility Model", ["GARCH", "EGARCH", "HARCH"], index=["GARCH", "EGARCH", "HARCH"].index(st.session_state['vol_type']), key='vol_type_widget')
        st.session_state['vol_type'] = vol_type # Update session state

    with col2:
        if 'garch_q' not in st.session_state:
            st.session_state['garch_q'] = 1
        # Changed min_value to 0 for p and q
        garch_q = st.slider("MA term (q)", 0, 15, value=st.session_state['garch_q'], key='garch_q_widget')
        st.session_state['garch_q'] = garch_q # Update session state

        if 'mean_type' not in st.session_state:
            st.session_state['mean_type'] = "Constant"
        mean_type = st.selectbox("Mean Model", ["Constant", "AR", "Zero Mean"], index=["Constant", "AR", "Zero Mean"].index(st.session_state['mean_type']), key='mean_type_widget')
        st.session_state['mean_type'] = mean_type # Update session state

    st.subheader("EWMA Parameters")
    if 'lambda_' not in st.session_state:
        st.session_state['lambda_'] = 0.94
    lambda_ = st.slider("EWMA Lambda (λ)", min_value=0.85, max_value=0.99, value=st.session_state['lambda_'], step=0.01, key='lambda_widget')
    st.session_state['lambda_'] = lambda_ # Update session state

    # Sidebar toggle for annualization (kept in sidebar as it's a display option)
    st.sidebar.header("Display Options")
    if 'annualize_volatility' not in st.session_state:
        st.session_state['annualize_volatility'] = True
    annualize = st.sidebar.toggle("📅 Annualize Volatility", value=st.session_state['annualize_volatility'], key='annualize_volatility_widget')
    st.session_state['annualize_volatility'] = annualize # Update session state


    # --- Run Models Button ---
    if st.button("🔍 Run Volatility Models", key='run_models_button'):
        try:
            # --- GARCH ---
            mean = mean_type.lower() if mean_type != "Zero Mean" else "zero"
            # Ensure returns data is clean for GARCH
            clean_returns = returns.dropna().copy()
            clean_returns = clean_returns[clean_returns != 0]  # Remove zero returns

            if clean_returns.empty:
                 st.error("Cleaned returns data is empty. Cannot fit GARCH model.")
                 # Clear previous modeling results on error
                 if 'modeling_results' in st.session_state:
                     del st.session_state['modeling_results']
                 return

            # Check if p and q are both zero, which is not supported by arch_model directly for volatility
            if garch_p == 0 and garch_q == 0:
                 st.error("GARCH(0,0) is not a valid volatility model. Please set p or q to a value greater than 0.")
                 if 'modeling_results' in st.session_state:
                     del st.session_state['modeling_results']
                 return


            garch_model = arch_model(clean_returns, mean=mean, vol=vol_type.upper(), p=garch_p, q=garch_q, rescale=True)
            garch_fit = garch_model.fit(disp='off')

            st.success("✅ GARCH model fitted successfully.")
            st.text("Model Summary:")
            # Display summary using st.text or st.write
            st.text(garch_fit.summary())

            # --- EWMA ---
            # Ensure returns data is clean for EWMA
            clean_returns_ewma = returns.dropna().copy()
            if clean_returns_ewma.empty:
                 st.error("Cleaned returns data is empty. Cannot calculate EWMA volatility.")
                 # Clear previous modeling results on error
                 if 'modeling_results' in st.session_state:
                     del st.session_state['modeling_results']
                 return

            # Calculate EWMA volatility (unscaled initially)
            ewma_vol_unscaled = clean_returns_ewma.ewm(span=(2 / (1 - lambda_) - 1)).std()

            st.subheader("📉 EWMA Volatility Estimate (Unscaled)")
            st.markdown(f"""
                - **Mean Volatility:** {ewma_vol_unscaled.mean():.4f}
                - **Latest Volatility:** {ewma_vol_unscaled.iloc[-1]:.4f}
                - **Max Volatility:** {ewma_vol_unscaled.max():.4f}
                - **Min Volatility:** {ewma_vol_unscaled.min():.4f}
            """)

            # --- Calculate Volatilities for Plotting ---
            # GARCH conditional volatility (unscaled initially)
            garch_cond_vol_unscaled = garch_fit.conditional_volatility

            # Historical volatility (using a rolling window, e.g., 30 days)
            hist_vol_unscaled = clean_returns_ewma.rolling(window=30).std().dropna()


            # Store modeling results in session state
            st.session_state['modeling_results'] = {
                'garch_fit': garch_fit,
                'ewma_vol_unscaled': ewma_vol_unscaled,
                'garch_cond_vol_unscaled': garch_cond_vol_unscaled,
                'hist_vol_unscaled': hist_vol_unscaled,
                'lambda_': lambda_, # Store lambda for forecasting
                'processed_data': clean_returns_ewma # Store processed_data (clean returns) for forecasting date index
            }
            st.success("Volatility models fitted.")

        except Exception as e:
            st.error(f"Modeling failed: {e}")
            # Clear previous modeling results on error
            if 'modeling_results' in st.session_state:
                del st.session_state['modeling_results']


    # --- Display Modeling Results (if available in session state) ---
    if 'modeling_results' in st.session_state:
        results = st.session_state['modeling_results']
        garch_fit = results.get('garch_fit')
        ewma_vol_unscaled = results.get('ewma_vol_unscaled')
        garch_cond_vol_unscaled = results.get('garch_cond_vol_unscaled')
        hist_vol_unscaled = results.get('hist_vol_unscaled')
        lambda_ = results.get('lambda_') # Retrieve lambda
        processed_data = results.get('processed_data') # Retrieve processed_data

        # Check if essential modeling results are present
        if garch_fit is None or ewma_vol_unscaled is None or garch_cond_vol_unscaled is None or hist_vol_unscaled is None or lambda_ is None or processed_data is None:
             st.warning("Incomplete modeling results found in session state. Please run the volatility models again.")
             return


        st.markdown("----")
        st.subheader("📉 Volatility Comparison (GARCH vs EWMA vs Historical)")

        # Apply annualization based on the sidebar toggle state
        scaling_factor = np.sqrt(252) if st.session_state.get('annualize_volatility', True) else 1 # Default to True if not set

        garch_cond_vol_scaled = garch_cond_vol_unscaled * scaling_factor
        ewma_vol_scaled = ewma_vol_unscaled * scaling_factor
        hist_vol_scaled = hist_vol_unscaled * scaling_factor

        # Align lengths for plotting
        min_len = min(len(garch_cond_vol_scaled), len(ewma_vol_scaled), len(hist_vol_scaled))

        garch_cond_vol_scaled = garch_cond_vol_scaled[-min_len:]
        ewma_vol_scaled = ewma_vol_scaled[-min_len:]
        hist_vol_scaled = hist_vol_scaled[-min_len:]


        # Plot
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(garch_cond_vol_scaled.index, garch_cond_vol_scaled, label="GARCH Volatility", color="blue")
        ax.plot(ewma_vol_scaled.index, ewma_vol_scaled, label="EWMA Volatility", color="green", linestyle="--")
        ax.plot(hist_vol_scaled.index, hist_vol_scaled, label="30-Day Historical Volatility", color="black", linestyle=":")
        ax.set_title("Volatility Estimates: GARCH vs EWMA vs Historical")
        ylabel = "Annualized Volatility" if st.session_state.get('annualize_volatility', True) else "Daily Volatility"
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


        # --- FORECASTING SECTION ---
        st.markdown("----")
        st.subheader("🔮 Forecast Volatility")

        # Use session state for number of forecast days
        if 'n_days_forecast' not in st.session_state:
            st.session_state['n_days_forecast'] = 5
        n_days = st.number_input("Number of days to forecast", min_value=1, max_value=100, value=st.session_state['n_days_forecast'], key='n_days_forecast_widget')
        st.session_state['n_days_forecast'] = n_days # Update session state


        if st.button("📅 Forecast Volatility", key='run_forecast_button'):
            try:
                # --- GARCH Forecast ---
                garch_forecast = garch_fit.forecast(horizon=n_days)
                garch_var = garch_forecast.variance.values[-1]
                # Scale GARCH forecast based on annualization toggle
                garch_vol = np.sqrt(garch_var) * scaling_factor
                garch_vol_series = pd.Series(garch_vol, name="GARCH Forecast")

                # ±1 Std Dev Confidence Band (scaled)
                # The forecast variance already includes the model's estimated variance.
                # For a simple confidence band, we can use the standard deviation of the forecast volatility itself,
                # but a more rigorous approach involves the forecast distribution.
                # For simplicity here, we'll use a basic band based on the forecast values.
                # Note: This is a simplification and not a true confidence interval from the GARCH forecast distribution.
                std_dev_forecast = garch_vol_series.std() if len(garch_vol_series) > 1 else 0
                garch_upper = pd.Series(garch_vol + std_dev_forecast, name="GARCH Upper")
                garch_lower = pd.Series(garch_vol - std_dev_forecast, name="GARCH Lower")


                # --- EWMA Forecast ---
                # Retrieve the last unscaled EWMA variance
                last_ewma_var_unscaled = ewma_vol_unscaled.iloc[-1]**2
                # Retrieve the last squared return (unscaled)
                # Ensure returns data is available and not empty
                if returns.dropna().empty:
                     st.error("Returns data is empty. Cannot perform EWMA forecast.")
                     if 'forecast_results' in st.session_state:
                         del st.session_state['forecast_results']
                     return
                last_ret_squared_unscaled = returns.dropna().iloc[-1] ** 2


                # Initialize forecast array for unscaled variance
                ewma_forecast_var_unscaled = np.zeros(n_days)
                # The first forecast step uses the last observed variance
                ewma_forecast_var_unscaled[0] = last_ewma_var_unscaled

                # Recursive forecast of unscaled variance
                for t in range(1, n_days):
                    # The EWMA forecast uses the previous forecast variance and the last observed squared return
                    ewma_forecast_var_unscaled[t] = lambda_ * ewma_forecast_var_unscaled[t - 1] + (1 - lambda_) * last_ret_squared_unscaled

                # Convert back to standard deviation and apply scaling
                ewma_forecast_scaled = np.sqrt(ewma_forecast_var_unscaled) * scaling_factor
                ewma_vol_series = pd.Series(ewma_forecast_scaled, name="EWMA Forecast")

                # Create forecast DataFrame
                forecast_df = pd.concat([garch_vol_series, ewma_vol_series, garch_upper, garch_lower], axis=1)

                # --- FIX: Get the last date from the original_data DataFrame's index ---
                if original_data is None or original_data.empty:
                     st.error("Original data not available to set forecast dates.")
                     if 'forecast_results' in st.session_state:
                         del st.session_state['forecast_results']
                     return
                last_data_date = original_data.index[-1]
                # --- End of FIX ---

                # Set index for the forecast period (starting from the day after the last data point)
                forecast_dates = pd.date_range(start=(last_data_date + timedelta(days=1)), periods=n_days, freq='B') # 'B' for business days
                forecast_df.index = forecast_dates


                st.dataframe(forecast_df.style.format("{:.4%}"))

                # --- Plot Forecast ---
                st.subheader("Volatility Forecast (GARCH vs EWMA)")
                fig, ax = plt.subplots(figsize=(10, 5))
                # Forecasts
                ax.plot(forecast_df.index, forecast_df["GARCH Forecast"], label="GARCH Forecast", color="blue")
                ax.plot(forecast_df.index, forecast_df["EWMA Forecast"], label="EWMA Forecast", color="green", linestyle="--")
                # Confidence band
                ax.fill_between(forecast_df.index, forecast_df["GARCH Lower"], forecast_df["GARCH Upper"],
                                color="blue", alpha=0.2, label="GARCH ±1 Std Dev (Approx.)") # Label as approximate


                ax.set_title("Volatility Forecast")
                ylabel = "Annualized Volatility" if st.session_state.get('annualize_volatility', True) else "Daily Volatility"
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Date")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Store forecast results in session state
                st.session_state['forecast_results'] = forecast_df

            except Exception as e:
                st.error(f"Forecasting failed: {e}")
                # Clear previous forecast results on error
                if 'forecast_results' in st.session_state:
                    del st.session_state['forecast_results']

        # --- Display Forecast Results (if available in session state) ---
        elif 'forecast_results' in st.session_state:
            st.markdown("----")
            st.subheader("🔮 Volatility Forecast Results")
            forecast_df = st.session_state['forecast_results']
            st.dataframe(forecast_df.style.format("{:.4%}"))

            st.subheader("Volatility Forecast (GARCH vs EWMA)")
            fig, ax = plt.subplots(figsize=(10, 5))
            # Forecasts
            ax.plot(forecast_df.index, forecast_df["GARCH Forecast"], label="GARCH Forecast", color="blue")
            ax.plot(forecast_df.index, forecast_df["EWMA Forecast"], label="EWMA Forecast", color="green", linestyle="--")
            # Confidence band
            ax.fill_between(forecast_df.index, forecast_df["GARCH Lower"], forecast_df["GARCH Upper"],
                            color="blue", alpha=0.2, label="GARCH ±1 Std Dev (Approx.)") # Label as approximate

            ax.set_title("Volatility Forecast")
            ylabel = "Annualized Volatility" if st.session_state.get('annualize_volatility', True) else "Daily Volatility"
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Date")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)