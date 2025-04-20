import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from datetime import datetime, timedelta

# --- Init session state ---
if 'page' not in st.session_state:
    st.session_state.page = "main"

# --- MAIN PAGE ---
def show_main_page():
    st.title("📈 Stock Analysis App with ACF & PACF")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Stock Ticker Symbol as used in Yfinance", "AAPL")
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=3660))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    analyze_button = st.sidebar.button("Analyze")

    if analyze_button or 'analyzed' in st.session_state:
        if analyze_button:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)

                if data.empty:
                    st.error("No data found. Please check the ticker symbol and date range.")
                    return
                else:
                    # Calculate returns
                    data['Return'] = data['Close'].pct_change().dropna() * 100

                    # Save in session_state
                    st.session_state['returns'] = data['Return']
                    st.session_state['ticker'] = ticker
                    st.session_state['analyzed'] = True

            except Exception as e:
                st.error(f"Error downloading data: {e}")
                return

        # Display plots using data in session state
        st.subheader(f"Closing Price for {st.session_state['ticker']}")
        st.line_chart(yf.download(st.session_state['ticker'], start=start_date, end=end_date)['Close'])

        st.subheader("Percentage Return")
        st.line_chart(st.session_state['returns'])

        st.subheader("Autocorrelation Function (ACF)")
        fig_acf, ax_acf = plt.subplots()
        plot_acf(st.session_state['returns'].dropna()**2, ax=ax_acf, lags=25)
        st.pyplot(fig_acf)

        st.subheader("Partial Autocorrelation Function (PACF)")
        fig_pacf, ax_pacf = plt.subplots()
        plot_pacf(st.session_state['returns'].dropna()**2, ax=ax_pacf, lags=25, method='ywm')
        st.pyplot(fig_pacf)

        # Navigation button
        if st.button("Go to GARCH / RiskMetrics Modeling"):
            st.session_state.page = "modeling"
            st.rerun()


# --- MODELING PAGE ---
def show_modeling_page():
    st.title("📊 GARCH & RiskMetrics (EWMA) Modeling")

    if 'returns' not in st.session_state:
        st.warning("No return data found. Please run the analysis first.")
        if st.button("Back to Analysis"):
            st.session_state.page = "main"
            st.rerun()
        return

    returns = st.session_state.returns
    ticker = st.session_state.ticker
    st.write(f"Modeling volatility for **{ticker}**")

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("GARCH Parameters")
    garch_p = st.sidebar.slider("AR term (p)", 1, 15, 1)
    garch_q = st.sidebar.slider("MA term (q)", 1, 15, 1)
    vol_type = st.sidebar.selectbox("Volatility Model", ["GARCH", "EGARCH", "HARCH"], index=0)
    mean_type = st.sidebar.selectbox("Mean Model", ["Constant", "AR", "Zero Mean"], index=0)

    st.sidebar.header("EWMA Parameters")
    lambda_ = st.sidebar.slider("EWMA Lambda (λ)", min_value=0.85, max_value=0.99, value=0.94, step=0.01)

    # --- RUN BOTH MODELS ---
    st.subheader("📌 Run Models")
    if st.button("🔍 Run Volatility Models"):
        try:
            # --- GARCH ---
            mean = mean_type.lower() if mean_type != "Zero Mean" else "zero"
            clean_returns = returns.dropna()
            garch_model = arch_model(clean_returns, mean=mean, vol=vol_type.upper(), p=garch_p, q=garch_q, rescale=True)
            garch_fit = garch_model.fit(disp='off')
            
            st.subheader("📋 GARCH Model Summary")
            st.text_area("Model Summary", value=str(garch_fit.summary()), height=400)


            # --- EWMA ---
            ewma_vol = (returns).dropna().ewm(span=(2 / (1 - lambda_) - 1)).std()/10
            st.subheader("📉 EWMA Volatility Estimate")
            st.markdown(f"""
                - **Mean Volatility:** {ewma_vol.mean():.4%}  
                - **Latest Volatility:** {ewma_vol.iloc[-1]:.4%}  
                - **Max Volatility:** {ewma_vol.max():.4%}  
                - **Min Volatility:** {ewma_vol.min():.4%}
            """)

            # --- Save models to session state ---
            st.session_state['garch_fit'] = garch_fit
            st.session_state['ewma_last_vol'] = ewma_vol.iloc[-1]
            st.session_state['ewma_vol_series'] = ewma_vol

            # --- Plot GARCH vs EWMA ---
            st.subheader("📉 Volatility Comparison (GARCH vs EWMA)")

            # Sidebar toggle
            annualize = st.sidebar.checkbox("📅 Annualize Volatility", value=False)

            # Unscaled volatilities
            garch_cond_vol = garch_fit.conditional_volatility
            ewma_vol = returns.dropna().ewm(span=(2 / (1 - lambda_) - 1)).std()
            hist_vol = returns.dropna()  # use actual returns instead of rolling std

            # Align lengths
            min_len = min(len(garch_cond_vol), len(ewma_vol), len(hist_vol))

            # Scale for annualization
            scaling_factor = np.sqrt(252) if annualize else 1
            garch_cond_vol = garch_cond_vol[-min_len:] * scaling_factor
            ewma_vol = ewma_vol[-min_len:] * scaling_factor
            hist_vol = hist_vol[-min_len:] * scaling_factor

            # Plot
            fig, ax = plt.subplots(figsize=(30, 10))
            ax.plot(garch_cond_vol.index, garch_cond_vol, label="GARCH Volatility", color="blue")
            ax.plot(ewma_vol.index, ewma_vol, label="EWMA Volatility", color="green", linestyle="--")
            ax.plot(hist_vol.index, hist_vol, label="30-Day Historical Volatility", color="black", linestyle=":")
            ax.set_title("Volatility Estimates: GARCH vs EWMA vs Historical")
            ax.set_ylabel("Annualized Volatility" if annualize else "Daily Volatility")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Modeling failed: {e}")

    # --- FORECASTING SECTION ---
    if 'garch_fit' in st.session_state and 'ewma_last_vol' in st.session_state:
        st.markdown("----")
        st.subheader("🔮 Forecast Volatility")

        n_days = st.number_input("Number of days to forecast", min_value=1, max_value=100, value=5)

        if st.button("📅 Forecast Volatility"):
            try:
                garch_fit = st.session_state['garch_fit']
                garch_forecast = garch_fit.forecast(horizon=n_days)
                garch_var = garch_forecast.variance.values[-1]
                garch_vol = np.sqrt(garch_var)/10
                garch_vol_series = pd.Series(garch_vol, name="GARCH Forecast")

                # ±1 Std Dev Confidence Band
                std_dev = garch_vol.std()
                garch_upper = pd.Series(garch_vol + std_dev, name="GARCH Upper")
                garch_lower = pd.Series(garch_vol - std_dev, name="GARCH Lower")

                # EWMA Forecast
                ewma_vol = st.session_state['ewma_vol_series']
                last_ret_squared = returns.dropna().iloc[-1] ** 2
                last_var = ewma_vol.iloc[-1] ** 2
                
                # Initialize forecast array
                ewma_forecast = np.zeros(n_days)
                ewma_forecast[0] = last_var

                # Recursive forecast of variance
                for t in range(1, n_days):
                    ewma_forecast[t] = lambda_ * ewma_forecast[t - 1] + (1 - lambda_) * last_ret_squared

                # Convert back to standard deviation
                ewma_forecast = np.sqrt(ewma_forecast)
                ewma_vol_series = pd.Series(ewma_forecast, name="EWMA Forecast")

                forecast_df = pd.concat([garch_vol_series, ewma_vol_series, garch_upper, garch_lower], axis=1)
                forecast_df.index = pd.date_range(start=pd.Timestamp.today(), periods=n_days, freq='B')

                st.dataframe(forecast_df.style.format("{:.4%}"))

                # Plot
                fig, ax = plt.subplots(figsize=(10, 5))
                # Forecasts
                ax.plot(forecast_df.index, forecast_df["GARCH Forecast"], label="GARCH Forecast", color="blue")
                ax.plot(forecast_df.index, forecast_df["EWMA Forecast"], label="EWMA Forecast", color="green", linestyle="--")
                # Confidence band
                ax.fill_between(forecast_df.index, forecast_df["GARCH Lower"], forecast_df["GARCH Upper"],
                                color="blue", alpha=0.2, label="GARCH ±1 Std Dev")

                ax.set_title("Volatility Forecast vs Historical Volatility")
                ax.set_ylabel("Volatility")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Forecasting failed: {e}")

    if st.button("Back to Analysis"):
        st.session_state.page = "main"
        st.rerun()


# --- APP ROUTER ---
if st.session_state.page == "main":
    show_main_page()
elif st.session_state.page == "modeling":
    show_modeling_page()