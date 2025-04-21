import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

def show_modeling():
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
            clean_returns = clean_returns[clean_returns != 0]  # Remove zero returns
            garch_model = arch_model(clean_returns, mean=mean, vol=vol_type.upper(), p=garch_p, q=garch_q, rescale=True)
            garch_fit = garch_model.fit(disp='off')

            st.success("✅ GARCH model fitted successfully.")
            st.text("Model Summary:")
            st.text(garch_fit.summary())

            # --- EWMA ---
            ewma_vol = (returns).dropna().ewm(span=(2 / (1 - lambda_) - 1)).std() / 100  # Scale by 100 for percentage
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

            # Sidebar toggle for annualization
            annualize = st.sidebar.toggle("📅 Annualize Volatility", value=True)

            # Unscaled volatilities
            garch_cond_vol = garch_fit.conditional_volatility * np.sqrt(252) / 100 # Annualized
            ewma_vol = returns.dropna().ewm(span=(2 / (1 - lambda_) - 1)).std() * np.sqrt(252) / 100  # Annualized
            hist_vol = returns.dropna() * np.sqrt(252) / 100 # use actual returns instead of rolling std

            # Align lengths
            min_len = min(len(garch_cond_vol), len(ewma_vol), len(hist_vol))

            # Scale for annualization
            scaling_factor = np.sqrt(252) if annualize else 1
            garch_cond_vol = garch_cond_vol[-min_len:] * scaling_factor
            ewma_vol = ewma_vol[-min_len:] * scaling_factor
            hist_vol = hist_vol[-min_len:] * scaling_factor

            # Plot
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(garch_cond_vol.index, garch_cond_vol, label="GARCH Volatility", color="blue")
            ax.plot(ewma_vol.index, ewma_vol, label="EWMA Volatility", color="green", linestyle="--")
            ax.plot(hist_vol.index, hist_vol, label="30-Day Historical Volatility", color="black", linestyle=":")
            ax.set_title("Volatility Estimates: GARCH vs EWMA vs Historical")
            ax.set_ylabel("Annualized Volatility")
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
                garch_vol = np.sqrt(garch_var) / 100 * np.sqrt(252)  # Annualized
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
                ewma_forecast = np.sqrt(ewma_forecast) / 100 * np.sqrt(252)  # Annualized
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
                ax.set_ylabel("Annualized Volatility")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Forecasting failed: {e}")