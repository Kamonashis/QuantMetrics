import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def show_regression():
    st.title("📉 Regression Analysis")

    # --- Ticker Inputs ---
    st.sidebar.header("Ticker Selection")
    tickers = st.sidebar.text_area("Enter tickers separated by commas. The first symbol is the dependent variable!", "^IXIC, AAPL, MSFT, GOOG").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]

    # --- Date Range ---
    st.sidebar.header("Date Range")
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 5)

    # --- Options ---
    st.sidebar.header("Options")
    start_date = st.sidebar.date_input("Start Date", start_date, min_value=start_date, max_value=end_date) 
    end_date = st.sidebar.date_input("End Date", end_date, min_value=start_date, max_value=end_date)
    st.sidebar.markdown("**Note:** The date range is inclusive of the start and end dates.")
    robust = st.sidebar.checkbox("Robust Regression", value=False)
    st.sidebar.markdown("**Note:** Robust regression is used to reduce the influence of outliers.")
    rolling = st.sidebar.checkbox("Rolling Regression")
    st.sidebar.markdown("**Note:** Rolling regression calculates the regression coefficients over a rolling window.")
    st.sidebar.markdown("**Note:** This may take a while for large datasets.")
    window = st.sidebar.slider("Rolling Window Size", 30, 252, 60) if rolling else None

    # --- Load Data ---
    if st.button("Run Regression"):
        try:
            df = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
            returns = df.pct_change().dropna()

            y = returns[tickers[0]]
            X = returns[tickers[1:]]

            if X.empty:
                st.error("You need at least 2 tickers to run regression.")
                return

            X = sm.add_constant(X)

            if rolling:
                st.subheader("📉 Rolling Regression Coefficients")
                st.markdown("This section shows the rolling regression coefficients for the selected window size.")
                st.markdown(f"**Rolling Window Size:** {window} days")
                st.markdown("**Dependent Variable:** " + tickers[0])
                rolling_results = pd.DataFrame(index=returns.index[window:])
                for col in X.columns:
                    rolling_results[col] = returns[[tickers[0]] + list(X.columns)].rolling(window).apply(
                        lambda x: sm.OLS(x[:, 0], sm.add_constant(x[:, 1:])).fit().params.get(col, np.nan),
                        raw=True
                    )
                st.line_chart(rolling_results)
            else:
                model = sm.OLS(y, X)
                fit = model.fit(cov_type="HC3") if robust else model.fit()

                st.subheader("📊 Model Summary")
                st.markdown("The model summary provides detailed statistics about the regression model.")
                st.markdown("**Dependent Variable:** " + tickers[0])
                st.text(fit.summary())

                st.markdown(f"**AIC:** {fit.aic:.4f} | **BIC:** {fit.bic:.4f} | **R²:** {fit.rsquared:.4%}")

                # --- VIF ---
                st.subheader("📈 Multicollinearity Check (VIF)")
                st.markdown("Variance Inflation Factor (VIF) is used to detect multicollinearity in regression models. A VIF value greater than 10 indicates high multicollinearity.")
                st.markdown("VIF = 1 / (1 - R²)")
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X.columns
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                st.dataframe(vif_data)

                # --- Residuals ---
                st.subheader("📊 Residual Analysis")
                st.markdown("Residuals are the differences between the observed and predicted values. They should be normally distributed.")
                st.markdown("**Residuals:** " + str(fit.resid.describe()))
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                ax[0].hist(fit.resid, bins=30, color="skyblue", edgecolor="black")
                ax[0].set_title("Residual Histogram")
                sm.qqplot(fit.resid, line='45', ax=ax[1])
                ax[1].set_title("Q-Q Plot")
                st.pyplot(fig)

                # --- Forecasting ---
                st.subheader("🔮 Forecast Next Day Return")
                st.markdown("Using the last observation to predict the next day's return.")
                st.markdown("**Last Observation:** " + str(X.iloc[-1].to_dict()))
                last_obs = X.iloc[-1]
                forecast = fit.predict([last_obs])[0]
                st.markdown(f"**Predicted return for {tickers[0]}:** `{forecast:.4%}`")

        except Exception as e:
            st.error(f"Error: {e}")