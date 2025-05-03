import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime, timedelta
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def show_regression():
    st.title("📉 Regression Analysis (Returns)")
    st.markdown("Perform regression analysis on the **returns** of selected tickers.")

    # --- Input Section ---
    st.header("Input Parameters")
    # Use session state to persist ticker input value
    if 'regression_tickers_input' not in st.session_state:
        st.session_state['regression_tickers_input'] = "^IXIC, NVDA, AAPL, MSFT, GOOG"
    tickers_input = st.text_area("Enter tickers separated by commas. The first symbol is the dependent variable!", value=st.session_state['regression_tickers_input'], key='regression_tickers_input_widget')
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    st.session_state['regression_tickers_input'] = tickers_input # Update session state


    st.header("Date Range")
    end_date = datetime.today()
    start_date_default = end_date - timedelta(days=365 * 5)

    col1, col2 = st.columns(2)
    with col1:
        # Use session state to persist start date
        if 'regression_start_date' not in st.session_state:
            st.session_state['regression_start_date'] = start_date_default
        start_date = st.date_input("Start Date", value=st.session_state['regression_start_date'], min_value=start_date_default, max_value=end_date, key='regression_start_date_widget')
        st.session_state['regression_start_date'] = start_date # Update session state
    with col2:
        # Use session state to persist end date
        if 'regression_end_date' not in st.session_state:
            st.session_state['regression_end_date'] = end_date
        end_date = st.date_input("End Date", value=st.session_state['regression_end_date'], min_value=start_date, max_value=end_date, key='regression_end_date_widget')
        st.session_state['regression_end_date'] = end_date # Update session state
    st.markdown("**Note:** The date range is inclusive of the start and end dates.")

    st.header("Regression Options")
    # Use session state for regression type radio button
    if 'regression_type' not in st.session_state:
        st.session_state['regression_type'] = 'Standard OLS'
    regression_type = st.radio(
        "Select Regression Type",
        ('Standard OLS', 'Robust Regression', 'Rolling Regression'),
        index=('Standard OLS', 'Robust Regression', 'Rolling Regression').index(st.session_state['regression_type']),
        key='regression_type_widget'
    )
    st.session_state['regression_type'] = regression_type # Update session state


    robust = (regression_type == 'Robust Regression')
    rolling = (regression_type == 'Rolling Regression')

    if robust:
        st.markdown("**Note:** Robust regression is used to reduce the influence of outliers.")

    window = None
    if rolling:
        # Use session state for window size slider
        if 'regression_window_size' not in st.session_state:
            st.session_state['regression_window_size'] = 60
        window = st.slider("Rolling Window Size", 30, 252, value=st.session_state['regression_window_size'], key='regression_window_size_widget')
        st.session_state['regression_window_size'] = window # Update session state
        st.markdown("**Note:** Rolling regression calculates the regression coefficients over a rolling window.")
        st.markdown("**Note:** This may take a while for large datasets.")


    # --- Run Regression Button ---
    if st.button("Run Regression", key='run_regression_button'):
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers to run regression.")
            # Clear previous results if button is clicked with insufficient tickers
            if 'regression_results' in st.session_state:
                del st.session_state['regression_results']
            return

        try:
            # Download 'Close' prices
            df = yf.download(tickers, start=start_date, end=end_date)['Close']

            if df.empty:
                st.error("No data retrieved for the selected tickers and date range.")
                if 'regression_results' in st.session_state:
                    del st.session_state['regression_results']
                return

            # Calculate returns and drop NaNs
            returns = df.pct_change().dropna()

            if returns.empty:
                st.error("No return data available for the selected tickers and date range.")
                if 'regression_results' in st.session_state:
                    del st.session_state['regression_results']
                return

            # Define dependent and independent variables using returns
            y = returns[tickers[0]]
            X = returns[tickers[1:]]

            # Check if X is empty after calculating returns and dropping NaNs
            if X.empty:
                 st.error("No return data available for independent variables after cleaning.")
                 if 'regression_results' in st.session_state:
                    del st.session_state['regression_results']
                 return

            # Add a constant (intercept) to the independent variables
            X = sm.add_constant(X)

            # Store necessary data and parameters for display in session state
            regression_results = {
                'tickers': tickers, # Store tickers for display
                'regression_type': regression_type, # Store regression type
                'y_label': f"Returns ({tickers[0]})", # Label for dependent variable
                'X_labels': list(X.columns), # Labels for independent variables
            }


            if rolling:
                st.subheader("📉 Rolling Regression Coefficients")
                st.markdown("This section shows the rolling regression coefficients for the selected window size.")
                st.markdown(f"**Rolling Window Size:** {window} days")
                st.markdown("**Dependent Variable (Returns):** " + tickers[0])

                if returns.shape[0] < window:
                    st.warning(f"Not enough data points ({returns.shape[0]}) for the specified rolling window size ({window}). Please reduce the window size or extend the date range.")
                    if 'regression_results' in st.session_state:
                        del st.session_state['regression_results']
                    return

                # Perform rolling regression on Returns
                # Adjust index start for rolling window based on returns DataFrame
                rolling_results_df = pd.DataFrame(index=returns.index[window-1:])
                for col in X.columns:
                    # Use returns DataFrame for rolling calculation
                    # Ensure the constant is handled correctly within the lambda
                    # Select columns based on original tickers list to avoid issues with 'const'
                    cols_for_rolling = [tickers[0]] + tickers[1:]
                    rolling_results_df[col] = returns[cols_for_rolling].rolling(window).apply(
                        lambda x: sm.OLS(x.iloc[:, 0], sm.add_constant(x.iloc[:, 1:])).fit().params.get(col, np.nan),
                        raw=False # Set raw=False to pass DataFrame slices
                    )
                # Drop rows with NaN values that result from the rolling calculation
                rolling_results_df.dropna(inplace=True)
                regression_results['rolling_results_df'] = rolling_results_df

            else: # Handles Standard OLS and Robust Regression
                # Perform standard OLS regression on Returns
                model = sm.OLS(y, X)
                fit = model.fit(cov_type="HC3") if robust else model.fit() # Use robust errors if selected

                regression_results['model_summary'] = fit.summary()
                regression_results['aic'] = fit.aic
                regression_results['bic'] = fit.bic
                regression_results['rsquared'] = fit.rsquared
                regression_results['residuals'] = fit.resid
                regression_results['X_data'] = X # Store X data for VIF and forecasting

                # --- Forecasting ---
                # Create a DataFrame with one row from the last observation of X
                last_obs_df = pd.DataFrame([X.iloc[-1]])
                # Predict using the DataFrame with the correct shape (1, k+1)
                forecast = fit.predict(last_obs_df)[0]
                regression_results['forecast'] = forecast


            # Store all regression results in session state
            st.session_state['regression_results'] = regression_results
            st.success("Regression analysis complete!")

        except Exception as e:
            st.error(f"An error occurred during regression analysis: {e}")
            # Clear results on error
            if 'regression_results' in st.session_state:
                del st.session_state['regression_results']

    # --- Display Results (if available in session state) ---
    if 'regression_results' in st.session_state:
        results = st.session_state['regression_results']
        regression_type = results['regression_type']
        tickers = results['tickers'] # Retrieve tickers for display
        y_label = results['y_label'] # Retrieve y_label

        if regression_type == 'Rolling Regression':
            st.subheader("📉 Rolling Regression Coefficients")
            window = st.session_state.get('regression_window_size', 60) # Retrieve window size
            st.markdown("This section shows the rolling regression coefficients for the selected window size.")
            st.markdown(f"**Rolling Window Size:** {window} days")
            st.markdown("**Dependent Variable (Returns):** " + tickers[0])

            if 'rolling_results_df' in results:
                st.line_chart(results['rolling_results_df'])
            else:
                 st.warning("Rolling regression results are not available.")

        else: # Handles Standard OLS and Robust Regression
            st.subheader("📊 Model Summary")
            st.markdown("The model summary provides detailed statistics about the regression model.")
            st.markdown("**Dependent Variable (Returns):** " + tickers[0])
            st.text(results['model_summary'])

            st.markdown(f"**AIC:** {results['aic']:.4f} | **BIC:** {results['bic']:.4f} | **R²:** {results['rsquared']:.4%}")

            # --- Multicollinearity and Residuals in Two Columns ---
            col_vif, col_resid = st.columns(2)

            with col_vif:
                # --- VIF ---
                st.subheader("📈 Multicollinearity Check (VIF)")
                st.markdown("Variance Inflation Factor (VIF) is used to detect multicollinearity in regression models. A VIF value greater than 10 indicates high multicollinearity.")
                st.markdown("VIF = 1 / (1 - R²)")
                # Recalculate VIF using the stored X data
                if 'X_data' in results:
                    X_data = results['X_data']
                    vif_data = pd.DataFrame()
                    vif_data["Feature"] = X_data.columns
                    vif_data["VIF"] = [variance_inflation_factor(X_data.values, i) for i in range(X_data.shape[1])]
                    st.dataframe(vif_data)
                else:
                    st.warning("Independent variable data (X) not available for VIF calculation.")


            with col_resid:
                # --- Residuals ---
                st.subheader("📊 Residual Analysis")
                st.markdown("Residuals are the differences between the observed and predicted values. They should ideally be randomly distributed around zero.")
                if 'residuals' in results:
                    st.markdown("**Residuals Summary:**")
                    st.write(results['residuals'].describe()) # Display summary statistics for residuals
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    ax[0].hist(results['residuals'], bins=30, color="skyblue", edgecolor="black")
                    ax[0].set_title("Residual Histogram")
                    sm.qqplot(results['residuals'], line='45', ax=ax[1])
                    ax[1].set_title("Q-Q Plot")
                    st.pyplot(fig)
                else:
                    st.warning("Residuals data not available for analysis.")

            # --- Forecasting ---
            st.subheader("🔮 Forecast Next Day Return")
            st.markdown("Using the last observation to predict the next day's return.")
            if 'X_data' in results:
                st.markdown("**Last Observation of Independent Variables (including constant):**")
                st.write(results['X_data'].iloc[-1].to_dict()) # Display the last observation used for prediction
            else:
                 st.warning("Independent variable data (X) not available for displaying last observation.")

            if 'forecast' in results:
                st.markdown(f"**Predicted return for {tickers[0]}:** `{results['forecast']:.4%}`") # Display forecast with appropriate formatting
            else:
                 st.warning("Forecast result not available.")