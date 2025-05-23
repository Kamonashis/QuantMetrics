# Stock Analysis App with ACF & PACF, GARCH, and EWMA Volatility Models

This application allows users to perform stock analysis using historical price data, calculate percentage returns, model volatility using GARCH (Generalized Autoregressive Conditional Heteroskedasticity) and EWMA (Exponentially Weighted Moving Average), and visualize the results.

## Features
The application is structured into several pages, each providing distinct functionalities:

- **🏠 Home**: Welcome page with an overview of the application and its capabilities.
- **📊 Analysis**:
    - Fetch and display historical stock price data.
    - Option to use **Simple Returns** or **Log (Continuously Compounded) Returns**.
    - Visualize price trends and selected return distributions.
    - Display **Skewness** and **Excess Kurtosis** of returns.
    - Calculate and plot **Rolling Annualized Volatility** for user-defined window sizes.
    - Plot Autocorrelation (ACF) and Partial Autocorrelation (PACF) for **squared returns** (to analyze volatility clustering).
    - Perform and display results of Seasonal Decomposition and ADF Test on residuals.
- **🔍 Modeling**:
    - Implement and compare volatility models:
        - GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
        - EWMA (Exponentially Weighted Moving Average)
    - Display model summaries and visualize volatility estimates.
    - Forecast future volatility with confidence bands for GARCH.
- **📈 Correlation Analysis**:
    - Calculate and visualize the correlation matrix for the returns of multiple stocks.
    - Display an interactive heatmap of correlations.
- **📉 Regression Analysis**:
    - Perform regression analysis on stock returns (dependent vs. independent variables).
    - Supports Standard OLS, Robust Regression, and Rolling Window Regression.
    - Provides model summaries, VIF for multicollinearity, residual analysis, and next-day return forecasting.

## Setup Instructions

### Prerequisites
To run this app locally, you will need to have Python 3.7 or higher installed on your machine. You also need to install the following dependencies:

1. **Streamlit**: For the web app interface.
2. **yfinance**: To fetch historical stock data.
3. **pandas, numpy**: For data manipulation.
4. **matplotlib, seaborn, plotly, plotly.express**: For data visualization and interactive plots.
5. **statsmodels**: For statistical models and tests (e.g., ACF, PACF, OLS regression).
6. **scipy**: For scientific computing (used in various statistical calculations).
7. **arch**: For GARCH volatility modeling.

It is recommended to install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Running the App
After installing the required dependencies, run the app by using the following command in your terminal:

```bash
streamlit run app.py
```

This will launch the app in your default web browser.

## How to Use the App

Navigate through the app using the sidebar on the left. Each page offers different tools for stock analysis:

### 🏠 Home Page
- Provides a welcome message and an overview of the application's features.
- Offers quick links or instructions to get started with different analysis types.

### 📊 Analysis Page
Provides tools for in-depth analysis of a single stock's historical data.
- **Input Parameters**:
    - **Stock Ticker Symbol**: Input a single stock ticker (e.g., AAPL, MSFT).
    - **Return Type**: Choose between "Simple" or "Log (Continuously Compounded)" returns. This selection affects return calculations and subsequent analyses.
    - **Start Date & End Date**: Define the historical period for analysis.
    - **Seasonal Decomposition Period**: Set the period for seasonal decomposition (e.g., 30 for monthly data if daily prices).
    - **Decomposition Model**: Choose "Additive" or "Multiplicative".
    - **Rolling Volatility Window(s)**: Enter comma-separated day numbers for rolling volatility calculation (e.g., "30,60,90").
- **Run Analysis Button**: Fetches data and performs calculations.
- **Displayed Outputs**:
    - **Stock Closing Price**: Line chart of historical closing prices.
    - **Daily Returns**: Line chart of the selected type of daily returns.
    - **Return Distribution Statistics**:
        - **Skewness**: Measures the asymmetry of the return distribution.
        - **Excess Kurtosis (Fisher)**: Measures the 'tailedness' of the return distribution.
    - **Rolling Annualized Volatility**: Chart displaying the annualized rolling volatility for the specified window(s).
    - **ACF & PACF of Squared Returns**: Autocorrelation and Partial Autocorrelation plots for squared returns, useful for identifying volatility clustering.
    - **Interactive Seasonal Decomposition**: Plots of observed, trend, seasonal, and residual components of the closing price.
    - **Augmented Dickey-Fuller (ADF) Test on Residuals**: Statistical test for stationarity of the decomposition residuals.

### 🔍 Modeling Page
This page allows for volatility modeling and forecasting.
- **Inputs from Analysis Page**: Uses the ticker and date range from the Analysis page.
- **GARCH Model**: Configure p and q terms for the GARCH model.
- **EWMA Model**: Uses a standard smoothing factor.
- **Run Volatility Models Button**: Fits GARCH and EWMA models.
- **Outputs**:
    - **GARCH Model Summary**: Detailed statistics of the fitted GARCH model.
    - **Volatility Comparison Plot**: Compares GARCH, EWMA, and historical volatility.
- **Forecasting Volatility**:
    - **Forecast Days**: Specify the number of days for future volatility forecast.
    - **Forecast Volatility Button**: Generates and plots the forecast.
    - **Output**: Volatility forecast plot with confidence bands for GARCH.

### 📈 Correlation Analysis Page
- **Stock Ticker Symbols**: Input multiple stock tickers separated by commas (e.g., AAPL, MSFT, GOOG).
- **Date Range**: Define the historical period for analysis (uses common start/end dates).
- **Run Analysis Button**: Fetches data, calculates returns, and displays:
    - **Correlation Matrix Heatmap**: An interactive heatmap showing the correlation coefficients between the returns of the selected stocks.

### 📉 Regression Analysis Page
Perform regression analysis on stock returns.
- **Ticker Symbols**: Enter multiple tickers. The first ticker is the dependent variable (Y), and subsequent tickers are independent variables (X).
- **Date Range**: Specify the start and end dates for the historical data.
- **Regression Options**:
    - **Select Regression Type**: Choose from 'Standard OLS', 'Robust Regression', or 'Rolling Regression'.
    - **Rolling Window Size**: If 'Rolling Regression' is selected, specify the window size (e.g., 60 days).
- **Run Regression Button**: Executes the analysis.
- **Outputs**:
    - **Model Summary**: Detailed OLS regression results (coefficients, R-squared, p-values, etc.). For Rolling Regression, a plot of rolling coefficients is shown.
    - **AIC/BIC/R²**: Model fit statistics.
    - **Multicollinearity Check (VIF)**: Variance Inflation Factor for independent variables.
    - **Residual Analysis**: Histogram and Q-Q plot of residuals.
    - **Forecast Next Day Return**: Prediction for the dependent variable's next-day return based on the latest data.

### General Navigation
- The sidebar allows switching between any of these pages at any time.
- Some pages might retain input values (like tickers or dates) when you switch between them to provide a smoother workflow.

## Additional Information
- **GARCH Model**: Generalized Autoregressive Conditional Heteroskedasticity is a class of models used to analyze and forecast time-varying volatility in financial markets.
- **EWMA Model**: Exponentially Weighted Moving Average is used to estimate the volatility with a smoothing factor.

## Troubleshooting
- If you encounter errors related to missing data, please check the stock ticker symbol and date range.
- Ensure your internet connection is stable to fetch stock data from Yahoo Finance.

## Future Enhancements
- Add a feature to compare multiple tickers at once (e.g., calculate and visualize correlation between stock returns).
- Add more volatility models like EGARCH or HARCH.

## 🚀 Updates & New Features (21 April, 2025)

This app has been significantly enhanced to support more advanced quantitative analysis and interactivity. Here's what's new:

### 🔁 Multi-Page Navigation
- Structured the app into multiple pages using `app.py` as a router.
- Sidebar navigation allows switching between:
  - **Home**
  - **Analysis**
  - **Modeling**
  - **📈 Correlation Analysis**
  - **Regression Analysis**

### 📈 Volatility Modeling Enhancements
- GARCH and EWMA volatility models now:
  - Show **annualized** volatility.
  - Display **EWMA statistics** (mean, min, max, latest).
  - Present GARCH model summary with cleaner formatting.

### 📈 Correlation Analysis Page
- Accepts multiple tickers via textarea input.
- Displays an interactive heatmap.
- **Correlation values stay centered** in their respective cells for clarity.

### 📊 Regression Analysis Page
- Automatically determines model type based on the number of tickers:
  - Single independent variable: Simple Linear Regression.
  - Multiple variables: Multiple Linear Regression.
- Pulls historical price data using `yfinance`.
- Includes full model diagnostics:
  - Residual plots
  - R-squared and adjusted R-squared
  - AIC/BIC for model selection
  - VIF for multicollinearity detection
- Supports:
  - **Robust Regression**
  - **Rolling Window Regression**
  - **Forecasting with prediction intervals**