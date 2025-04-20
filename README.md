# Stock Analysis App with ACF & PACF, GARCH, and EWMA Volatility Models

This application allows users to perform stock analysis using historical price data, calculate percentage returns, model volatility using GARCH (Generalized Autoregressive Conditional Heteroskedasticity) and EWMA (Exponentially Weighted Moving Average), and visualize the results.

## Features
- **Main Page**: Analyze a stock's historical returns, visualize the closing price, returns, and calculate autocorrelation (ACF) and partial autocorrelation (PACF).
- **Modeling Page**: Run volatility models such as GARCH and EWMA. Visualize and compare volatility estimates.
- **Forecasting**: Forecast future volatility based on the GARCH and EWMA models.

## Setup Instructions

### Prerequisites
To run this app locally, you will need to have Python 3.7 or higher installed on your machine. You also need to install the following dependencies:

1. **Streamlit**: For the web app interface
2. **yfinance**: To fetch historical stock data
3. **arch**: For GARCH modeling
4. **statsmodels**: For statistical plots like ACF and PACF
5. **pandas, numpy, matplotlib**: For data manipulation and visualization

To install the necessary dependencies, run:

```bash
pip install streamlit yfinance arch statsmodels pandas numpy matplotlib
```

### Running the App
After installing the required dependencies, run the app by using the following command in your terminal:

```bash
streamlit run app.py
```

This will launch the app in your default web browser.

## How to Use the App

### Main Page: Stock Analysis
- **Stock Ticker Symbol**: Input the stock ticker (e.g., AAPL, MSFT, etc.). This symbol will be used to fetch data from Yahoo Finance.
- **Start Date & End Date**: Define the time period for analysis.
- **Analyze Button**: Click this button to fetch the stock data and analyze the returns.

Once the analysis is complete:
- **Stock Closing Price**: A line chart showing the stock's closing price over the specified date range.
- **Percentage Returns**: A line chart showing the percentage returns of the stock.
- **ACF & PACF Plots**: Autocorrelation and Partial Autocorrelation function plots to analyze the returns.

### Modeling Page: GARCH & EWMA Volatility Modeling
- **GARCH Model**: A time-series model used to forecast volatility. You can choose the AR term (p) and MA term (q) for the GARCH model.
- **EWMA Model**: An exponentially weighted moving average model to calculate volatility.
- **Run Volatility Models Button**: Click this button to fit the GARCH and EWMA models and see their volatility estimates.

Once the models are fitted:
- **GARCH Model Summary**: Displays the summary of the fitted GARCH model.
- **Volatility Comparison Plot**: A plot comparing GARCH, EWMA, and Historical Volatility.

### Forecasting Volatility
- **Forecast Days**: Specify how many days to forecast the volatility.
- **Forecast Volatility Button**: Click this button to forecast the volatility for the specified days using the fitted GARCH and EWMA models.

Once the forecast is complete:
- **Volatility Forecast Plot**: A plot showing the forecasted volatility for the specified days.
- **Confidence Bands**: ±1 standard deviation confidence bands for the GARCH forecast.

### Navigation
- **Back to Analysis**: A button to return to the main page for new analysis.
- **Go to Modeling**: A button to switch to the modeling page.

## Additional Information
- **GARCH Model**: Generalized Autoregressive Conditional Heteroskedasticity is a class of models used to analyze and forecast time-varying volatility in financial markets.
- **EWMA Model**: Exponentially Weighted Moving Average is used to estimate the volatility with a smoothing factor.

## Troubleshooting
- If you encounter errors related to missing data, please check the stock ticker symbol and date range.
- Ensure your internet connection is stable to fetch stock data from Yahoo Finance.

## Future Enhancements
- Add a feature to compare multiple tickers at once (e.g., calculate and visualize correlation between stock returns).
- Add more volatility models like EGARCH or HARCH.
