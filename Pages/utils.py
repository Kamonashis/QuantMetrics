import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model

def download_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None
        data['Return'] = data['Close'].pct_change().dropna() * 100
        return data
    except Exception as e:
        return None

def run_garch_model(returns, p=1, q=1, vol='GARCH', mean='constant'):
    model = arch_model(returns, p=p, q=q, mean=mean, vol=vol, rescale=True)
    fit = model.fit(disp='off')
    return fit

def run_ewma(returns, lambda_=0.94):
    span = (2 / (1 - lambda_)) - 1
    ewma_vol = returns.ewm(span=span).std()
    return ewma_vol

def forecast_ewma(returns, lambda_, n_days):
    last_ret_squared = returns.iloc[-1] ** 2
    last_vol = run_ewma(returns, lambda_).iloc[-1]
    last_var = last_vol ** 2

    forecast = np.zeros(n_days)
    forecast[0] = last_var
    for t in range(1, n_days):
        forecast[t] = lambda_ * forecast[t - 1] + (1 - lambda_) * last_ret_squared

    return np.sqrt(forecast)