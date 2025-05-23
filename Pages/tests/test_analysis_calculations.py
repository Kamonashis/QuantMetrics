import pandas as pd
import numpy as np
import pytest

# Sample data for testing
@pytest.fixture
def sample_price_data():
    # Prices: 100, 102, 101, 103, 105, 104
    return pd.Series([100.0, 102.0, 101.0, 103.0, 105.0, 104.0])

@pytest.fixture
def sample_returns_data():
    # Chosen to be simple for manual verification of rolling std, skew, kurtosis
    return pd.Series([0.02, -0.01, 0.02, 0.02, -0.01])

# 1. Test Return Calculations
def test_simple_returns(sample_price_data):
    expected_returns = pd.Series([np.nan, 0.02, -0.00980392156862745, 0.019801980198019802, 0.01941747572815534, -0.009523809523809525])
    calculated_returns = sample_price_data.pct_change()
    pd.testing.assert_series_equal(calculated_returns, expected_returns, rtol=1e-7)

def test_log_returns(sample_price_data):
    expected_returns = pd.Series([np.nan, np.log(102/100), np.log(101/102), np.log(103/101), np.log(105/103), np.log(104/105)])
    calculated_returns = np.log(sample_price_data / sample_price_data.shift(1))
    pd.testing.assert_series_equal(calculated_returns, expected_returns, rtol=1e-7)

# 2. Test Rolling Volatility
def test_rolling_volatility(sample_returns_data):
    window = 3
    # For series [0.02, -0.01, 0.02, 0.02, -0.01]
    # Window 1: [0.02, -0.01, 0.02], std = 0.0152752523 (pandas ddof=1)
    # Window 2: [-0.01, 0.02, 0.02], std = 0.0173205081
    # Window 3: [0.02, 0.02, -0.01], std = 0.0173205081
    # Annualized factor
    annual_factor = np.sqrt(252)
    expected_volatility = pd.Series([
        np.nan, 
        np.nan, 
        0.015275252316519465 * annual_factor, 
        0.01732050807568877 * annual_factor, 
        0.01732050807568877 * annual_factor
    ])
    
    returns_series = pd.Series(sample_returns_data) # Ensure it's a Series
    calculated_volatility = returns_series.rolling(window=window).std() * annual_factor
    pd.testing.assert_series_equal(calculated_volatility, expected_volatility, rtol=1e-7)

# 3. Test Skewness and Kurtosis
def test_skewness(sample_returns_data):
    # Skewness of [0.02, -0.01, 0.02, 0.02, -0.01]
    # Calculated using scipy.stats.skew: -0.3850370133002083
    expected_skewness = -0.3850370133002083
    calculated_skewness = sample_returns_data.skew() # .skew() is directly from pandas Series
    assert np.isclose(calculated_skewness, expected_skewness, rtol=1e-7)

def test_kurtosis(sample_returns_data):
    # Kurtosis of [0.02, -0.01, 0.02, 0.02, -0.01] (Fisher's definition, Normal is 0)
    # Calculated using scipy.stats.kurtosis: -1.504
    expected_kurtosis = -1.5040000000000005
    calculated_kurtosis = sample_returns_data.kurtosis() # .kurtosis() is directly from pandas Series
    assert np.isclose(calculated_kurtosis, expected_kurtosis, rtol=1e-7)
