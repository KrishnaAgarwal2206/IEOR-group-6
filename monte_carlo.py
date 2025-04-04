# -*- coding: utf-8 -*-
"""Stock Price Prediction Simulation (Final Working Version)"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ======================
# 1. DATA COLLECTION
# ======================
def fetch_stock_data(ticker="AAPL", days=258):
    """Fetch historical stock data using yfinance"""
    data = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    return data[['Close']]

stock_data = fetch_stock_data()
print(f"Fetched {len(stock_data)} trading days of data")

# ======================
# 2. TRAIN/TEST SPLIT
# ======================
train = stock_data.iloc[:-7].copy()  # First 251 days
test = stock_data.iloc[-7:].copy()   # Last 7 days for testing

# ======================
# 3. MOVING AVERAGE MODEL
# ======================
def moving_average_predict(data, window=10):
    """Predict next day's price as average of last 'window' days"""
    data = data.copy()
    data.loc[:, 'MA_Prediction'] = data['Close'].rolling(window=window).mean()
    data.loc[:, 'Next_Day_Pred'] = data['MA_Prediction'].shift(-1)
    return data.dropna()

train = moving_average_predict(train)
ma_predictions = train['Next_Day_Pred'].iloc[-7:].values  # Convert to numpy array

# ======================
# 4. MONTE CARLO SIMULATION (FIXED VERSION)
# ======================
def monte_carlo_simulation(last_price, volatility, days=7, n_simulations=1000):
    """Generate future price paths using random walks"""
    # Ensure last_price is a scalar value
    if isinstance(last_price, pd.Series):
        last_price = last_price.iloc[0]
    
    # Generate all random returns at once
    daily_returns = np.random.normal(0, volatility, (n_simulations, days))
    # Calculate cumulative product for each simulation path
    cum_returns = (1 + daily_returns).cumprod(axis=1)
    # Multiply by last price (using numpy array operations)
    simulations = last_price * cum_returns
    return simulations

# Calculate required parameters
returns = train['Close'].pct_change().dropna()
volatility = returns.std()
last_price = train['Close'].iloc[-1]  # Get scalar value

mc_simulations = monte_carlo_simulation(last_price, volatility)
mc_median = np.median(mc_simulations, axis=0)  # Median prediction for each day

# ======================
# 5. ERROR ANALYSIS
# ======================
def calculate_errors(actual, predicted):
    """Calculate MAE and RMSE"""
    actual_values = actual.values if isinstance(actual, pd.Series) else actual
    mae = mean_absolute_error(actual_values, predicted)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted))
    return mae, rmse

# MA Model Errors
ma_mae, ma_rmse = calculate_errors(test['Close'], ma_predictions)

# MC Model Errors
mc_mae, mc_rmse = calculate_errors(test['Close'], mc_median)

print(f"\nMoving Average Errors: MAE=${ma_mae:.2f}, RMSE=${ma_rmse:.2f}")
print(f"Monte Carlo Errors: MAE=${mc_mae:.2f}, RMSE=${mc_rmse:.2f}")

# ======================
# 6. VISUALIZATION
# ======================
plt.figure(figsize=(12, 6))

# Plot Monte Carlo simulations
plt.subplot(1, 2, 1)
plt.plot(mc_simulations.T, color='blue', alpha=0.1)
plt.plot(test['Close'].values, color='red', linewidth=2, label='Actual')
plt.title('Monte Carlo Simulations (1000 Paths)')
plt.xlabel('Days')
plt.ylabel('Price ($)')

# Plot Moving Average predictions
plt.subplot(1, 2, 2)
plt.plot(train.index[-21:], train['Close'].iloc[-21:], label='Historical')
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, ma_predictions, label='MA Prediction')
plt.title('Moving Average Prediction')
plt.legend()

plt.tight_layout()
plt.show()