import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

#  DATA COLLECTION
def fetch_stock_data(ticker="TSLA", days=272):
    """Fetch historical stock data using yfinance"""
    data = yf.download(ticker, period=f"{days}d", interval="1d", auto_adjust=True)
    return data[['Close']]

stock_data = fetch_stock_data()
print(f"Fetched {len(stock_data)} trading days of data")
# TRAIN/TEST SPLIT
train = stock_data.iloc[:-252].copy()  # First 20 days
test = stock_data.iloc[-252:].copy()   # Last 252 days for testing

# MONTE CARLO SIMULATION 

def monte_carlo_simulation(last_price, volatility, days=252, n_simulations=1000):
    if isinstance(last_price, pd.Series):
        last_price = last_price.iloc[0]
    daily_returns = np.random.normal(0, volatility, (n_simulations, days))
    # Calculate cumulative product for each simulation path
    cum_returns = (1 + daily_returns).cumprod(axis=1)
    # Multiply by last price (using numpy array operations)
    simulations = last_price * cum_returns
    return simulations

# Calculate required parameters
returns = train['Close'].pct_change().dropna()
volatility = returns.std()
last_price = train['Close'].iloc[-1]  
mc_simulations = monte_carlo_simulation(last_price, volatility)
mc_median = np.median(mc_simulations, axis=0) 

# ERROR ANALYSIS

def calculate_errors(actual, predicted):
    actual_values = actual.values if isinstance(actual, pd.Series) else actual
    mae = mean_absolute_error(actual_values, predicted)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted))
    return mae, rmse

def calculate_percentage_error(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return mape


mc_mae, mc_rmse = calculate_errors(test['Close'], mc_median)

print(f"Monte Carlo Errors: MAE=${mc_mae:.2f}, RMSE=${mc_rmse:.2f}")

mc_mape = calculate_percentage_error(test['Close'].values, mc_median)
print(f"Monte Carlo Percentage Error (MAPE) = {mc_mape:.2f}%")

# Plot Monte Carlo simulations
plt.figure(figsize=(12, 6))
plt.subplot(1, 1, 1)
plt.plot(mc_simulations.T, color='blue', alpha=0.1)
plt.plot(test['Close'].values, color='red', linewidth=2, label='Actual')
plt.title('Monte Carlo Simulations (1000 Paths)')
plt.xlabel('Days')
plt.ylabel('Price ($)')
plt.show()