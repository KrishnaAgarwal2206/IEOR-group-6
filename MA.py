import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean()

def calculate_ema(data, span=20):
    return data['Close'].ewm(span=span, adjust=False).mean()

def calculate_errors(data):
    result = data.copy()
    
    close_prices = result['Close'].to_numpy().flatten()
    sma_values = result['SMA'].to_numpy().flatten()
    ema_values = result['EMA'].to_numpy().flatten()
    
    sma_error = np.full_like(close_prices, np.nan)
    ema_error = np.full_like(close_prices, np.nan)
    
    valid_mask = (~np.isnan(close_prices)) & (~np.isnan(sma_values)) & (close_prices != 0)
    
    sma_error[valid_mask] = ((close_prices[valid_mask] - sma_values[valid_mask]) / 
                            close_prices[valid_mask]) * 100
    
    valid_mask = (~np.isnan(close_prices)) & (~np.isnan(ema_values)) & (close_prices != 0)
    ema_error[valid_mask] = ((close_prices[valid_mask] - ema_values[valid_mask]) / 
                            close_prices[valid_mask]) * 100
    
    result['SMA_Error_Pct'] = sma_error
    result['EMA_Error_Pct'] = ema_error
    
    return result

def plot_stock_data(stock_data, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Price plot
    ax1.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax1.plot(stock_data.index, stock_data['SMA'], label='SMA (20-day)', color='red', linestyle='dashed')
    ax1.plot(stock_data.index, stock_data['EMA'], label='EMA (20-day)', color='green', linestyle='dotted')
    ax1.set_title(f'{ticker} Stock Price with SMA & EMA')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid()
    
    ax2.plot(stock_data.index, stock_data['SMA_Error_Pct'], label='SMA Error %', color='red', linestyle='dashed')
    ax2.plot(stock_data.index, stock_data['EMA_Error_Pct'], label='EMA Error %', color='green', linestyle='dotted')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title('Percentage Error from Actual Price')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Error (%)')
    ax2.legend()
    ax2.grid()
    
    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\nError Statistics:")
    print(f"SMA Mean Absolute Error: {np.nanmean(np.abs(stock_data['SMA_Error_Pct'])):.2f}%")
    print(f"EMA Mean Absolute Error: {np.nanmean(np.abs(stock_data['EMA_Error_Pct'])):.2f}%")
    print(f"SMA Max Error: {np.nanmax(np.abs(stock_data['SMA_Error_Pct'])):.2f}%")
    print(f"EMA Max Error: {np.nanmax(np.abs(stock_data['EMA_Error_Pct'])):.2f}%")

# Main execution
if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2022-01-01"
    end_date = "2024-01-01"

    stock_data = fetch_stock_data(ticker, start_date, end_date)
    stock_data['SMA'] = calculate_sma(stock_data)
    stock_data['EMA'] = calculate_ema(stock_data)
    stock_data = calculate_errors(stock_data)

    plot_stock_data(stock_data, ticker)