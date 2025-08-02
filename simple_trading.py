#!/usr/bin/env python3
"""
Simplified imports for notebook - creates a simpler version without complex cross-dependencies
"""

import sys
import os
from pathlib import Path
import warnings

# Add project root to Python path
project_root = Path(__file__).parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import essential libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SimpleDataManager:
    """Simplified data manager for notebook use"""
    
    @staticmethod
    def get_historical_data(symbol, start_date, end_date, frequency='1d'):
        """Get historical data using yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=frequency)
            
            # Standardize column names
            data.columns = [col.lower() for col in data.columns]
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

class SimpleTradingStrategy:
    """Base class for simplified trading strategies"""
    
    def __init__(self, config=None):
        self.config = config or {}
    
    def calculate_ema(self, data, period):
        """Calculate EMA"""
        return data.ewm(span=period).mean()
    
    def calculate_sma(self, data, period):
        """Calculate SMA"""
        return data.rolling(window=period).mean()
    
    def calculate_zscore(self, data, window=20):
        """Calculate Z-score"""
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        return (data - rolling_mean) / rolling_std

class SimpleBacktester:
    """Simplified backtester for notebook use"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
    
    def calculate_returns(self, data, signals):
        """Calculate strategy returns"""
        # Simple implementation
        returns = []
        position = 0
        
        for i, signal in enumerate(signals):
            if signal == 1 and position == 0:  # Buy
                position = 1
            elif signal == -1 and position == 1:  # Sell
                position = 0
        
        return pd.Series(returns, index=data.index)

# Create global instances
data_manager = SimpleDataManager()

print("✅ Simplified trading modules loaded successfully!")
print("Available classes:")
print("  - data_manager: SimpleDataManager")
print("  - SimpleTradingStrategy: Base strategy class")
print("  - SimpleBacktester: Basic backtesting")
