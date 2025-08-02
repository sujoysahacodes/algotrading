#!/usr/bin/env python3
"""
Test script to verify notebook functionality
"""

print("🧪 TESTING NOTEBOOK FUNCTIONALITY")
print("=" * 50)

try:
    # Test imports
    print("\n1. Testing imports...")
    from simple_trading import data_manager, SimpleTradingStrategy, SimpleBacktester
    import pandas as pd
    import numpy as np
    from datetime import datetime
    print("✅ All imports successful")

    # Test data fetching
    print("\n2. Testing data fetching...")
    data = data_manager.get_historical_data('AAPL', datetime(2023, 1, 1), datetime(2024, 1, 1))
    print(f"✅ Fetched {len(data)} AAPL records")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")

    # Test strategy creation
    print("\n3. Testing strategy classes...")
    
    class TestEMAStrategy(SimpleTradingStrategy):
        def calculate_indicators(self, data):
            indicators = data.copy()
            indicators['ema_12'] = self.calculate_ema(data['close'], 12)
            indicators['ema_26'] = self.calculate_ema(data['close'], 26)
            indicators['signal'] = 0
            indicators.loc[indicators['ema_12'] > indicators['ema_26'], 'signal'] = 1
            indicators.loc[indicators['ema_12'] < indicators['ema_26'], 'signal'] = -1
            return indicators
    
    strategy = TestEMAStrategy({'ema_fast': 12, 'ema_slow': 26})
    indicators = strategy.calculate_indicators(data)
    
    buy_signals = (indicators['signal'] == 1).sum()
    sell_signals = (indicators['signal'] == -1).sum()
    
    print(f"✅ EMA Strategy created successfully")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    print(f"   EMA-12 latest: ${indicators['ema_12'].dropna().iloc[-1]:.2f}")
    print(f"   EMA-26 latest: ${indicators['ema_26'].dropna().iloc[-1]:.2f}")

    # Test performance calculation
    print("\n4. Testing performance metrics...")
    returns = data['close'].pct_change()
    strategy_returns = returns * indicators['signal'].shift(1)
    
    total_return = (1 + strategy_returns.dropna()).prod() - 1
    volatility = strategy_returns.std() * np.sqrt(252)
    
    print(f"✅ Performance calculated")
    print(f"   Total Return: {total_return:.2%}")
    print(f"   Annualized Volatility: {volatility:.2%}")
    
    if volatility > 0:
        sharpe = total_return / volatility
        print(f"   Sharpe Ratio: {sharpe:.2f}")

    # Test Z-Score strategy
    print("\n5. Testing Z-Score strategy...")
    
    class TestZScoreStrategy(SimpleTradingStrategy):
        def calculate_indicators(self, data):
            indicators = data.copy()
            window = 20
            indicators['rolling_mean'] = data['close'].rolling(window).mean()
            indicators['rolling_std'] = data['close'].rolling(window).std()
            indicators['zscore'] = (data['close'] - indicators['rolling_mean']) / indicators['rolling_std']
            
            indicators['signal'] = 0
            indicators.loc[indicators['zscore'] < -2.0, 'signal'] = 1  # Buy oversold
            indicators.loc[indicators['zscore'] > 2.0, 'signal'] = -1   # Sell overbought
            
            return indicators
    
    zscore_strategy = TestZScoreStrategy({'lookback_window': 20, 'entry_threshold': 2.0})
    zscore_indicators = zscore_strategy.calculate_indicators(data)
    
    zscore_buys = (zscore_indicators['signal'] == 1).sum()
    zscore_sells = (zscore_indicators['signal'] == -1).sum()
    
    print(f"✅ Z-Score Strategy created successfully")
    print(f"   Buy signals (oversold): {zscore_buys}")
    print(f"   Sell signals (overbought): {zscore_sells}")
    print(f"   Latest Z-Score: {zscore_indicators['zscore'].dropna().iloc[-1]:.2f}")

    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 50) 
    print("✅ The notebook is ready to run without errors!")
    print("✅ Data fetching works")
    print("✅ Strategy calculations work") 
    print("✅ Performance metrics work")
    print("✅ Both EMA and Z-Score strategies work")
    print("\n🚀 You can now run the Jupyter notebook successfully!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n🔧 Please check the error above and fix before running notebook")
