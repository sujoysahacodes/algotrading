"""
Test suite for the algorithmic trading system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from strategies import EMAAdxStrategy, ZScoreMeanReversionStrategy
from backtesting import BacktestEngine
from data.yahoo_provider import YahooDataProvider


class TestStrategies:
    """Test trading strategies."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        # Generate realistic price data
        np.random.seed(42)
        price_changes = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(price_changes))
        
        # Generate OHLCV data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        volumes = np.random.randint(1000000, 10000000, n)
        
        self.test_data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        # Fix first row
        self.test_data.iloc[0, 0] = self.test_data.iloc[0, 3]  # open = close
    
    def test_ema_adx_strategy(self):
        """Test EMA-ADX strategy."""
        config = {
            'ema_fast': 12,
            'ema_slow': 26,
            'adx_period': 14,
            'adx_threshold': 25
        }
        
        strategy = EMAAdxStrategy(config)
        
        # Test indicator calculation
        indicators = strategy.calculate_indicators(self.test_data)
        
        assert 'ema_fast' in indicators.columns
        assert 'ema_slow' in indicators.columns
        assert 'adx' in indicators.columns
        
        # Test signal generation
        signals = strategy.generate_signals(self.test_data)
        
        # Should generate some signals
        assert isinstance(signals, list)
        # Not testing exact count as it depends on market conditions
    
    def test_zscore_strategy(self):
        """Test Z-Score mean reversion strategy."""
        config = {
            'lookback_window': 20,
            'entry_threshold': 2.0,
            'exit_threshold': 0.5
        }
        
        strategy = ZScoreMeanReversionStrategy(config)
        
        # Test indicator calculation
        indicators = strategy.calculate_indicators(self.test_data)
        
        assert 'zscore' in indicators.columns
        assert 'rolling_mean' in indicators.columns
        assert 'rolling_std' in indicators.columns
        
        # Test signal generation
        signals = strategy.generate_signals(self.test_data)
        
        assert isinstance(signals, list)


class TestBacktesting:
    """Test backtesting engine."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        np.random.seed(42)
        price_changes = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(price_changes))
        
        self.test_data = pd.DataFrame({
            'open': np.roll(prices, 1),
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
        
        self.test_data.iloc[0, 0] = self.test_data.iloc[0, 3]
    
    def test_backtest_engine(self):
        """Test backtesting engine."""
        config = {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005
        }
        
        engine = BacktestEngine(config)
        
        # Create simple strategy
        strategy_config = {
            'ema_fast': 12,
            'ema_slow': 26,
            'adx_period': 14,
            'adx_threshold': 20  # Lower threshold to generate more signals
        }
        
        strategy = EMAAdxStrategy(strategy_config)
        
        # Run backtest
        result = engine.run_backtest(
            strategy=strategy,
            data=self.test_data,
            symbol='TEST',
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        # Check result structure
        assert result.strategy_name == 'EMA-ADX'
        assert result.symbol == 'TEST'
        assert result.initial_capital == 100000
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)


class TestDataProvider:
    """Test data providers."""
    
    def test_yahoo_provider(self):
        """Test Yahoo Finance provider."""
        config = {}
        provider = YahooDataProvider(config)
        
        # Test with a well-known symbol
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        
        data = provider.get_historical_data('AAPL', start_date, end_date, '1d')
        
        if not data.empty:  # Only test if data is available
            assert 'open' in data.columns
            assert 'high' in data.columns
            assert 'low' in data.columns
            assert 'close' in data.columns
            assert 'volume' in data.columns
            
            # Test data validation
            assert provider.validate_data(data)


if __name__ == '__main__':
    pytest.main([__file__])
