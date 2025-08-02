"""
Z-Score Mean Reversion Strategy

This strategy uses statistical measures to identify when prices deviate
significantly from their historical mean, expecting them to revert.

Mathematical Description:
- Z-Score = (Current Price - Moving Average) / Standard Deviation
- Entry when |Z-Score| > threshold
- Exit when |Z-Score| < exit_threshold
- Can be enhanced with Ornstein-Uhlenbeck process modeling
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from scipy.stats import norm

from .base import BaseStrategy, Signal, SignalType
from ..utils import get_logger

logger = get_logger(__name__)

class ZScoreMeanReversionStrategy(BaseStrategy):
    """Z-Score based mean reversion strategy implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Z-Score mean reversion strategy.
        
        Args:
            config: Strategy configuration containing:
                - lookback_window: Lookback period for mean and std calculation (default: 20)
                - entry_threshold: Z-score threshold for entry (default: 2.0)
                - exit_threshold: Z-score threshold for exit (default: 0.5)
                - use_ou_process: Whether to use Ornstein-Uhlenbeck modeling (default: False)
        """
        super().__init__("Z-Score Mean Reversion", config)
        
        self.lookback_window = self.get_config_parameter('lookback_window', 20)
        self.entry_threshold = self.get_config_parameter('entry_threshold', 2.0)
        self.exit_threshold = self.get_config_parameter('exit_threshold', 0.5)
        self.use_ou_process = self.get_config_parameter('use_ou_process', False)
        
        # OU process parameters
        if self.use_ou_process:
            self.ou_lookback = self.get_config_parameter('ou_lookback', 100)
            self.ou_entry_threshold = self.get_config_parameter('ou_entry_threshold', 2.0)
        
        logger.info(
            f"Z-Score Mean Reversion Strategy initialized: "
            f"Lookback={self.lookback_window}, "
            f"Entry threshold={self.entry_threshold}, "
            f"Exit threshold={self.exit_threshold}, "
            f"OU Process={self.use_ou_process}"
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Z-score and related indicators.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            DataFrame with calculated indicators
        """
        df = data.copy()
        
        # Calculate rolling mean and standard deviation
        df['rolling_mean'] = df['close'].rolling(window=self.lookback_window).mean()
        df['rolling_std'] = df['close'].rolling(window=self.lookback_window).std()
        
        # Calculate Z-score
        df['zscore'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        
        # Calculate Bollinger Bands as additional reference
        df['bb_upper'] = df['rolling_mean'] + (2 * df['rolling_std'])
        df['bb_lower'] = df['rolling_mean'] - (2 * df['rolling_std'])
        
        # Calculate percentage distance from bands
        df['pct_bb'] = (df['close'] - df['rolling_mean']) / (2 * df['rolling_std'])
        
        if self.use_ou_process:
            df = self._calculate_ou_indicators(df)
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate mean reversion signals based on Z-score.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            List of trading signals
        """
        if not self.validate_data(data):
            return []
        
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        signals = []
        
        for i in range(self.lookback_window, len(df)):
            current_row = df.iloc[i]
            
            # Skip if we don't have enough data
            if pd.isna(current_row['zscore']):
                continue
            
            current_zscore = current_row['zscore']
            
            # Generate signals based on Z-score thresholds
            if self.use_ou_process:
                signals.extend(self._generate_ou_signals(df, i, current_row))
            else:
                signals.extend(self._generate_zscore_signals(df, i, current_row))
        
        logger.info(f"Generated {len(signals)} signals for Z-Score Mean Reversion strategy")
        return signals
    
    def _generate_zscore_signals(self, df: pd.DataFrame, i: int, current_row: pd.Series) -> List[Signal]:
        """Generate signals based on Z-score thresholds.
        
        Args:
            df: Data DataFrame with indicators
            i: Current row index
            current_row: Current row data
        
        Returns:
            List of signals for current row
        """
        signals = []
        current_zscore = current_row['zscore']
        
        # Entry signals when Z-score exceeds threshold
        if abs(current_zscore) > self.entry_threshold:
            
            # Oversold condition (price too low, expect reversion up)
            if current_zscore < -self.entry_threshold:
                signal_strength = min(abs(current_zscore) / 4.0, 1.0)  # Cap at 4 sigma
                
                signal = Signal(
                    timestamp=current_row.name,
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=signal_strength,
                    metadata={
                        'zscore': current_zscore,
                        'rolling_mean': current_row['rolling_mean'],
                        'rolling_std': current_row['rolling_std'],
                        'price': current_row['close'],
                        'signal_reason': 'oversold_reversion'
                    }
                )
                signals.append(signal)
                self.add_signal(signal)
            
            # Overbought condition (price too high, expect reversion down)
            elif current_zscore > self.entry_threshold:
                signal_strength = min(abs(current_zscore) / 4.0, 1.0)  # Cap at 4 sigma
                
                signal = Signal(
                    timestamp=current_row.name,
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=signal_strength,
                    metadata={
                        'zscore': current_zscore,
                        'rolling_mean': current_row['rolling_mean'],
                        'rolling_std': current_row['rolling_std'],
                        'price': current_row['close'],
                        'signal_reason': 'overbought_reversion'
                    }
                )
                signals.append(signal)
                self.add_signal(signal)
        
        # Exit signals when Z-score returns to normal range
        elif abs(current_zscore) < self.exit_threshold and len(self.signals_history) > 0:
            last_signal = self.signals_history[-1]
            
            # Only generate exit if we have an open position
            if last_signal.signal_type != SignalType.HOLD:
                signal = Signal(
                    timestamp=current_row.name,
                    symbol=df.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.HOLD,  # Exit/close position
                    strength=1.0 - abs(current_zscore) / self.exit_threshold,
                    metadata={
                        'zscore': current_zscore,
                        'price': current_row['close'],
                        'signal_reason': 'mean_reversion_exit'
                    }
                )
                signals.append(signal)
                self.add_signal(signal)
        
        return signals
    
    def _calculate_ou_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ornstein-Uhlenbeck process indicators.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            DataFrame with OU indicators
        """
        df = data.copy()
        
        # Calculate log prices for OU modeling
        df['log_price'] = np.log(df['close'])
        df['log_returns'] = df['log_price'].diff()
        
        # Estimate OU parameters using rolling windows
        def estimate_ou_params(log_prices):
            """Estimate OU process parameters."""
            if len(log_prices) < 10:
                return np.nan, np.nan, np.nan
            
            y = log_prices.diff().dropna()
            x = log_prices.shift(1).dropna()
            
            if len(y) != len(x):
                min_len = min(len(y), len(x))
                y = y.iloc[:min_len]
                x = x.iloc[:min_len]
            
            try:
                # Linear regression: dy = alpha * (mu - y) * dt + sigma * dW
                # Simplified as: dy = a + b * y + error
                X = np.column_stack([np.ones(len(x)), x])
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                
                a, b = coeffs
                mu = -a / b if b != 0 else 0  # Long-term mean
                kappa = -b  # Mean reversion speed
                sigma = np.std(y - X @ coeffs)  # Volatility
                
                return mu, kappa, sigma
            except:
                return np.nan, np.nan, np.nan
        
        # Rolling OU parameter estimation
        ou_params = df['log_price'].rolling(window=self.ou_lookback).apply(
            lambda x: estimate_ou_params(x), raw=False
        )
        
        # For simplicity, we'll use a simpler approach
        # Calculate mean reversion level and half-life
        df['ou_mean'] = df['log_price'].rolling(window=self.ou_lookback).mean()
        df['ou_std'] = df['log_price'].rolling(window=self.ou_lookback).std()
        df['ou_zscore'] = (df['log_price'] - df['ou_mean']) / df['ou_std']
        
        # Estimate half-life of mean reversion
        df['price_deviation'] = df['log_price'] - df['ou_mean']
        df['ou_signal'] = np.where(
            abs(df['ou_zscore']) > self.ou_entry_threshold,
            -np.sign(df['ou_zscore']),  # Opposite direction for mean reversion
            0
        )
        
        return df
    
    def _generate_ou_signals(self, df: pd.DataFrame, i: int, current_row: pd.Series) -> List[Signal]:
        """Generate signals based on Ornstein-Uhlenbeck process.
        
        Args:
            df: Data DataFrame with indicators
            i: Current row index
            current_row: Current row data
        
        Returns:
            List of signals for current row
        """
        signals = []
        
        if 'ou_zscore' not in current_row or pd.isna(current_row['ou_zscore']):
            return signals
        
        ou_zscore = current_row['ou_zscore']
        
        if abs(ou_zscore) > self.ou_entry_threshold:
            signal_type = SignalType.BUY if ou_zscore < 0 else SignalType.SELL
            signal_strength = min(abs(ou_zscore) / 4.0, 1.0)
            
            signal = Signal(
                timestamp=current_row.name,
                symbol=df.attrs.get('symbol', 'UNKNOWN'),
                signal_type=signal_type,
                strength=signal_strength,
                metadata={
                    'ou_zscore': ou_zscore,
                    'ou_mean': current_row['ou_mean'],
                    'price': current_row['close'],
                    'log_price': current_row['log_price'],
                    'signal_reason': 'ou_mean_reversion'
                }
            )
            signals.append(signal)
            self.add_signal(signal)
        
        return signals
    
    def get_strategy_description(self) -> str:
        """Get detailed strategy description.
        
        Returns:
            Strategy description string
        """
        return f"""
        Z-Score Mean Reversion Strategy
        ===============================
        
        Parameters:
        - Lookback Window: {self.lookback_window}
        - Entry Threshold: {self.entry_threshold}
        - Exit Threshold: {self.exit_threshold}
        - Use OU Process: {self.use_ou_process}
        
        Logic:
        1. Calculate rolling mean and standard deviation
        2. Compute Z-score = (Price - Mean) / Std
        3. Enter LONG when Z-score < -threshold (oversold)
        4. Enter SHORT when Z-score > +threshold (overbought)
        5. Exit when |Z-score| < exit_threshold
        
        Mathematical Formulas:
        - Z-Score = (P(t) - μ(t)) / σ(t)
        - μ(t) = Rolling mean over lookback window
        - σ(t) = Rolling standard deviation over lookback window
        
        Optional OU Process:
        - Models price as dX = κ(θ - X)dt + σdW
        - κ: mean reversion speed, θ: long-term mean, σ: volatility
        """
