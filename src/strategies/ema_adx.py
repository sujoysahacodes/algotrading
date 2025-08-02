"""
EMA-ADX Trend Following Strategy

This strategy combines Exponential Moving Average (EMA) crossovers with
Average Directional Index (ADX) to filter out weak trends and improve
signal quality.

Mathematical Description:
- EMA: EMA(t) = α * Price(t) + (1-α) * EMA(t-1), where α = 2/(n+1)
- ADX: Measures trend strength, calculated from Directional Indicators (+DI, -DI)
- Signal: EMA crossover confirmed by ADX > threshold
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType
from ..utils import get_logger

logger = get_logger(__name__)

class EMAAdxStrategy(BaseStrategy):
    """EMA-ADX trend following strategy implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EMA-ADX strategy.
        
        Args:
            config: Strategy configuration containing:
                - ema_fast: Fast EMA period (default: 12)
                - ema_slow: Slow EMA period (default: 26)
                - adx_period: ADX period (default: 14)
                - adx_threshold: ADX threshold for trend confirmation (default: 25)
        """
        super().__init__("EMA-ADX", config)
        
        self.ema_fast = self.get_config_parameter('ema_fast', 12)
        self.ema_slow = self.get_config_parameter('ema_slow', 26)
        self.adx_period = self.get_config_parameter('adx_period', 14)
        self.adx_threshold = self.get_config_parameter('adx_threshold', 25)
        
        logger.info(
            f"EMA-ADX Strategy initialized: "
            f"EMA({self.ema_fast}, {self.ema_slow}), "
            f"ADX({self.adx_period}, threshold={self.adx_threshold})"
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA and ADX indicators.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            DataFrame with calculated indicators
        """
        df = data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = self._calculate_ema(df['close'], self.ema_fast)
        df['ema_slow'] = self._calculate_ema(df['close'], self.ema_slow)
        
        # Calculate ADX
        df = self._calculate_adx(df)
        
        # Calculate EMA signals
        df['ema_signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'ema_signal'] = 1
        df.loc[df['ema_fast'] < df['ema_slow'], 'ema_signal'] = -1
        
        # Calculate signal changes (crossovers)
        df['ema_signal_change'] = df['ema_signal'].diff()
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on EMA crossovers and ADX confirmation.
        
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
        
        # Generate signals on EMA crossovers with ADX confirmation
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            
            # Skip if ADX is below threshold (weak trend)
            if current_row['adx'] < self.adx_threshold:
                continue
            
            # Check for bullish crossover (fast EMA crosses above slow EMA)
            if (current_row['ema_signal_change'] == 2 or  # From -1 to 1
                (current_row['ema_signal_change'] == 1 and df.iloc[i-1]['ema_signal'] == 0)):  # From 0 to 1
                
                signal_strength = min(current_row['adx'] / 50, 1.0)  # Normalize ADX to 0-1
                
                signal = Signal(
                    timestamp=current_row.name,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.BUY,
                    strength=signal_strength,
                    metadata={
                        'ema_fast': current_row['ema_fast'],
                        'ema_slow': current_row['ema_slow'],
                        'adx': current_row['adx'],
                        'price': current_row['close']
                    }
                )
                signals.append(signal)
                self.add_signal(signal)
            
            # Check for bearish crossover (fast EMA crosses below slow EMA)
            elif (current_row['ema_signal_change'] == -2 or  # From 1 to -1
                  (current_row['ema_signal_change'] == -1 and df.iloc[i-1]['ema_signal'] == 0)):  # From 0 to -1
                
                signal_strength = min(current_row['adx'] / 50, 1.0)  # Normalize ADX to 0-1
                
                signal = Signal(
                    timestamp=current_row.name,
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal_type=SignalType.SELL,
                    strength=signal_strength,
                    metadata={
                        'ema_fast': current_row['ema_fast'],
                        'ema_slow': current_row['ema_slow'],
                        'adx': current_row['adx'],
                        'price': current_row['close']
                    }
                )
                signals.append(signal)
                self.add_signal(signal)
        
        logger.info(f"Generated {len(signals)} signals for EMA-ADX strategy")
        return signals
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average.
        
        Args:
            prices: Price series
            period: EMA period
        
        Returns:
            EMA series
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def _calculate_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX).
        
        Args:
            data: Market data DataFrame
        
        Returns:
            DataFrame with ADX, +DI, -DI columns
        """
        df = data.copy()
        
        # True Range calculation
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Directional Movement calculation
        df['dm_plus'] = np.where(
            (df['high'] - df['high'].shift()) > (df['low'].shift() - df['low']),
            np.maximum(df['high'] - df['high'].shift(), 0),
            0
        )
        
        df['dm_minus'] = np.where(
            (df['low'].shift() - df['low']) > (df['high'] - df['high'].shift()),
            np.maximum(df['low'].shift() - df['low'], 0),
            0
        )
        
        # Smooth the True Range and Directional Movement
        df['atr'] = df['true_range'].rolling(window=self.adx_period).mean()
        df['dm_plus_smooth'] = df['dm_plus'].rolling(window=self.adx_period).mean()
        df['dm_minus_smooth'] = df['dm_minus'].rolling(window=self.adx_period).mean()
        
        # Calculate Directional Indicators
        df['di_plus'] = 100 * (df['dm_plus_smooth'] / df['atr'])
        df['di_minus'] = 100 * (df['dm_minus_smooth'] / df['atr'])
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window=self.adx_period).mean()
        
        # Clean up temporary columns
        temp_cols = ['tr1', 'tr2', 'tr3', 'true_range', 'dm_plus', 'dm_minus', 
                     'atr', 'dm_plus_smooth', 'dm_minus_smooth', 'dx']
        df = df.drop(columns=temp_cols)
        
        return df
    
    def get_strategy_description(self) -> str:
        """Get detailed strategy description.
        
        Returns:
            Strategy description string
        """
        return f"""
        EMA-ADX Trend Following Strategy
        ================================
        
        Parameters:
        - Fast EMA Period: {self.ema_fast}
        - Slow EMA Period: {self.ema_slow}
        - ADX Period: {self.adx_period}
        - ADX Threshold: {self.adx_threshold}
        
        Logic:
        1. Calculate fast and slow EMAs
        2. Calculate ADX to measure trend strength
        3. Generate BUY signal when fast EMA crosses above slow EMA and ADX > threshold
        4. Generate SELL signal when fast EMA crosses below slow EMA and ADX > threshold
        5. Signal strength is proportional to ADX value
        
        Mathematical Formulas:
        - EMA(t) = α × Price(t) + (1-α) × EMA(t-1), where α = 2/(period+1)
        - ADX measures the strength of trend based on directional indicators
        """
