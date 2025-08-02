"""
FX Trend Strategy - Custom trend indicator for FX markets

This strategy implements the custom FX trend indicator described in the requirements:
1. Resample prices at regular intervals
2. Calculate average price over longer periods  
3. Compute ratio of short-term to long-term average
4. Generate signals based on ratio thresholds
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from .base import BaseStrategy, Signal, SignalType
from ..utils import get_logger

logger = get_logger(__name__)

class FXTrendStrategy(BaseStrategy):
    """FX Trend following strategy using price ratio method."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FX Trend strategy.
        
        Args:
            config: Strategy configuration containing:
                - short_interval: Short-term sampling interval (default: '30s')
                - long_interval: Long-term averaging interval (default: '5min')
                - trend_threshold: Threshold for trend detection (default: 0.02)
                - no_trend_band: Band around 1.0 for no trend (default: 0.01)
        """
        super().__init__("FX-Trend", config)
        
        self.short_interval = self.get_config_parameter('short_interval', '30s')
        self.long_interval = self.get_config_parameter('long_interval', '5min')
        self.trend_threshold = self.get_config_parameter('trend_threshold', 0.02)
        self.no_trend_band = self.get_config_parameter('no_trend_band', 0.01)
        
        logger.info(
            f"FX Trend Strategy initialized: "
            f"Short interval={self.short_interval}, "
            f"Long interval={self.long_interval}, "
            f"Trend threshold={self.trend_threshold}"
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate FX trend indicators.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            DataFrame with calculated indicators
        """
        df = data.copy()
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Data index is not datetime, attempting conversion")
            df.index = pd.to_datetime(df.index)
        
        # Step 1: Resample prices at short intervals (e.g., 30 seconds)
        # Use the average of OHLC as representative price
        df['price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Resample to short interval
        try:
            short_resampled = df['price'].resample(self.short_interval).mean()
            short_resampled = short_resampled.dropna()
        except Exception as e:
            logger.warning(f"Failed to resample to {self.short_interval}, using original data: {e}")
            short_resampled = df['price']
        
        # Step 2: Calculate average price over longer period (e.g., 5 minutes)
        try:
            long_avg = short_resampled.resample(self.long_interval).mean()
            long_avg = long_avg.dropna()
        except Exception as e:
            logger.warning(f"Failed to resample to {self.long_interval}, using rolling mean: {e}")
            # Fallback to rolling window (approximate conversion)
            window_size = self._convert_interval_to_periods(self.long_interval, self.short_interval)
            long_avg = short_resampled.rolling(window=window_size).mean()
        
        # Step 3: Compute ratio of short-term price to long-term average
        # Align the series by reindexing
        aligned_short = short_resampled.reindex(long_avg.index, method='ffill')
        
        df_result = pd.DataFrame(index=long_avg.index)
        df_result['short_price'] = aligned_short
        df_result['long_avg'] = long_avg
        df_result['price_ratio'] = aligned_short / long_avg
        
        # Calculate trend signals
        df_result['trend_signal'] = 0
        
        # Uptrend: ratio significantly above 1
        uptrend_condition = df_result['price_ratio'] > (1 + self.trend_threshold)
        df_result.loc[uptrend_condition, 'trend_signal'] = 1
        
        # Downtrend: ratio significantly below 1  
        downtrend_condition = df_result['price_ratio'] < (1 - self.trend_threshold)
        df_result.loc[downtrend_condition, 'trend_signal'] = -1
        
        # No trend: ratio near 1 (within no_trend_band)
        no_trend_condition = (
            (df_result['price_ratio'] >= (1 - self.no_trend_band)) &
            (df_result['price_ratio'] <= (1 + self.no_trend_band))
        )
        df_result.loc[no_trend_condition, 'trend_signal'] = 0
        
        # Calculate signal changes
        df_result['trend_signal_change'] = df_result['trend_signal'].diff()
        
        # Calculate trend strength
        df_result['trend_strength'] = abs(df_result['price_ratio'] - 1)
        
        return df_result
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate FX trend signals.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            List of trading signals
        """
        if not self.validate_data(data):
            return []
        
        # Calculate indicators
        df = self.calculate_indicators(data)
        
        if df.empty:
            logger.warning("No indicator data generated")
            return []
        
        signals = []
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            previous_row = df.iloc[i-1]
            
            # Generate signals on trend changes
            if current_row['trend_signal_change'] != 0:
                
                # Bullish trend signal
                if current_row['trend_signal'] == 1 and previous_row['trend_signal'] != 1:
                    signal_strength = min(current_row['trend_strength'] / 0.1, 1.0)
                    
                    signal = Signal(
                        timestamp=current_row.name,
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal_type=SignalType.BUY,
                        strength=signal_strength,
                        metadata={
                            'price_ratio': current_row['price_ratio'],
                            'short_price': current_row['short_price'],
                            'long_avg': current_row['long_avg'],
                            'trend_strength': current_row['trend_strength'],
                            'signal_reason': 'fx_uptrend'
                        }
                    )
                    signals.append(signal)
                    self.add_signal(signal)
                
                # Bearish trend signal
                elif current_row['trend_signal'] == -1 and previous_row['trend_signal'] != -1:
                    signal_strength = min(current_row['trend_strength'] / 0.1, 1.0)
                    
                    signal = Signal(
                        timestamp=current_row.name,
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal_type=SignalType.SELL,
                        strength=signal_strength,
                        metadata={
                            'price_ratio': current_row['price_ratio'],
                            'short_price': current_row['short_price'],
                            'long_avg': current_row['long_avg'],
                            'trend_strength': current_row['trend_strength'],
                            'signal_reason': 'fx_downtrend'
                        }
                    )
                    signals.append(signal)
                    self.add_signal(signal)
                
                # No trend / exit signal
                elif current_row['trend_signal'] == 0 and previous_row['trend_signal'] != 0:
                    signal = Signal(
                        timestamp=current_row.name,
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal_type=SignalType.HOLD,
                        strength=1.0 - current_row['trend_strength'] / 0.05,
                        metadata={
                            'price_ratio': current_row['price_ratio'],
                            'signal_reason': 'fx_no_trend'
                        }
                    )
                    signals.append(signal)
                    self.add_signal(signal)
        
        logger.info(f"Generated {len(signals)} signals for FX Trend strategy")
        return signals
    
    def _convert_interval_to_periods(self, target_interval: str, base_interval: str) -> int:
        """Convert time interval to number of periods.
        
        Args:
            target_interval: Target interval (e.g., '5min')
            base_interval: Base interval (e.g., '30s')
        
        Returns:
            Number of base periods in target interval
        """
        # Simple conversion for common intervals
        interval_seconds = {
            's': 1, 'sec': 1, 'second': 1,
            'min': 60, 'minute': 60,
            'h': 3600, 'hour': 3600,
            'd': 86400, 'day': 86400
        }
        
        def parse_interval(interval_str):
            # Extract number and unit
            import re
            match = re.match(r'(\d+)([a-zA-Z]+)', interval_str)
            if match:
                num, unit = match.groups()
                return int(num) * interval_seconds.get(unit, 60)
            return 60  # Default to 60 seconds
        
        target_seconds = parse_interval(target_interval)
        base_seconds = parse_interval(base_interval)
        
        return max(1, target_seconds // base_seconds)
    
    def calibrate_parameters(self, data: pd.DataFrame, calibrations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calibrate strategy parameters over different timeframes.
        
        Args:
            data: Historical data for calibration
            calibrations: List of parameter combinations to test
        
        Returns:
            Best performing parameters
        """
        best_params = None
        best_score = -np.inf
        
        for params in calibrations:
            # Temporarily update parameters
            original_params = {
                'trend_threshold': self.trend_threshold,
                'no_trend_band': self.no_trend_band
            }
            
            self.trend_threshold = params.get('trend_threshold', self.trend_threshold)
            self.no_trend_band = params.get('no_trend_band', self.no_trend_band)
            
            try:
                # Generate signals with current parameters
                signals = self.generate_signals(data)
                
                if len(signals) == 0:
                    continue
                
                # Simple scoring based on signal frequency and strength
                signal_strength_avg = np.mean([s.strength for s in signals])
                signal_frequency = len(signals) / len(data)
                
                # Balanced score favoring moderate frequency and high strength
                score = signal_strength_avg * (1 - abs(signal_frequency - 0.1))
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_params['score'] = score
                
            except Exception as e:
                logger.warning(f"Calibration failed for params {params}: {e}")
            
            # Restore original parameters
            self.trend_threshold = original_params['trend_threshold']
            self.no_trend_band = original_params['no_trend_band']
        
        if best_params:
            logger.info(f"Best FX Trend parameters: {best_params}")
        
        return best_params or {}
    
    def get_strategy_description(self) -> str:
        """Get detailed strategy description.
        
        Returns:
            Strategy description string
        """
        return f"""
        FX Trend Strategy (Price Ratio Method)
        ======================================
        
        Parameters:
        - Short Interval: {self.short_interval}
        - Long Interval: {self.long_interval}
        - Trend Threshold: {self.trend_threshold}
        - No Trend Band: {self.no_trend_band}
        
        Logic:
        1. Resample prices at short intervals ({self.short_interval})
        2. Calculate average price over longer periods ({self.long_interval})
        3. Compute ratio = short_term_price / long_term_average
        4. Generate signals based on ratio:
           - Uptrend: ratio > 1 + threshold
           - Downtrend: ratio < 1 - threshold
           - No trend: ratio ≈ 1 (within no_trend_band)
        
        Mathematical Formula:
        - Ratio(t) = P_short(t) / P_long_avg(t)
        - Uptrend Signal: Ratio > 1 + threshold
        - Downtrend Signal: Ratio < 1 - threshold
        - No Trend: 1 - band < Ratio < 1 + band
        
        This approach is particularly effective for FX markets where
        short-term deviations from longer-term averages can indicate
        trend initiation or continuation.
        """
