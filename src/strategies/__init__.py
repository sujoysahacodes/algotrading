"""
Strategies module - Trading strategies implementation
"""

from .base import BaseStrategy, Signal, SignalType, Position
from .ema_adx import EMAAdxStrategy
from .zscore_reversion import ZScoreMeanReversionStrategy
from .fx_trend import FXTrendStrategy

__all__ = [
    'BaseStrategy',
    'Signal',
    'SignalType', 
    'Position',
    'EMAAdxStrategy',
    'ZScoreMeanReversionStrategy',
    'FXTrendStrategy'
]
