"""
Base Strategy Class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)

class SignalType(Enum):
    """Trading signal types."""
    BUY = 1
    SELL = -1
    HOLD = 0

class Position(Enum):
    """Position types."""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

class Signal:
    """Trading signal representation."""
    
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        signal_type: SignalType,
        strength: float = 1.0,
        metadata: Dict[str, Any] = None
    ):
        """Initialize trading signal.
        
        Args:
            timestamp: Signal timestamp
            symbol: Trading symbol
            signal_type: Type of signal (BUY, SELL, HOLD)
            strength: Signal strength (0-1)
            metadata: Additional signal metadata
        """
        self.timestamp = timestamp
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.metadata = metadata or {}

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        self.name = name
        self.config = config
        self.position = Position.NEUTRAL
        self.signals_history: List[Signal] = []
        self.performance_metrics: Dict[str, Any] = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame with OHLCV columns
        
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators required for the strategy.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            DataFrame with calculated indicators
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for strategy.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if data.empty:
            logger.warning(f"Empty data provided to {self.name}")
            return False
        
        if not all(col in data.columns for col in required_columns):
            logger.warning(f"Missing required columns in data for {self.name}")
            return False
        
        if data.isnull().any().any():
            logger.warning(f"Data contains null values for {self.name}")
            return False
        
        return True
    
    def update_position(self, signal: Signal) -> None:
        """Update strategy position based on signal.
        
        Args:
            signal: Trading signal
        """
        if signal.signal_type == SignalType.BUY:
            self.position = Position.LONG
        elif signal.signal_type == SignalType.SELL:
            self.position = Position.SHORT
        else:
            self.position = Position.NEUTRAL
    
    def add_signal(self, signal: Signal) -> None:
        """Add signal to history.
        
        Args:
            signal: Trading signal to add
        """
        self.signals_history.append(signal)
        self.update_position(signal)
        
        logger.info(
            f"{self.name}: {signal.signal_type.name} signal for {signal.symbol} "
            f"at {signal.timestamp} (strength: {signal.strength:.2f})"
        )
    
    def get_current_position(self) -> Position:
        """Get current strategy position.
        
        Returns:
            Current position
        """
        return self.position
    
    def get_signals_history(self, limit: int = None) -> List[Signal]:
        """Get signals history.
        
        Args:
            limit: Maximum number of signals to return
        
        Returns:
            List of historical signals
        """
        if limit:
            return self.signals_history[-limit:]
        return self.signals_history
    
    def reset(self) -> None:
        """Reset strategy state."""
        self.position = Position.NEUTRAL
        self.signals_history = []
        self.performance_metrics = {}
        logger.info(f"Strategy {self.name} reset")
    
    def get_config_parameter(self, key: str, default: Any = None) -> Any:
        """Get configuration parameter.
        
        Args:
            key: Parameter key
            default: Default value
        
        Returns:
            Parameter value
        """
        return self.config.get(key, default)
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic performance metrics.
        
        Args:
            returns: Strategy returns series
        
        Returns:
            Dictionary of performance metrics
        """
        if returns.empty:
            return {}
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * (252 ** 0.5),
            'sharpe_ratio': (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'win_rate': (returns > 0).mean(),
            'avg_win': returns[returns > 0].mean() if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() if (returns < 0).any() else 0
        }
        
        self.performance_metrics = metrics
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            returns: Returns series
        
        Returns:
            Maximum drawdown
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def __str__(self) -> str:
        """String representation of strategy."""
        return f"{self.name} (Position: {self.position.name}, Signals: {len(self.signals_history)})"
