"""
Backtesting Engine for Strategy Evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from ..strategies.base import BaseStrategy, Signal, SignalType
from ..utils import get_logger, config

logger = get_logger(__name__)

class BacktestResult:
    """Container for backtest results."""
    
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float
    ):
        """Initialize backtest result.
        
        Args:
            strategy_name: Name of tested strategy
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Performance tracking
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = []
        self.returns: List[float] = []
        self.positions: List[int] = []
        self.timestamps: List[datetime] = []
        
        # Summary metrics
        self.total_return: float = 0.0
        self.annualized_return: float = 0.0
        self.volatility: float = 0.0
        self.sharpe_ratio: float = 0.0
        self.max_drawdown: float = 0.0
        self.win_rate: float = 0.0
        self.num_trades: int = 0
        self.avg_trade_return: float = 0.0

class BacktestEngine:
    """Backtesting engine for strategy evaluation."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize backtest engine.
        
        Args:
            config_dict: Backtesting configuration
        """
        self.config = config_dict or config.get('backtesting', {})
        
        self.initial_capital = self.config.get('initial_capital', 100000)
        self.commission = self.config.get('commission', 0.001)
        self.slippage = self.config.get('slippage', 0.0005)
        self.benchmark = self.config.get('benchmark', 'SPY')
        
        logger.info(
            f"Backtest Engine initialized: "
            f"Capital=${self.initial_capital:,.2f}, "
            f"Commission={self.commission:.3f}, "
            f"Slippage={self.slippage:.4f}"
        )
    
    def run_backtest(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> BacktestResult:
        """Run backtest for a strategy.
        
        Args:
            strategy: Trading strategy to test
            data: Historical market data
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            Backtesting results
        """
        logger.info(f"Starting backtest for {strategy.name} on {symbol}")
        
        # Validate data
        if data.empty:
            raise ValueError("Empty data provided for backtesting")
        
        # Set date range
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
        
        # Filter data by date range
        data = data.loc[start_date:end_date].copy()
        data.attrs['symbol'] = symbol
        
        # Initialize backtest result
        result = BacktestResult(
            strategy_name=strategy.name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital
        )
        
        # Reset strategy state
        strategy.reset()
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        if not signals:
            logger.warning(f"No signals generated for {strategy.name}")
            return result
        
        # Simulate trading
        result = self._simulate_trading(result, signals, data)
        
        # Calculate performance metrics
        self._calculate_performance_metrics(result)
        
        logger.info(
            f"Backtest completed for {strategy.name}: "
            f"Total Return: {result.total_return:.2%}, "
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}, "
            f"Max Drawdown: {result.max_drawdown:.2%}"
        )
        
        return result
    
    def _simulate_trading(
        self,
        result: BacktestResult,
        signals: List[Signal],
        data: pd.DataFrame
    ) -> BacktestResult:
        """Simulate trading based on signals.
        
        Args:
            result: Backtest result container
            signals: List of trading signals
            data: Market data
        
        Returns:
            Updated backtest result
        """
        cash = self.initial_capital
        position = 0  # Number of shares
        position_value = 0
        current_price = 0
        
        # Create a complete timeline
        signal_dict = {signal.timestamp: signal for signal in signals}
        
        for timestamp, row in data.iterrows():
            current_price = row['close']
            
            # Check for signals at this timestamp
            if timestamp in signal_dict:
                signal = signal_dict[timestamp]
                
                # Execute trade based on signal
                trade_result = self._execute_trade(
                    signal, current_price, cash, position
                )
                
                if trade_result:
                    cash = trade_result['cash']
                    position = trade_result['position']
                    result.trades.append(trade_result['trade_record'])
            
            # Calculate portfolio value
            position_value = position * current_price
            portfolio_value = cash + position_value
            
            # Record state
            result.portfolio_values.append(portfolio_value)
            result.positions.append(position)
            result.timestamps.append(timestamp)
            
            # Calculate returns
            if len(result.portfolio_values) > 1:
                prev_value = result.portfolio_values[-2]
                if prev_value > 0:
                    period_return = (portfolio_value - prev_value) / prev_value
                    result.returns.append(period_return)
                else:
                    result.returns.append(0.0)
            else:
                result.returns.append(0.0)
        
        return result
    
    def _execute_trade(
        self,
        signal: Signal,
        current_price: float,
        cash: float,
        current_position: int
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade based on signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            cash: Available cash
            current_position: Current position size
        
        Returns:
            Trade execution result
        """
        # Apply slippage
        if signal.signal_type == SignalType.BUY:
            execution_price = current_price * (1 + self.slippage)
        elif signal.signal_type == SignalType.SELL:
            execution_price = current_price * (1 - self.slippage)
        else:
            execution_price = current_price
        
        # Calculate position size based on signal strength and available capital
        portfolio_value = cash + (current_position * current_price)
        max_position_value = portfolio_value * 0.95  # Keep 5% cash buffer
        
        new_position = current_position
        trade_value = 0
        trade_shares = 0
        
        if signal.signal_type == SignalType.BUY and current_position <= 0:
            # Buy signal - calculate shares to buy
            available_cash = cash * 0.95  # Use 95% of available cash
            trade_shares = int(available_cash / execution_price)
            trade_value = trade_shares * execution_price
            
            if trade_shares > 0 and trade_value <= cash:
                new_position = current_position + trade_shares
                cash -= trade_value
                cash -= trade_value * self.commission  # Commission
        
        elif signal.signal_type == SignalType.SELL and current_position >= 0:
            # Sell signal - calculate shares to sell
            if current_position > 0:
                # Close existing long position
                trade_shares = current_position
                trade_value = trade_shares * execution_price
                new_position = 0
                cash += trade_value
                cash -= trade_value * self.commission  # Commission
            else:
                # Open short position (if allowed)
                available_cash = cash * 0.95
                trade_shares = int(available_cash / execution_price)
                trade_value = trade_shares * execution_price
                
                if trade_shares > 0:
                    new_position = -trade_shares
                    cash -= trade_value * self.commission  # Commission for short
        
        elif signal.signal_type == SignalType.HOLD:
            # Close any existing position
            if current_position != 0:
                trade_shares = abs(current_position)
                trade_value = trade_shares * execution_price
                
                if current_position > 0:
                    cash += trade_value
                else:
                    cash -= trade_value
                
                cash -= trade_value * self.commission
                new_position = 0
        
        # Record trade if position changed
        if new_position != current_position:
            trade_record = {
                'timestamp': signal.timestamp,
                'signal_type': signal.signal_type.name,
                'price': execution_price,
                'shares': trade_shares,
                'value': trade_value,
                'position_before': current_position,
                'position_after': new_position,
                'cash_after': cash,
                'signal_strength': signal.strength,
                'metadata': signal.metadata
            }
            
            return {
                'cash': cash,
                'position': new_position,
                'trade_record': trade_record
            }
        
        return None
    
    def _calculate_performance_metrics(self, result: BacktestResult) -> None:
        """Calculate performance metrics for backtest result.
        
        Args:
            result: Backtest result to update
        """
        if not result.portfolio_values:
            return
        
        # Basic metrics
        final_value = result.portfolio_values[-1]
        result.total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Convert returns to pandas Series for calculations
        returns_series = pd.Series(result.returns)
        
        if len(returns_series) > 0 and not returns_series.isna().all():
            # Annualized return
            trading_days = len(returns_series)
            years = trading_days / 252  # Assume 252 trading days per year
            if years > 0:
                result.annualized_return = (1 + result.total_return) ** (1/years) - 1
            
            # Volatility (annualized)
            result.volatility = returns_series.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            if result.volatility > 0:
                result.sharpe_ratio = (result.annualized_return - risk_free_rate) / result.volatility
            
            # Maximum drawdown
            portfolio_series = pd.Series(result.portfolio_values)
            rolling_max = portfolio_series.expanding().max()
            drawdowns = (portfolio_series - rolling_max) / rolling_max
            result.max_drawdown = drawdowns.min()
        
        # Trade-based metrics
        if result.trades:
            result.num_trades = len(result.trades)
            
            # Calculate individual trade returns
            trade_returns = []
            for i, trade in enumerate(result.trades):
                if i > 0 and trade['position_before'] != 0:
                    prev_trade = result.trades[i-1]
                    if prev_trade['position_after'] * trade['position_before'] < 0:
                        # Position closed
                        entry_price = prev_trade['price']
                        exit_price = trade['price']
                        
                        if prev_trade['position_after'] > 0:  # Long position
                            trade_return = (exit_price - entry_price) / entry_price
                        else:  # Short position
                            trade_return = (entry_price - exit_price) / entry_price
                        
                        trade_returns.append(trade_return)
            
            if trade_returns:
                trade_returns = np.array(trade_returns)
                result.win_rate = (trade_returns > 0).mean()
                result.avg_trade_return = trade_returns.mean()
    
    def compare_strategies(
        self,
        strategies: List[BaseStrategy],
        data: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """Compare multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            data: Market data
            symbol: Trading symbol
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for strategy in strategies:
            try:
                result = self.run_backtest(strategy, data, symbol)
                
                results.append({
                    'Strategy': result.strategy_name,
                    'Total Return': result.total_return,
                    'Annualized Return': result.annualized_return,
                    'Volatility': result.volatility,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Max Drawdown': result.max_drawdown,
                    'Win Rate': result.win_rate,
                    'Number of Trades': result.num_trades,
                    'Avg Trade Return': result.avg_trade_return
                })
                
            except Exception as e:
                logger.error(f"Error backtesting {strategy.name}: {e}")
                results.append({
                    'Strategy': strategy.name,
                    'Total Return': np.nan,
                    'Annualized Return': np.nan,
                    'Volatility': np.nan,
                    'Sharpe Ratio': np.nan,
                    'Max Drawdown': np.nan,
                    'Win Rate': np.nan,
                    'Number of Trades': 0,
                    'Avg Trade Return': np.nan
                })
        
        return pd.DataFrame(results)
