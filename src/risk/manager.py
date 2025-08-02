"""
Risk Management Module
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from ..utils import get_logger, config

logger = get_logger(__name__)

class RiskManager:
    """Risk management system for algorithmic trading."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """Initialize risk manager.
        
        Args:
            config_dict: Risk management configuration
        """
        self.config = config_dict or config.get('risk', {})
        
        # Position sizing parameters
        self.max_position_size = self.config.get('max_position_size', 0.1)
        self.kelly_fraction = self.config.get('position_sizing', {}).get('kelly_fraction', 0.25)
        
        # Risk limits
        self.stop_loss = self.config.get('stop_loss', 0.02)
        self.take_profit = self.config.get('take_profit', 0.04)
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 0.1)
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        
        # VaR parameters
        self.var_confidence = self.config.get('var_confidence', 0.95)
        
        logger.info(
            f"Risk Manager initialized: "
            f"Max position size: {self.max_position_size:.1%}, "
            f"Stop loss: {self.stop_loss:.1%}, "
            f"VaR confidence: {self.var_confidence:.1%}"
        )
    
    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        volatility: float = None,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        method: str = 'fixed_fraction'
    ) -> int:
        """Calculate optimal position size.
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            portfolio_value: Total portfolio value
            volatility: Asset volatility (optional)
            win_rate: Historical win rate (optional)
            avg_win: Average winning trade (optional)
            avg_loss: Average losing trade (optional)
            method: Position sizing method ('fixed_fraction', 'kelly', 'volatility_target')
        
        Returns:
            Number of shares to trade
        """
        if method == 'kelly' and all(x is not None for x in [win_rate, avg_win, avg_loss]):
            return self._kelly_position_size(
                signal_strength, current_price, portfolio_value, 
                win_rate, avg_win, avg_loss
            )
        
        elif method == 'volatility_target' and volatility is not None:
            return self._volatility_target_position_size(
                signal_strength, current_price, portfolio_value, volatility
            )
        
        else:
            return self._fixed_fraction_position_size(
                signal_strength, current_price, portfolio_value
            )
    
    def _fixed_fraction_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float
    ) -> int:
        """Calculate position size using fixed fraction method.
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            portfolio_value: Total portfolio value
        
        Returns:
            Number of shares to trade
        """
        # Adjust position size based on signal strength
        adjusted_fraction = self.max_position_size * signal_strength
        position_value = portfolio_value * adjusted_fraction
        
        shares = int(position_value / current_price)
        
        logger.debug(
            f"Fixed fraction position sizing: "
            f"Signal strength: {signal_strength:.2f}, "
            f"Position value: ${position_value:,.2f}, "
            f"Shares: {shares}"
        )
        
        return shares
    
    def _kelly_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> int:
        """Calculate position size using Kelly Criterion.
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            portfolio_value: Total portfolio value
            win_rate: Historical win rate
            avg_win: Average winning trade return
            avg_loss: Average losing trade return
        
        Returns:
            Number of shares to trade
        """
        # Kelly fraction = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-p
        
        if avg_loss == 0 or avg_win <= 0:
            return self._fixed_fraction_position_size(signal_strength, current_price, portfolio_value)
        
        b = abs(avg_win / avg_loss)  # Odds ratio
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor and signal strength
        safe_kelly = kelly_fraction * self.kelly_fraction * signal_strength
        
        # Cap at maximum position size
        final_fraction = min(safe_kelly, self.max_position_size)
        
        position_value = portfolio_value * max(0, final_fraction)
        shares = int(position_value / current_price)
        
        logger.debug(
            f"Kelly position sizing: "
            f"Kelly fraction: {kelly_fraction:.3f}, "
            f"Safe Kelly: {safe_kelly:.3f}, "
            f"Shares: {shares}"
        )
        
        return shares
    
    def _volatility_target_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        target_vol: float = 0.15
    ) -> int:
        """Calculate position size targeting specific volatility.
        
        Args:
            signal_strength: Signal strength (0-1)
            current_price: Current asset price
            portfolio_value: Total portfolio value
            volatility: Asset volatility
            target_vol: Target portfolio volatility
        
        Returns:
            Number of shares to trade
        """
        if volatility <= 0:
            return self._fixed_fraction_position_size(signal_strength, current_price, portfolio_value)
        
        # Position fraction to achieve target volatility
        target_fraction = target_vol / volatility
        
        # Adjust for signal strength
        adjusted_fraction = target_fraction * signal_strength
        
        # Cap at maximum position size
        final_fraction = min(adjusted_fraction, self.max_position_size)
        
        position_value = portfolio_value * final_fraction
        shares = int(position_value / current_price)
        
        logger.debug(
            f"Volatility target position sizing: "
            f"Target vol: {target_vol:.1%}, "
            f"Asset vol: {volatility:.1%}, "
            f"Shares: {shares}"
        )
        
        return shares
    
    def check_risk_limits(
        self,
        current_portfolio_value: float,
        max_portfolio_value: float,
        current_positions: Dict[str, Dict[str, Any]],
        proposed_trade: Dict[str, Any] = None
    ) -> Tuple[bool, List[str]]:
        """Check if trade violates risk limits.
        
        Args:
            current_portfolio_value: Current portfolio value
            max_portfolio_value: Maximum historical portfolio value
            current_positions: Current portfolio positions
            proposed_trade: Proposed trade details
        
        Returns:
            Tuple of (is_allowed, violations)
        """
        violations = []
        
        # Check maximum drawdown
        current_drawdown = (max_portfolio_value - current_portfolio_value) / max_portfolio_value
        if current_drawdown > self.max_drawdown:
            violations.append(f"Maximum drawdown exceeded: {current_drawdown:.1%} > {self.max_drawdown:.1%}")
        
        # Check portfolio concentration
        if proposed_trade:
            symbol = proposed_trade.get('symbol')
            trade_value = proposed_trade.get('value', 0)
            
            # Calculate position concentration after trade
            new_position_value = current_positions.get(symbol, {}).get('value', 0) + trade_value
            concentration = abs(new_position_value) / current_portfolio_value
            
            if concentration > self.max_position_size:
                violations.append(
                    f"Position concentration too high for {symbol}: "
                    f"{concentration:.1%} > {self.max_position_size:.1%}"
                )
        
        # Check total portfolio risk
        total_risk = sum(
            abs(pos.get('value', 0)) for pos in current_positions.values()
        ) / current_portfolio_value
        
        if total_risk > self.max_portfolio_risk:
            violations.append(f"Total portfolio risk too high: {total_risk:.1%} > {self.max_portfolio_risk:.1%}")
        
        is_allowed = len(violations) == 0
        
        if violations:
            logger.warning(f"Risk limit violations: {'; '.join(violations)}")
        
        return is_allowed, violations
    
    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        position_type: str,
        volatility: float = None,
        atr: float = None
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            position_type: 'long' or 'short'
            volatility: Asset volatility (optional)
            atr: Average True Range (optional)
        
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if position_type.lower() == 'long':
            stop_loss_price = entry_price * (1 - self.stop_loss)
            take_profit_price = entry_price * (1 + self.take_profit)
        else:  # short
            stop_loss_price = entry_price * (1 + self.stop_loss)
            take_profit_price = entry_price * (1 - self.take_profit)
        
        # Adjust based on volatility if available
        if atr and atr > 0:
            atr_multiplier = 2.0
            if position_type.lower() == 'long':
                stop_loss_price = max(stop_loss_price, entry_price - atr * atr_multiplier)
                take_profit_price = min(take_profit_price, entry_price + atr * atr_multiplier * 2)
            else:
                stop_loss_price = min(stop_loss_price, entry_price + atr * atr_multiplier)
                take_profit_price = max(take_profit_price, entry_price - atr * atr_multiplier * 2)
        
        return stop_loss_price, take_profit_price
    
    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = None,
        method: str = 'historical'
    ) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Portfolio returns series
            confidence: Confidence level (default from config)
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
        
        Returns:
            VaR value
        """
        if confidence is None:
            confidence = self.var_confidence
        
        if returns.empty or returns.isna().all():
            return 0.0
        
        if method == 'historical':
            return self._historical_var(returns, confidence)
        elif method == 'parametric':
            return self._parametric_var(returns, confidence)
        else:
            return self._historical_var(returns, confidence)  # Fallback
    
    def _historical_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate historical VaR.
        
        Args:
            returns: Returns series
            confidence: Confidence level
        
        Returns:
            Historical VaR
        """
        return np.percentile(returns.dropna(), (1 - confidence) * 100)
    
    def _parametric_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate parametric VaR assuming normal distribution.
        
        Args:
            returns: Returns series
            confidence: Confidence level
        
        Returns:
            Parametric VaR
        """
        returns_clean = returns.dropna()
        if len(returns_clean) < 2:
            return 0.0
        
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        # Z-score for given confidence level
        z_score = norm.ppf(1 - confidence)
        
        var = mean_return + z_score * std_return
        return var
    
    def calculate_expected_shortfall(
        self,
        returns: pd.Series,
        confidence: float = None
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Portfolio returns series
            confidence: Confidence level
        
        Returns:
            Expected Shortfall
        """
        if confidence is None:
            confidence = self.var_confidence
        
        if returns.empty:
            return 0.0
        
        var = self.calculate_var(returns, confidence, 'historical')
        
        # Expected Shortfall is the mean of returns below VaR
        tail_returns = returns[returns <= var]
        
        if tail_returns.empty:
            return var
        
        return tail_returns.mean()
    
    def generate_risk_report(
        self,
        portfolio_returns: pd.Series,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """Generate comprehensive risk report.
        
        Args:
            portfolio_returns: Portfolio returns series
            positions: Current positions
            portfolio_value: Current portfolio value
        
        Returns:
            Risk report dictionary
        """
        if portfolio_returns.empty:
            return {'error': 'No return data available'}
        
        # Basic risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        var_1d = self.calculate_var(portfolio_returns, 0.95)
        var_5d = self.calculate_var(portfolio_returns, 0.99)
        expected_shortfall = self.calculate_expected_shortfall(portfolio_returns, 0.95)
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        current_drawdown = drawdowns.iloc[-1]
        
        # Position analysis
        total_exposure = sum(abs(pos.get('value', 0)) for pos in positions.values())
        gross_exposure = total_exposure / portfolio_value if portfolio_value > 0 else 0
        
        largest_position = max(
            (abs(pos.get('value', 0)) / portfolio_value for pos in positions.values()),
            default=0
        )
        
        report = {
            'portfolio_value': portfolio_value,
            'volatility_annualized': volatility,
            'var_95_1day': var_1d,
            'var_99_1day': var_5d,
            'expected_shortfall_95': expected_shortfall,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'gross_exposure': gross_exposure,
            'largest_position_pct': largest_position,
            'num_positions': len(positions),
            'risk_limits': {
                'max_position_size': self.max_position_size,
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_drawdown_limit': self.max_drawdown,
                'stop_loss': self.stop_loss
            }
        }
        
        # Risk warnings
        warnings = []
        if current_drawdown < -self.max_drawdown * 0.8:
            warnings.append("Approaching maximum drawdown limit")
        
        if gross_exposure > self.max_portfolio_risk * 1.2:
            warnings.append("High portfolio leverage")
        
        if largest_position > self.max_position_size * 1.1:
            warnings.append("Position concentration risk")
        
        report['warnings'] = warnings
        
        logger.info(f"Risk report generated: VaR(95%)={var_1d:.1%}, Vol={volatility:.1%}")
        
        return report
