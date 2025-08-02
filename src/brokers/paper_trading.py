"""
Paper Trading Broker Implementation

A simulated broker for testing trading strategies without real money.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from .base import BaseBroker, Order, Position, Account, OrderStatus, OrderType, OrderSide
from ..data.manager import data_manager
from ..utils import get_logger

logger = get_logger(__name__)

class PaperTradingBroker(BaseBroker):
    """Paper trading broker implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize paper trading broker.
        
        Args:
            config: Broker configuration
        """
        super().__init__(config)
        
        # Account setup
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission = config.get('commission', 0.001)
        self.slippage = config.get('slippage', 0.0005)
        
        # State
        self.account = Account()
        self.account.cash = self.initial_capital
        self.account.buying_power = self.initial_capital
        self.account.portfolio_value = self.initial_capital
        
        self.orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []
        
        logger.info(f"Paper trading broker initialized with ${self.initial_capital:,.2f}")
    
    async def connect(self) -> bool:
        """Connect to paper trading broker.
        
        Returns:
            True (always successful for paper trading)
        """
        self.is_connected = True
        logger.info("Connected to paper trading broker")
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from paper trading broker."""
        self.is_connected = False
        logger.info("Disconnected from paper trading broker")
    
    async def get_account_info(self) -> Account:
        """Get account information.
        
        Returns:
            Account information
        """
        await self._update_account_value()
        return self.account
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions.
        
        Returns:
            Dictionary of symbol -> position
        """
        await self._update_position_values()
        return self.account.positions
    
    async def place_order(self, order: Order) -> str:
        """Place a trading order.
        
        Args:
            order: Order to place
        
        Returns:
            Order ID
        """
        if not self.validate_order(order):
            raise ValueError("Invalid order")
        
        # Generate order ID
        order_id = str(uuid.uuid4())
        order.order_id = order_id
        order.created_at = datetime.now()
        order.status = OrderStatus.PENDING
        
        # Store order
        self.orders[order_id] = order
        
        logger.info(
            f"Order placed: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {order.order_type.value} (ID: {order_id})"
        )
        
        # For market orders, execute immediately
        if order.order_type == OrderType.MARKET:
            await self._execute_order(order)
        
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if cancelled successfully
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        # Move to history
        self._order_history.append(order)
        del self.orders[order_id]
        
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status.
        
        Args:
            order_id: Order ID
        
        Returns:
            Order with current status
        """
        if order_id in self.orders:
            return self.orders[order_id]
        
        # Check history
        for order in self._order_history:
            if order.order_id == order_id:
                return order
        
        raise ValueError(f"Order not found: {order_id}")
    
    async def get_open_orders(self) -> List[Order]:
        """Get all open orders.
        
        Returns:
            List of open orders
        """
        return [order for order in self.orders.values() 
                if order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]]
    
    async def _execute_order(self, order: Order) -> None:
        """Execute an order.
        
        Args:
            order: Order to execute
        """
        try:
            # Get current market price
            current_price = await self._get_current_price(order.symbol)
            
            if current_price is None:
                order.status = OrderStatus.REJECTED
                logger.error(f"Could not get price for {order.symbol}")
                return
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                execution_price = current_price * (1 + self.slippage)
            else:
                execution_price = current_price * (1 - self.slippage)
            
            # Check if we can afford the order
            order_value = order.quantity * execution_price
            commission_cost = order_value * self.commission
            total_cost = order_value + commission_cost
            
            if order.side == OrderSide.BUY and total_cost > self.account.cash:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Insufficient cash for order {order.order_id}")
                return
            
            # Execute the order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.updated_at = datetime.now()
            
            # Update positions
            await self._update_position(order.symbol, order.quantity, execution_price, order.side)
            
            # Update cash
            if order.side == OrderSide.BUY:
                self.account.cash -= total_cost
            else:
                self.account.cash += order_value - commission_cost
            
            # Move to history
            self._order_history.append(order)
            del self.orders[order.order_id]
            
            logger.info(
                f"Order executed: {order.side.value} {order.quantity} {order.symbol} "
                f"@ ${execution_price:.2f} (ID: {order.order_id})"
            )
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Error executing order {order.order_id}: {e}")
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Current price or None if not available
        """
        try:
            # For paper trading, we'll use the latest close price
            # In a real implementation, this would be real-time data
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=5)
            
            data = data_manager.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                frequency='1d'
            )
            
            if not data.empty:
                return float(data['close'].iloc[-1])
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    async def _update_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: OrderSide
    ) -> None:
        """Update position after trade execution.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            side: Order side
        """
        # Adjust quantity based on side
        if side == OrderSide.SELL:
            quantity = -quantity
        
        if symbol in self.account.positions:
            position = self.account.positions[symbol]
            
            # Calculate new average price
            old_value = position.quantity * position.avg_price
            new_value = quantity * price
            total_quantity = position.quantity + quantity
            
            if total_quantity == 0:
                # Position closed
                del self.account.positions[symbol]
            else:
                position.avg_price = (old_value + new_value) / total_quantity
                position.quantity = total_quantity
        else:
            # New position
            if quantity != 0:
                self.account.positions[symbol] = Position(symbol, quantity, price)
    
    async def _update_position_values(self) -> None:
        """Update market values of all positions."""
        for symbol, position in self.account.positions.items():
            current_price = await self._get_current_price(symbol)
            
            if current_price:
                position.market_value = position.quantity * current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    async def _update_account_value(self) -> None:
        """Update account portfolio value."""
        await self._update_position_values()
        
        # Calculate total portfolio value
        positions_value = sum(pos.market_value for pos in self.account.positions.values())
        self.account.portfolio_value = self.account.cash + positions_value
        
        # Update buying power (simplified)
        self.account.buying_power = self.account.cash
