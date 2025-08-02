"""
Base Broker Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class Order:
    """Order representation."""
    
    def __init__(
        self,
        symbol: str,
        quantity: int,
        side: OrderSide,
        order_type: OrderType,
        price: float = None,
        stop_price: float = None,
        time_in_force: str = "day"
    ):
        """Initialize order.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            side: Buy or sell
            order_type: Order type
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
        """
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.order_type = order_type
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        
        # Set by broker
        self.order_id: Optional[str] = None
        self.status: OrderStatus = OrderStatus.PENDING
        self.filled_quantity: int = 0
        self.avg_fill_price: float = 0.0
        self.created_at: Optional[datetime] = None
        self.updated_at: Optional[datetime] = None

class Position:
    """Position representation."""
    
    def __init__(self, symbol: str, quantity: int, avg_price: float):
        """Initialize position.
        
        Args:
            symbol: Trading symbol
            quantity: Position quantity (positive for long, negative for short)
            avg_price: Average entry price
        """
        self.symbol = symbol
        self.quantity = quantity
        self.avg_price = avg_price
        self.market_value: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0

class Account:
    """Account information."""
    
    def __init__(self):
        """Initialize account."""
        self.buying_power: float = 0.0
        self.cash: float = 0.0
        self.portfolio_value: float = 0.0
        self.positions: Dict[str, Position] = {}
        self.day_trades: int = 0
        self.is_pattern_day_trader: bool = False

class BaseBroker(ABC):
    """Abstract base class for broker implementations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize broker.
        
        Args:
            config: Broker configuration
        """
        self.config = config
        self.is_connected = False
        self.is_paper_trading = config.get('paper_trading', True)
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API.
        
        Returns:
            True if connected successfully
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker API."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Account:
        """Get account information.
        
        Returns:
            Account information
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions.
        
        Returns:
            Dictionary of symbol -> position
        """
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """Place a trading order.
        
        Args:
            order: Order to place
        
        Returns:
            Order ID
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order ID to cancel
        
        Returns:
            True if cancelled successfully
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status.
        
        Args:
            order_id: Order ID
        
        Returns:
            Order with current status
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """Get all open orders.
        
        Returns:
            List of open orders
        """
        pass
    
    def validate_order(self, order: Order) -> bool:
        """Validate order before placing.
        
        Args:
            order: Order to validate
        
        Returns:
            True if order is valid
        """
        # Basic validation
        if order.quantity <= 0:
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            return False
        
        return True
