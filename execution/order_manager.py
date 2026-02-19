"""
Order management and order generation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Order object."""
    order_id: str
    commodity: str
    contract: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class OrderManager:
    """
    Manage order lifecycle.
    
    Responsibilities:
    - Generate orders from position changes
    - Track order status
    - Handle partial fills
    - Order validation
    """
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        
    def generate_orders_from_positions(self,
                                      current_positions: Dict[str, float],
                                      target_positions: Dict[str, float],
                                      contracts: Dict[str, str],
                                      current_prices: Dict[str, float]) -> List[Order]:
        """
        Generate orders to move from current to target positions.
        
        Args:
            current_positions: Dict mapping commodity -> current position
            target_positions: Dict mapping commodity -> target position
            contracts: Dict mapping commodity -> contract symbol
            current_prices: Dict mapping commodity -> current price
            
        Returns:
            List of orders
        """
        orders = []
        
        for commodity in target_positions.keys():
            current = current_positions.get(commodity, 0.0)
            target = target_positions.get(commodity, 0.0)
            
            delta = target - current
            
            # Skip if no change needed
            if abs(delta) < 0.01:  # Minimum order size threshold
                continue
            
            # Determine side
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            quantity = abs(delta)
            
            # Create order
            order = self.create_order(
                commodity=commodity,
                contract=contracts.get(commodity, 'UNKNOWN'),
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            orders.append(order)
            
            logger.info(f"Generated order: {side.value} {quantity:.2f} {commodity} @ MARKET")
        
        return orders
    
    def create_order(self,
                    commodity: str,
                    contract: str,
                    side: OrderSide,
                    quantity: float,
                    order_type: OrderType,
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> Order:
        """Create a new order."""
        self.order_counter += 1
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            commodity=commodity,
            contract=contract,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        self.orders[order_id] = order
        
        return order
    
    def validate_order(self, order: Order, account_value: float, max_order_value: float) -> tuple[bool, str]:
        """
        Validate order before submission.
        
        Returns:
            (is_valid, reason)
        """
        # Check quantity is positive
        if order.quantity <= 0:
            return False, "Quantity must be positive"
        
        # Check order value doesn't exceed limits
        if order.limit_price:
            order_value = order.quantity * order.limit_price
            if order_value > max_order_value:
                return False, f"Order value {order_value:.0f} exceeds limit {max_order_value:.0f}"
        
        # Check account has sufficient margin (simplified)
        # In production, calculate actual margin requirements
        
        return True, "OK"
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           filled_price: Optional[float] = None,
                           filled_quantity: Optional[float] = None):
        """Update order status."""
        if order_id not in self.orders:
            logger.error(f"Order {order_id} not found")
            return
        
        order = self.orders[order_id]
        order.status = status
        
        if filled_price:
            order.filled_price = filled_price
        
        if filled_quantity:
            order.filled_quantity = filled_quantity
        
        if status == OrderStatus.FILLED:
            order.filled_at = datetime.now()
        
        logger.info(f"Order {order_id} updated: {status.value}")
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return [o for o in self.orders.values() 
                if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]]
    
    def get_filled_orders(self, start_date: Optional[datetime] = None) -> List[Order]:
        """Get filled orders."""
        filled = [o for o in self.orders.values() if o.status == OrderStatus.FILLED]
        
        if start_date:
            filled = [o for o in filled if o.filled_at and o.filled_at >= start_date]
        
        return filled
    
    def get_order_history(self) -> pd.DataFrame:
        """Get order history as DataFrame."""
        if not self.orders:
            return pd.DataFrame()
        
        records = []
        for order in self.orders.values():
            records.append({
                'order_id': order.order_id,
                'commodity': order.commodity,
                'contract': order.contract,
                'side': order.side.value,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'status': order.status.value,
                'created_at': order.created_at,
                'filled_at': order.filled_at,
                'filled_price': order.filled_price,
                'filled_quantity': order.filled_quantity
            })
        
        return pd.DataFrame(records)