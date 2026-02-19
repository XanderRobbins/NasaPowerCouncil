"""
Trade execution engine.
"""
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger

from execution.order_manager import OrderManager, Order, OrderStatus, OrderSide
from execution.slippage_model import SlippageModel


class TradeExecutor:
    """
    Execute trades with slippage and transaction costs.
    
    Handles:
    - Order submission
    - Simulated fills (for backtesting)
    - Real broker integration (for live trading)
    """
    
    def __init__(self, 
                 order_manager: OrderManager,
                 slippage_model: SlippageModel,
                 paper_trading: bool = True):
        self.order_manager = order_manager
        self.slippage_model = slippage_model
        self.paper_trading = paper_trading
        
        # Broker connection (placeholder)
        self.broker = None
        
    def execute_orders(self,
                      orders: List[Order],
                      current_prices: Dict[str, float],
                      average_volumes: Dict[str, float]) -> Dict[str, Dict]:
        """
        Execute a list of orders.
        
        Args:
            orders: List of orders to execute
            current_prices: Dict mapping commodity -> current price
            average_volumes: Dict mapping commodity -> average volume
            
        Returns:
            Dict mapping order_id -> execution details
        """
        execution_results = {}
        
        for order in orders:
            result = self.execute_order(
                order,
                current_prices.get(order.commodity, 0),
                average_volumes.get(order.commodity, 10000)
            )
            
            execution_results[order.order_id] = result
        
        return execution_results
    
    def execute_order(self,
                     order: Order,
                     current_price: float,
                     average_volume: float) -> Dict:
        """
        Execute a single order.
        
        Args:
            order: Order to execute
            current_price: Current market price
            average_volume: Average daily volume
            
        Returns:
            Execution details
        """
        # Validate order
        is_valid, reason = self.order_manager.validate_order(order, account_value=1e6, max_order_value=1e5)
        
        if not is_valid:
            logger.error(f"Order validation failed: {reason}")
            self.order_manager.update_order_status(order.order_id, OrderStatus.REJECTED)
            return {'status': 'REJECTED', 'reason': reason}
        
        # Mark as submitted
        self.order_manager.update_order_status(order.order_id, OrderStatus.SUBMITTED)
        
        if self.paper_trading:
            # Simulate fill
            result = self._simulate_fill(order, current_price, average_volume)
        else:
            # Execute via broker
            result = self._execute_via_broker(order, current_price)
        
        return result
    
    def _simulate_fill(self, order: Order, mid_price: float, average_volume: float) -> Dict:
        """
        Simulate order fill (for backtesting or paper trading).
        """
        # Compute execution with slippage
        execution_details = self.slippage_model.simulate_fill(order, mid_price, average_volume)
        
        # Update order
        self.order_manager.update_order_status(
            order.order_id,
            OrderStatus.FILLED,
            filled_price=execution_details['execution_price'],
            filled_quantity=order.quantity
        )
        
        logger.info(f"[SIMULATED] Filled {order.side.value} {order.quantity:.2f} {order.commodity} "
                   f"@ {execution_details['execution_price']:.2f} "
                   f"(slippage: {execution_details['slippage']:.4f}, "
                   f"total cost: ${execution_details['total_cost']:.2f})")
        
        return {
            'status': 'FILLED',
            'order_id': order.order_id,
            **execution_details
        }
    
    def _execute_via_broker(self, order: Order, current_price: float) -> Dict:
        """
        Execute order via broker API.
        
        PLACEHOLDER - Implement with your broker's API
        (Interactive Brokers, TD Ameritrade, etc.)
        """
        if self.broker is None:
            logger.error("No broker connection configured")
            return {'status': 'FAILED', 'reason': 'No broker connection'}
        
        try:
            # Example for Interactive Brokers TWS API:
            # contract = self._create_ib_contract(order.contract)
            # ib_order = self._create_ib_order(order)
            # trade = self.broker.placeOrder(contract, ib_order)
            
            # Wait for fill...
            # fill_price = trade.orderStatus.avgFillPrice
            
            # For now, placeholder:
            fill_price = current_price
            
            self.order_manager.update_order_status(
                order.order_id,
                OrderStatus.FILLED,
                filled_price=fill_price,
                filled_quantity=order.quantity
            )
            
            logger.info(f"[LIVE] Filled {order.side.value} {order.quantity:.2f} {order.commodity} @ {fill_price:.2f}")
            
            return {
                'status': 'FILLED',
                'order_id': order.order_id,
                'execution_price': fill_price,
                'filled_quantity': order.quantity
            }
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            self.order_manager.update_order_status(order.order_id, OrderStatus.REJECTED)
            return {'status': 'FAILED', 'reason': str(e)}
    
    def get_total_transaction_costs(self, start_date: Optional[datetime] = None) -> float:
        """
        Calculate total transaction costs.
        
        Args:
            start_date: Calculate costs from this date onwards
            
        Returns:
            Total transaction costs in dollars
        """
        filled_orders = self.order_manager.get_filled_orders(start_date)
        
        total_costs = 0.0
        for order in filled_orders:
            if order.filled_price and order.filled_quantity:
                # Estimate costs (in production, track actual costs)
                costs = self.slippage_model.compute_transaction_costs(order)
                total_costs += costs
        
        return total_costs