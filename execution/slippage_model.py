"""
Slippage and transaction cost modeling.
"""
import numpy as np
from typing import Dict
from loguru import logger

from execution.order_manager import Order, OrderSide, OrderType


class SlippageModel:
    """
    Model slippage and transaction costs.
    
    Components:
    - Bid-ask spread
    - Market impact (function of order size)
    - Commission
    - Exchange fees
    """
    
    def __init__(self,
                 base_spread_bps: float = 2.0,
                 market_impact_coef: float = 0.1,
                 commission_per_contract: float = 2.50,
                 exchange_fee_per_contract: float = 1.00):
        """
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            market_impact_coef: Market impact coefficient
            commission_per_contract: Commission per contract
            exchange_fee_per_contract: Exchange fee per contract
        """
        self.base_spread_bps = base_spread_bps
        self.market_impact_coef = market_impact_coef
        self.commission_per_contract = commission_per_contract
        self.exchange_fee_per_contract = exchange_fee_per_contract
        
    def compute_slippage(self, 
                        order: Order,
                        mid_price: float,
                        average_volume: float) -> float:
        """
        Compute expected slippage for an order.
        
        Args:
            order: Order object
            mid_price: Mid-market price
            average_volume: Average daily volume
            
        Returns:
            Slippage in price units (positive = adverse)
        """
        # Component 1: Bid-ask spread
        spread_slippage = mid_price * (self.base_spread_bps / 10000) / 2
        
        # Component 2: Market impact
        # Impact increases with order size relative to volume
        volume_ratio = order.quantity / (average_volume + 1)
        impact_slippage = mid_price * self.market_impact_coef * np.sqrt(volume_ratio)
        
        # Total slippage (always adverse to order direction)
        total_slippage = spread_slippage + impact_slippage
        
        return total_slippage
    
    def compute_transaction_costs(self, order: Order) -> float:
        """
        Compute transaction costs.
        
        Returns:
            Total transaction cost in dollars
        """
        commission = order.quantity * self.commission_per_contract
        exchange_fees = order.quantity * self.exchange_fee_per_contract
        
        total_cost = commission + exchange_fees
        
        return total_cost
    
    def get_execution_price(self,
                           order: Order,
                           mid_price: float,
                           average_volume: float) -> float:
        """
        Get expected execution price including slippage.
        
        Args:
            order: Order object
            mid_price: Mid-market price
            average_volume: Average daily volume
            
        Returns:
            Expected execution price
        """
        slippage = self.compute_slippage(order, mid_price, average_volume)
        
        # Apply slippage in adverse direction
        if order.side == OrderSide.BUY:
            execution_price = mid_price + slippage
        else:  # SELL
            execution_price = mid_price - slippage
        
        return execution_price
    
    def simulate_fill(self,
                     order: Order,
                     mid_price: float,
                     average_volume: float) -> Dict[str, float]:
        """
        Simulate order fill with slippage and costs.
        
        Returns:
            Dict with execution details
        """
        execution_price = self.get_execution_price(order, mid_price, average_volume)
        transaction_costs = self.compute_transaction_costs(order)
        slippage = self.compute_slippage(order, mid_price, average_volume)
        
        # Total cost (slippage + commissions)
        slippage_cost = slippage * order.quantity
        total_cost = slippage_cost + transaction_costs
        
        # Cost in basis points of notional
        notional = mid_price * order.quantity
        total_cost_bps = (total_cost / notional) * 10000 if notional > 0 else 0
        
        return {
            'execution_price': execution_price,
            'slippage': slippage,
            'slippage_cost': slippage_cost,
            'transaction_costs': transaction_costs,
            'total_cost': total_cost,
            'total_cost_bps': total_cost_bps
        }


class AdaptiveSlippageModel(SlippageModel):
    """
    Adaptive slippage model that adjusts based on market conditions.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volatility_multiplier = 1.0
        self.liquidity_multiplier = 1.0
        
    def update_market_conditions(self, volatility_percentile: float, liquidity_score: float):
        """
        Update model parameters based on market conditions.
        
        Args:
            volatility_percentile: Current volatility percentile (0-100)
            liquidity_score: Liquidity score (0-1, higher = more liquid)
        """
        # Increase slippage in high volatility
        self.volatility_multiplier = 1.0 + (volatility_percentile / 100) * 0.5
        
        # Increase slippage in low liquidity
        self.liquidity_multiplier = 1.0 + (1 - liquidity_score) * 0.5
        
        logger.debug(f"Updated slippage model: vol_mult={self.volatility_multiplier:.2f}, "
                    f"liq_mult={self.liquidity_multiplier:.2f}")
    
    def compute_slippage(self, order: Order, mid_price: float, average_volume: float) -> float:
        """Compute slippage with adaptive adjustments."""
        base_slippage = super().compute_slippage(order, mid_price, average_volume)
        
        # Apply multipliers
        adjusted_slippage = base_slippage * self.volatility_multiplier * self.liquidity_multiplier
        
        return adjusted_slippage