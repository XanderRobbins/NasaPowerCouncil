"""
Trade execution module.
"""

from execution.order_manager import (
    OrderManager,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
)

from execution.slippage_model import (
    SlippageModel,
    AdaptiveSlippageModel,
)

from execution.executor import TradeExecutor

__all__ = [
    'OrderManager',
    'Order',
    'OrderSide',
    'OrderType',
    'OrderStatus',
    'SlippageModel',
    'AdaptiveSlippageModel',
    'TradeExecutor',
]