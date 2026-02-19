"""
Climate Futures Trading System

A systematic trading system that exploits persistent weather anomalies 
in commodity futures markets.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Expose main components for easy imports
from config.settings import (
    PHASE_1_COMMODITIES,
    TARGET_PORTFOLIO_VOL,
    MAX_SINGLE_COMMODITY,
    MAX_DRAWDOWN_THRESHOLD
)

from backtest.engine import run_backtest
from live.live_engine import start_live_trading

__all__ = [
    'run_backtest',
    'start_live_trading',
    'PHASE_1_COMMODITIES',
    'TARGET_PORTFOLIO_VOL',
    'MAX_SINGLE_COMMODITY',
    'MAX_DRAWDOWN_THRESHOLD',
]