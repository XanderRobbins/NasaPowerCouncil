"""
Risk Management: Stop losses, drawdown controls, position limits.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from loguru import logger

from config.settings import HARD_STOP_MULTIPLIER, MAX_DRAWDOWN_THRESHOLD


class RiskManager:
    """
    Manage portfolio risk.
    
    Controls:
    - Hard stops (price-based)
    - Soft stops (signal reversal)
    - Portfolio drawdown limits
    - Position concentration limits
    - Leverage limits
    """
    
    def __init__(self,
                 hard_stop_mult: float = HARD_STOP_MULTIPLIER,
                 max_drawdown: float = MAX_DRAWDOWN_THRESHOLD,
                 max_leverage: float = 1.0):
        self.hard_stop_mult = hard_stop_mult
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        
    def check_position_stop(self, 
                           position: float,
                           entry_price: float,
                           current_price: float,
                           expected_vol: float) -> Tuple[bool, str]:
        """
        Check if position should be stopped out.
        
        Hard stop: Loss > hard_stop_mult × expected_vol
        
        Args:
            position: Position size (positive = long, negative = short)
            entry_price: Entry price
            current_price: Current price
            expected_vol: Expected N-day volatility
            
        Returns:
            (should_stop, reason)
        """
        # Compute PnL
        if position > 0:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        # Check hard stop
        stop_threshold = -self.hard_stop_mult * expected_vol
        
        if pnl_pct < stop_threshold:
            return True, f"Hard stop hit: {pnl_pct:.2%} < {stop_threshold:.2%}"
        
        return False, "OK"
    
    def check_signal_reversal(self,
                             current_signal: float,
                             current_position: float,
                             threshold: float = 0.5) -> Tuple[bool, str]:
        """
        Check if signal has reversed (soft stop).
        
        Args:
            current_signal: Current signal value
            current_position: Current position
            threshold: Threshold for reversal
            
        Returns:
            (should_stop, reason)
        """
        # Check if sign has flipped significantly
        if current_position > 0 and current_signal < -threshold:
            return True, f"Signal reversed: {current_signal:.2f} (was long)"
        elif current_position < 0 and current_signal > threshold:
            return True, f"Signal reversed: {current_signal:.2f} (was short)"
        
        # Check if signal has weakened significantly
        if abs(current_signal) < abs(current_position) * 0.3:
            return True, f"Signal weakened: {current_signal:.2f} vs position {current_position:.2f}"
        
        return False, "OK"
    
    def check_portfolio_drawdown(self, 
                                returns: pd.Series) -> Tuple[bool, float, str]:
        """
        Check portfolio drawdown.
        
        Args:
            returns: Portfolio return series
            
        Returns:
            (should_reduce, reduction_factor, reason)
        """
        if len(returns) < 2:
            return False, 1.0, "Insufficient history"
        
        # Compute cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Compute drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        current_drawdown = drawdown.iloc[-1]
        
        if current_drawdown < -self.max_drawdown:
            # Reduce exposure by 50%
            reduction = 0.5
            return True, reduction, f"Max drawdown hit: {current_drawdown:.2%}"
        elif current_drawdown < -self.max_drawdown * 0.7:
            # Reduce exposure by 25%
            reduction = 0.75
            return True, reduction, f"Approaching max drawdown: {current_drawdown:.2%}"
        
        return False, 1.0, "OK"
    
    def check_leverage(self, 
                      positions: Dict[str, float],
                      account_value: float) -> Tuple[bool, str]:
        """
        Check if leverage is within limits.
        
        Args:
            positions: Dict mapping commodity -> notional position value
            account_value: Total account value
            
        Returns:
            (is_ok, reason)
        """
        total_notional = sum(abs(p) for p in positions.values())
        leverage = total_notional / account_value if account_value > 0 else 0
        
        if leverage > self.max_leverage:
            return False, f"Leverage too high: {leverage:.2f}x > {self.max_leverage}x"
        
        return True, "OK"
    
    def apply_risk_limits(self,
                         positions: Dict[str, float],
                         signals: Dict[str, float],
                         entry_prices: Dict[str, float],
                         current_prices: Dict[str, float],
                         expected_vols: Dict[str, float],
                         portfolio_returns: pd.Series) -> Dict[str, float]:
        """
        Apply all risk limits and adjust positions.
        
        Returns:
            Adjusted positions
        """
        adjusted_positions = positions.copy()
        
        # Check portfolio drawdown
        should_reduce, reduction_factor, dd_reason = self.check_portfolio_drawdown(portfolio_returns)
        if should_reduce:
            logger.warning(f"Reducing all positions by {(1-reduction_factor)*100:.0f}%: {dd_reason}")
            for commodity in adjusted_positions:
                adjusted_positions[commodity] *= reduction_factor
        
        # Check individual position stops
        positions_to_close = []
        
        for commodity, position in adjusted_positions.items():
            if abs(position) < 1e-6:
                continue
            
            # Hard stop check
            should_stop, reason = self.check_position_stop(
                position,
                entry_prices.get(commodity, current_prices.get(commodity, 0)),
                current_prices.get(commodity, 0),
                expected_vols.get(commodity, 0.15)
            )
            
            if should_stop:
                logger.warning(f"Closing {commodity} position: {reason}")
                positions_to_close.append(commodity)
                continue
            
            # Signal reversal check
            should_stop, reason = self.check_signal_reversal(
                signals.get(commodity, 0),
                position
            )
            
            if should_stop:
                logger.warning(f"Closing {commodity} position: {reason}")
                positions_to_close.append(commodity)
        
        # Close flagged positions
        for commodity in positions_to_close:
            adjusted_positions[commodity] = 0.0
        
        return adjusted_positions
    
    def get_position_sizing_multiplier(self, 
                                      recent_sharpe: float,
                                      target_sharpe: float = 1.0) -> float:
        """
        Adjust position sizing based on recent performance.
        
        If strategy is performing well, increase size.
        If struggling, decrease size.
        
        Args:
            recent_sharpe: Recent Sharpe ratio (e.g., 60-day)
            target_sharpe: Target Sharpe ratio
            
        Returns:
            Multiplier (0.5 - 1.5)
        """
        if recent_sharpe > target_sharpe * 1.2:
            return 1.2  # Increase by 20%
        elif recent_sharpe < target_sharpe * 0.5:
            return 0.7  # Decrease by 30%
        else:
            return 1.0


def apply_risk_management(positions: Dict[str, float],
                         signals: Dict[str, float],
                         entry_prices: Dict[str, float],
                         current_prices: Dict[str, float],
                         expected_vols: Dict[str, float],
                         portfolio_returns: pd.Series) -> Dict[str, float]:
    """
    Convenience function to apply risk management.
    """
    manager = RiskManager()
    adjusted_positions = manager.apply_risk_limits(
        positions,
        signals,
        entry_prices,
        current_prices,
        expected_vols,
        portfolio_returns
    )
    
    return adjusted_positions