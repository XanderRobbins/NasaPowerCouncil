"""
Contract Selection: Choose which futures contract to trade based on stress timing.
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from config.contracts import get_contract_spec, get_active_contract_month
from config.calendars import get_current_stage


class ContractSelector:
    """
    Select optimal futures contract based on:
    - Stress timing (current season vs storage)
    - Liquidity
    - Term structure
    - Roll schedule
    """
    
    def __init__(self, commodity: str):
        self.commodity = commodity
        self.spec = get_contract_spec(commodity)
        
    def select_contract(self, 
                       stress_timing: str, 
                       current_date: datetime,
                       market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Select which contract to trade.
        
        Args:
            stress_timing: 'current_season', 'storage', or 'immediate'
            current_date: Current date
            market_data: Optional dict with volume, open interest, term structure
            
        Returns:
            Contract symbol (e.g., 'ZCZ24' for Dec 2024 corn)
        """
        # Get base contract month
        contract_month = get_active_contract_month(
            self.commodity, 
            stress_timing, 
            current_date
        )
        
        # If market data available, check liquidity
        if market_data is not None:
            # Check if contract has sufficient liquidity
            volume = market_data.get('volume', {}).get(contract_month, 0)
            if volume < self.spec['typical_volume_threshold']:
                logger.warning(f"Low liquidity for {contract_month}: {volume} contracts")
                # Fall back to front month
                contract_month = self._get_front_month(current_date)
        
        logger.info(f"Selected contract: {contract_month} (stress timing: {stress_timing})")
        
        return contract_month
    
    def _get_front_month(self, current_date: datetime) -> str:
        """Get the front (most liquid) contract."""
        return get_active_contract_month(self.commodity, 'immediate', current_date)
    
    def should_roll(self, 
                   current_contract: str, 
                   current_date: datetime,
                   days_to_expiry: int = 5) -> bool:
        """
        Determine if we should roll to next contract.
        
        Args:
            current_contract: Current contract symbol
            current_date: Current date
            days_to_expiry: Roll this many days before expiry
            
        Returns:
            True if should roll
        """
        # Extract expiry from contract (simplified - in production, use actual expiry dates)
        # For now, assume need to roll based on calendar
        
        # Rule: Roll 5 days before first notice day (for physical delivery contracts)
        # This is simplified - real implementation needs actual contract specs
        
        return False  # Placeholder
    
    def get_calendar_spread(self, 
                           front_month: str, 
                           back_month: str,
                           ratio: float = 1.0) -> Dict[str, float]:
        """
        Create calendar spread position.
        
        Example: Long Dec corn, Short Mar corn
        
        Args:
            front_month: Front contract
            back_month: Back contract
            ratio: Ratio of contracts (default 1:1)
            
        Returns:
            Dict with positions for each leg
        """
        return {
            front_month: ratio,
            back_month: -ratio
        }
    
    def express_as_spread(self, 
                         signal: float,
                         current_date: datetime,
                         term_structure_slope: float) -> Dict[str, float]:
        """
        Express position as calendar spread if appropriate.
        
        If curve is steep, express via spread to capture roll yield.
        
        Args:
            signal: Raw signal
            current_date: Current date
            term_structure_slope: Slope of term structure (annualized)
            
        Returns:
            Dict with contract positions
        """
        # Threshold: if curve > 10% annualized, use spread
        if abs(term_structure_slope) > 0.10:
            front = self._get_front_month(current_date)
            back = get_active_contract_month(self.commodity, 'storage', current_date)
            
            # If signal is long and curve is in contango (upward sloping)
            if signal > 0 and term_structure_slope > 0:
                # Long front, short back (harvest the roll yield)
                return self.get_calendar_spread(front, back, ratio=abs(signal))
            elif signal < 0 and term_structure_slope < 0:
                # Short front, long back
                return self.get_calendar_spread(front, back, ratio=-abs(signal))
        
        # Default: outright position
        contract = self.select_contract('current_season', current_date)
        return {contract: signal}


def select_contracts_for_portfolio(positions: Dict[str, float],
                                   current_date: datetime,
                                   stress_timings: Dict[str, str],
                                   market_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Dict[str, float]]:
    """
    Select contracts for entire portfolio.
    
    Args:
        positions: Dict mapping commodity -> position size
        current_date: Current date
        stress_timings: Dict mapping commodity -> stress timing
        market_data: Optional market data per commodity
        
    Returns:
        Dict mapping commodity -> {contract: position}
    """
    contract_positions = {}
    
    for commodity, position in positions.items():
        if abs(position) < 1e-6:
            continue
        
        selector = ContractSelector(commodity)
        stress_timing = stress_timings.get(commodity, 'current_season')
        
        # Get market data for this commodity
        commodity_market_data = market_data.get(commodity) if market_data else None
        
        # Select contract
        contract = selector.select_contract(
            stress_timing, 
            current_date,
            commodity_market_data
        )
        
        contract_positions[commodity] = {contract: position}
    
    logger.info(f"Selected contracts for {len(contract_positions)} commodities")
    
    return contract_positions