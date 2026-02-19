"""
Portfolio construction: position sizing and risk allocation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from loguru import logger

from config.settings import TARGET_PORTFOLIO_VOL, MAX_SINGLE_COMMODITY


class PortfolioConstructor:
    """
    Construct portfolio positions from signals.
    
    Approach:
    1. Vol-normalize each position
    2. Apply risk limits
    3. Scale to target portfolio vol
    4. Equal risk contribution (optional)
    """
    
    def __init__(self, 
                 target_vol: float = TARGET_PORTFOLIO_VOL,
                 max_single_commodity: float = MAX_SINGLE_COMMODITY):
        self.target_vol = target_vol
        self.max_single = max_single_commodity
        
    def compute_position_sizes(self, 
                              signals: Dict[str, float], 
                              vols: Dict[str, float]) -> Dict[str, float]:
        """
        Compute position sizes for each commodity.
        
        Position_i = Signal_i / Vol_i
        
        Then scale to target portfolio vol.
        
        Args:
            signals: Dict mapping commodity -> signal
            vols: Dict mapping commodity -> realized vol
            
        Returns:
            Dict mapping commodity -> position size
        """
        positions = {}
        
        # Step 1: Vol-normalize
        for commodity, signal in signals.items():
            vol = vols.get(commodity, 0.15)  # Default 15% vol
            
            if vol < 0.01:
                vol = 0.01  # Floor to avoid division by zero
            
            position = signal / vol
            positions[commodity] = position
        
        # Step 2: Cap single commodity exposure
        for commodity in positions:
            positions[commodity] = np.clip(
                positions[commodity], 
                -self.max_single, 
                self.max_single
            )
        
        # Step 3: Compute portfolio vol
        position_array = np.array(list(positions.values()))
        vol_array = np.array([vols.get(c, 0.15) for c in positions.keys()])
        
        # Simplified: assume uncorrelated (conservative)
        portfolio_vol = np.sqrt(np.sum((position_array * vol_array) ** 2))
        
        if portfolio_vol < 1e-6:
            logger.warning("Portfolio vol near zero, returning zero positions")
            return {k: 0.0 for k in positions.keys()}
        
        # Step 4: Scale to target
        scale_factor = self.target_vol / portfolio_vol
        
        for commodity in positions:
            positions[commodity] *= scale_factor
        
        logger.info(f"Portfolio constructed: {len(positions)} commodities")
        logger.info(f"Target vol: {self.target_vol:.2%}, Achieved vol: {portfolio_vol * scale_factor:.2%}")
        
        return positions
    
    def apply_equal_risk_contribution(self, 
                                     positions: Dict[str, float],
                                     vols: Dict[str, float]) -> Dict[str, float]:
        """
        Adjust positions for equal risk contribution.
        
        Each commodity contributes equally to portfolio risk.
        """
        # Compute risk contributions
        risk_contributions = {}
        total_risk = 0.0
        
        for commodity, position in positions.items():
            vol = vols.get(commodity, 0.15)
            risk = abs(position) * vol
            risk_contributions[commodity] = risk
            total_risk += risk
        
        if total_risk < 1e-6:
            return positions
        
        # Target: each commodity contributes 1/N of risk
        n_commodities = len(positions)
        target_risk_per_commodity = total_risk / n_commodities
        
        # Adjust positions
        adjusted_positions = {}
        for commodity, position in positions.items():
            current_risk = risk_contributions[commodity]
            if current_risk > 1e-6:
                adjustment = target_risk_per_commodity / current_risk
                adjusted_positions[commodity] = position * adjustment
            else:
                adjusted_positions[commodity] = 0.0
        
        return adjusted_positions
    
    def diversification_score(self, positions: Dict[str, float]) -> float:
        """
        Compute diversification score (1 = fully diversified, 0 = single asset).
        
        Uses Herfindahl index.
        """
        weights = np.array([abs(p) for p in positions.values()])
        total_weight = weights.sum()
        
        if total_weight < 1e-6:
            return 0.0
        
        weights_norm = weights / total_weight
        herfindahl = np.sum(weights_norm ** 2)
        
        # Normalize: 1/N (fully diversified) to 1 (single asset)
        n = len(positions)
        diversification = (1 - herfindahl) / (1 - 1/n) if n > 1 else 0.0
        
        return diversification


def construct_portfolio(signals: Dict[str, float], 
                       vols: Dict[str, float]) -> Dict[str, float]:
    """
    Convenience function to construct portfolio.
    
    Args:
        signals: Dict mapping commodity -> signal
        vols: Dict mapping commodity -> realized vol
        
    Returns:
        Dict mapping commodity -> position size
    """
    constructor = PortfolioConstructor()
    positions = constructor.compute_position_sizes(signals, vols)
    
    # Apply equal risk contribution
    positions = constructor.apply_equal_risk_contribution(positions, vols)
    
    # Log diversification
    div_score = constructor.diversification_score(positions)
    logger.info(f"Portfolio diversification score: {div_score:.2f}")
    
    return positions