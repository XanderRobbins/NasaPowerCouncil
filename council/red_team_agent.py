"""
Red Team Agent: Actively looks for reasons the signal might fail.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from datetime import datetime, timedelta

from council.base_agent import BaseAgent


class RedTeamAgent(BaseAgent):
    """
    Adversarial agent that finds reasons to NOT trade.
    
    Checks:
    - USDA report proximity (data releases can overwhelm weather signals)
    - COT extreme positioning (crowding)
    - Seasonal patterns (is this just seasonal?)
    - Cross-commodity substitution (e.g., soy meal vs corn for feed)
    - Geopolitical events
    """
    
    def __init__(self):
        super().__init__("RedTeamAgent")
        self.penalty = 0.0
        self.red_flags = []
        
        # USDA report schedule (approximate, major reports)
        self.usda_report_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Monthly
        self.usda_report_day = 12  # Typically around 12th of month
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find reasons the signal might fail.
        """
        self.red_flags = []
        self.penalty = 0.0
        
        commodity = context.get('commodity')
        current_date = context.get('current_date', datetime.now())
        
        # 1. Check USDA report proximity
        if self._usda_report_upcoming(commodity, current_date, days=7):
            self.red_flags.append("USDA report within 7 days")
            self.penalty += 0.3
        
        # 2. Check COT positioning (if available)
        cot_data = context.get('cot_positioning')
        if cot_data is not None:
            if self._cot_extreme(cot_data):
                self.red_flags.append("Extreme COT positioning detected")
                self.penalty += 0.2
        
        # 3. Check signal age (stale signal)
        signal_date = context.get('signal_date')
        if signal_date is not None:
            days_old = (current_date - pd.to_datetime(signal_date)).days
            if days_old > 10:
                self.red_flags.append(f"Signal is {days_old} days old")
                self.penalty += 0.15
        
        # 4. Check for calendar effects
        if self._is_year_end(current_date):
            self.red_flags.append("Year-end calendar effect")
            self.penalty += 0.1
        
        # 5. Dollar strength (if provided)
        if context.get('dollar_surge', False):
            self.red_flags.append("Dollar strength surge")
            self.penalty += 0.1
        
        # 6. Cross-commodity check
        if commodity == 'corn' and context.get('soybean_signal_opposite', False):
            self.red_flags.append("Soybean signal contradicts corn signal")
            self.penalty += 0.1
        
        # 7. Liquidity check
        volume = context.get('average_volume')
        if volume is not None and volume < 5000:
            self.red_flags.append("Low liquidity detected")
            self.penalty += 0.2
        
        # Cap penalty at 0.6 (60% max reduction)
        self.penalty = min(self.penalty, 0.6)
        
        return {
            'penalty': self.penalty,
            'red_flags': self.red_flags,
            'n_flags': len(self.red_flags)
        }
    
    def _usda_report_upcoming(self, commodity: str, current_date: datetime, days: int = 7) -> bool:
        """
        Check if USDA report is upcoming.
        
        USDA releases WASDE (World Agricultural Supply and Demand Estimates) 
        typically around the 9th-12th of each month.
        """
        if commodity not in ['corn', 'soybeans', 'wheat', 'cotton', 'rice']:
            return False
        
        # Check if we're within 'days' of a report date
        report_day = self.usda_report_day
        
        days_until_report = report_day - current_date.day
        if 0 <= days_until_report <= days:
            return True
        
        # Check next month too
        next_month = (current_date.replace(day=1) + timedelta(days=32)).replace(day=report_day)
        days_until_next = (next_month - current_date).days
        if 0 <= days_until_next <= days:
            return True
        
        return False
    
    def _cot_extreme(self, cot_data: Dict[str, float]) -> bool:
        """
        Check if COT positioning is at extreme levels.
        
        Extreme positioning can lead to reversals.
        """
        net_position_pct = cot_data.get('net_position_percentile', 50)
        
        # Extreme if in top/bottom 10%
        if net_position_pct > 90 or net_position_pct < 10:
            return True
        
        return False
    
    def _is_year_end(self, current_date: datetime) -> bool:
        """Check if we're in year-end period (tax loss selling, window dressing)."""
        return current_date.month == 12 and current_date.day > 15
    
    def get_score(self) -> float:
        """Return inverse penalty as score."""
        return 1.0 - self.penalty
    
    def get_penalty(self) -> float:
        """Return raw penalty value."""
        return self.penalty