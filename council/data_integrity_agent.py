"""
Data Integrity Agent: Checks for data quality issues.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any

from council.base_agent import BaseAgent


class DataIntegrityAgent(BaseAgent):
    """
    Validates data quality.
    
    Checks:
    - Missing values
    - Outl anomalies
    - Data staleness
    - Extreme values
    """
    
    def __init__(self):
        super().__init__("DataIntegrityAgent")
        self.score = 1.0
        self.issues = []
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check data quality.
        
        Context should contain:
        - features: Feature DataFrame
        - prices: Price series
        - dates: Date index
        """
        self.issues = []
        
        features = context.get('features')
        prices = context.get('prices')
        
        if features is None or prices is None:
            self.score = 0.0
            self.issues.append("Missing required data")
            return {'issues': self.issues, 'score': self.score}
        
        # Check for missing values
        missing_rate = features.isnull().sum().sum() / (features.shape[0] * features.shape[1])
        if missing_rate > 0.1:
            self.issues.append(f"High missing rate: {missing_rate:.2%}")
            self.score *= 0.5
        
        # Check for temperature anomalies (physically impossible values)
        if 'temp_max' in features.columns:
            if (features['temp_max'] > 60).any():  # Celsius
                self.issues.append("Implausible temperature values detected")
                self.score *= 0.3
        
        # Check for negative precipitation
        if 'precipitation' in features.columns:
            if (features['precipitation'] < 0).any():
                self.issues.append("Negative precipitation detected")
                self.score *= 0.3
        
        # Check price data staleness
        if isinstance(prices, pd.Series) and hasattr(prices, 'index'):
            last_date = prices.index[-1]
            days_stale = (pd.Timestamp.now() - pd.to_datetime(last_date)).days
            if days_stale > 5:
                self.issues.append(f"Price data is {days_stale} days stale")
                self.score *= 0.7
        
        # Check for duplicate dates
        if 'date' in features.columns:
            duplicates = features['date'].duplicated().sum()
            if duplicates > 0:
                self.issues.append(f"{duplicates} duplicate dates found")
                self.score *= 0.6
        
        return {
            'issues': self.issues,
            'score': self.score,
            'missing_rate': missing_rate
        }
    
    def get_score(self) -> float:
        return self.score