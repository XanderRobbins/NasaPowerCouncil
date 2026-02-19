"""
Feature Drift Agent: Monitors distribution shifts in features.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

from council.base_agent import BaseAgent


class FeatureDriftAgent(BaseAgent):
    """
    Detects feature distribution drift.
    
    If features are drifting significantly from their historical distributions,
    the model may not generalize well.
    """
    
    def __init__(self, lookback: int = 252):
        super().__init__("FeatureDriftAgent")
        self.lookback = lookback
        self.score = 1.0
        self.drift_metrics = {}
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect feature drift using statistical tests.
        """
        features = context.get('features')
        
        if features is None:
            self.score = 0.5
            return {'score': self.score}
        
        # Get recent and historical data
        if len(features) < self.lookback * 2:
            self.score = 0.8
            return {'score': self.score, 'note': 'Insufficient history'}
        
        recent = features.iloc[-self.lookback:]
        historical = features.iloc[:-self.lookback]
        
        drift_scores = []
        
        # Check numerical features
        for col in features.select_dtypes(include=[np.number]).columns:
            if col == 'date':
                continue
            
            recent_vals = recent[col].dropna()
            hist_vals = historical[col].dropna()
            
            if len(recent_vals) < 30 or len(hist_vals) < 30:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(recent_vals, hist_vals)
            
            # Mean shift
            mean_shift = abs(recent_vals.mean() - hist_vals.mean()) / (hist_vals.std() + 1e-8)
            
            # Variance shift
            var_ratio = recent_vals.var() / (hist_vals.var() + 1e-8)
            
            self.drift_metrics[col] = {
                'ks_stat': ks_stat,
                'p_value': p_value,
                'mean_shift': mean_shift,
                'var_ratio': var_ratio
            }
            
            # Penalize drift
            if p_value < 0.05:  # Significant drift
                drift_scores.append(1 - ks_stat)
            else:
                drift_scores.append(1.0)
        
        # Aggregate score
        if drift_scores:
            self.score = np.mean(drift_scores)
        else:
            self.score = 0.8
        
        return {
            'score': self.score,
            'drift_metrics': self.drift_metrics,
            'n_features_checked': len(drift_scores)
        }
    
    def get_score(self) -> float:
        return self.score