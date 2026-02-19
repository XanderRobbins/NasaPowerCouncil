"""
Model Stability Agent: Tracks model performance degradation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from council.base_agent import BaseAgent


class ModelStabilityAgent(BaseAgent):
    """
    Monitors model performance over time.
    
    Tracks:
    - Rolling R²
    - Rolling hit rate (directional accuracy)
    - Prediction error trends
    - Drawdowns
    """
    
    def __init__(self, window: int = 60):
        super().__init__("ModelStabilityAgent")
        self.window = window
        self.score = 1.0
        self.performance_history = []
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model stability.
        """
        predictions = context.get('predictions')
        actuals = context.get('actuals')
        
        if predictions is None or actuals is None:
            self.score = 0.5
            return {'score': self.score}
        
        if len(predictions) < self.window:
            self.score = 0.8
            return {'score': self.score, 'note': 'Insufficient data'}
        
        # Recent performance
        recent_pred = predictions[-self.window:]
        recent_actual = actuals[-self.window:]
        
        # Historical performance
        if len(predictions) > self.window * 2:
            hist_pred = predictions[:-self.window]
            hist_actual = actuals[:-self.window]
        else:
            hist_pred = predictions
            hist_actual = actuals
        
        # Compute metrics
        recent_r2 = self._compute_r2(recent_actual, recent_pred)
        hist_r2 = self._compute_r2(hist_actual, hist_pred)
        
        recent_hit_rate = self._compute_hit_rate(recent_actual, recent_pred)
        hist_hit_rate = self._compute_hit_rate(hist_actual, hist_pred)
        
        # Check for degradation
        r2_degradation = hist_r2 - recent_r2
        hit_rate_degradation = hist_hit_rate - recent_hit_rate
        
        # Score based on degradation
        if r2_degradation > 0.1 or hit_rate_degradation > 0.1:
            # Significant degradation
            self.score = 0.5
        elif r2_degradation > 0.05 or hit_rate_degradation > 0.05:
            # Moderate degradation
            self.score = 0.7
        else:
            self.score = 1.0
        
        return {
            'score': self.score,
            'recent_r2': recent_r2,
            'hist_r2': hist_r2,
            'recent_hit_rate': recent_hit_rate,
            'hist_hit_rate': hist_hit_rate,
            'r2_degradation': r2_degradation,
            'hit_rate_degradation': hit_rate_degradation
        }
    
    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.0
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def _compute_hit_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute directional accuracy (hit rate)."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.5
        
        correct = np.sign(y_true) == np.sign(y_pred)
        return correct.sum() / len(correct)
    
    def get_score(self) -> float:
        return self.score