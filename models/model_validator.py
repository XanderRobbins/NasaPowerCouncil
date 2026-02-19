"""
Model validation and performance tracking.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from loguru import logger


class ModelValidator:
    """
    Validate model performance out-of-sample.
    """
    
    def __init__(self):
        self.validation_history = []
        
    def validate_predictions(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            commodity: str) -> Dict:
        """
        Validate model predictions.
        
        Returns:
            Dict with validation metrics
        """
        # Remove NaNs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            logger.warning(f"No valid predictions for {commodity}")
            return {}
        
        # Regression metrics
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Directional accuracy (hit rate)
        hit_rate = np.mean(np.sign(y_true) == np.sign(y_pred))
        
        # Information coefficient (correlation)
        ic = np.corrcoef(y_true, y_pred)[0, 1]
        
        metrics = {
            'commodity': commodity,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'hit_rate': hit_rate,
            'ic': ic,
            'n_samples': len(y_true)
        }
        
        self.validation_history.append(metrics)
        
        logger.info(f"Validation for {commodity}: R²={r2:.4f}, Hit Rate={hit_rate:.2%}, IC={ic:.4f}")
        
        return metrics
    
    def track_rolling_performance(self,
                                  y_true: pd.Series,
                                  y_pred: pd.Series,
                                  window: int = 60) -> pd.DataFrame:
        """
        Track rolling performance metrics.
        
        Returns:
            DataFrame with rolling metrics
        """
        results = []
        
        for i in range(window, len(y_true)):
            y_true_window = y_true.iloc[i-window:i]
            y_pred_window = y_pred.iloc[i-window:i]
            
            # Remove NaNs
            mask = ~(y_true_window.isna() | y_pred_window.isna())
            y_true_clean = y_true_window[mask].values
            y_pred_clean = y_pred_window[mask].values
            
            if len(y_true_clean) < window // 2:
                continue
            
            r2 = r2_score(y_true_clean, y_pred_clean)
            hit_rate = np.mean(np.sign(y_true_clean) == np.sign(y_pred_clean))
            
            results.append({
                'date': y_true.index[i],
                'r2': r2,
                'hit_rate': hit_rate
            })
        
        return pd.DataFrame(results)
    
    def detect_performance_degradation(self,
                                      recent_metrics: Dict,
                                      historical_avg: Dict,
                                      threshold: float = 0.3) -> bool:
        """
        Detect if model performance has degraded significantly.
        
        Args:
            recent_metrics: Recent performance metrics
            historical_avg: Historical average metrics
            threshold: Degradation threshold (30% by default)
            
        Returns:
            True if degradation detected
        """
        # Compare R² and hit rate
        r2_degradation = (historical_avg.get('r2', 0) - recent_metrics.get('r2', 0)) / (historical_avg.get('r2', 1) + 1e-8)
        hit_rate_degradation = (historical_avg.get('hit_rate', 0) - recent_metrics.get('hit_rate', 0)) / (historical_avg.get('hit_rate', 1) + 1e-8)
        
        if r2_degradation > threshold or hit_rate_degradation > threshold:
            logger.warning(f"Performance degradation detected: R² down {r2_degradation:.2%}, Hit rate down {hit_rate_degradation:.2%}")
            return True
        
        return False
    
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of all validation results."""
        if not self.validation_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.validation_history)