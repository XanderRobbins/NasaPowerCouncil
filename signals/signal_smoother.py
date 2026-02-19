"""
Smooth signals to avoid flickering and overtrading.
"""
import pandas as pd
import numpy as np
from loguru import logger

from config.settings import SIGNAL_EMA_SPAN


class SignalSmoother:
    """
    Apply exponential moving average (EMA) to signals.
    
    This reduces noise and creates persistence, which:
    1. Reduces transaction costs
    2. Aligns with the slow-moving thesis
    3. Prevents whipsaws
    """
    
    def __init__(self, ema_span: int = SIGNAL_EMA_SPAN):
        self.ema_span = ema_span
        
    def smooth_ema(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average.
        
        EMA_t = α × Signal_t + (1-α) × EMA_{t-1}
        where α = 2 / (span + 1)
        """
        signal_series = pd.Series(signal)
        smoothed = signal_series.ewm(span=self.ema_span, adjust=False).mean()
        
        return smoothed.values
    
    def smooth_sma(self, signal: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Apply simple moving average (alternative).
        """
        signal_series = pd.Series(signal)
        smoothed = signal_series.rolling(window=window, min_periods=1).mean()
        
        return smoothed.values
    
    def smooth_double_ema(self, signal: np.ndarray, span1: int = 3, span2: int = 7) -> np.ndarray:
        """
        Apply double EMA (faster and slower).
        """
        signal_series = pd.Series(signal)
        
        # Fast EMA
        ema1 = signal_series.ewm(span=span1, adjust=False).mean()
        
        # Slow EMA
        ema2 = signal_series.ewm(span=span2, adjust=False).mean()
        
        # Weighted average
        smoothed = 0.7 * ema1 + 0.3 * ema2
        
        return smoothed.values
    
    def apply_smoother(self, signal: np.ndarray, method: str = 'ema') -> np.ndarray:
        """
        Apply selected smoothing method.
        
        Args:
            signal: Raw signal
            method: 'ema', 'sma', or 'double_ema'
            
        Returns:
            Smoothed signal
        """
        if method == 'ema':
            return self.smooth_ema(signal)
        elif method == 'sma':
            return self.smooth_sma(signal)
        elif method == 'double_ema':
            return self.smooth_double_ema(signal)
        else:
            logger.warning(f"Unknown smoothing method: {method}, returning raw signal")
            return signal


def smooth_signal(signal: np.ndarray, method: str = 'ema') -> np.ndarray:
    """
    Convenience function to smooth a signal.
    """
    smoother = SignalSmoother()
    smoothed = smoother.apply_smoother(signal, method=method)
    
    logger.debug(f"Smoothed signal using {method}")
    
    return smoothed