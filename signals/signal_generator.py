"""
Convert model predictions into position sizes.
Simple vol-standardization with capping.
"""
import numpy as np
import pandas as pd
from loguru import logger

class SignalGenerator:
    """
    Transform model predictions into capped position signals.
    
    Steps:
    1. Standardize by realized vol: signal = prediction / vol
    2. Cap at ±3 (prevents blow-ups)
    3. Done. No smoothing.
    """
    
    def __init__(self, cap: float = 3.0, vol_window: int = 20):
        self.cap = cap
        self.vol_window = vol_window
    
    def _realized_vol(self, returns: pd.Series) -> pd.Series:
        """Rolling annualized volatility."""
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        # Handle NaNs at start with expanding window
        return vol.fillna(returns.expanding().std() * np.sqrt(252))
    
    def generate(self, predictions: np.ndarray, returns: pd.Series) -> np.ndarray:
        """
        Convert predictions to signals.
        
        Args:
            predictions: Raw model predictions (forward returns)
            returns: Historical returns (for vol calculation)
            
        Returns:
            Capped signals in [-cap, +cap]
        """
        # Compute volatility
        vol = self._realized_vol(returns)
        
        # Align lengths
        if len(predictions) != len(vol):
            min_len = min(len(predictions), len(vol))
            predictions = predictions[-min_len:]
            vol = vol.iloc[-min_len:]
        
        # Standardize by vol
        signals = predictions / vol.values
        
        # Handle edge cases
        signals = np.nan_to_num(signals, nan=0.0, posinf=self.cap, neginf=-self.cap)
        
        # Cap
        signals = np.clip(signals, -self.cap, self.cap)
        
        logger.info(f"Generated {len(signals)} signals | "
                   f"Mean: {signals.mean():.4f} | Std: {signals.std():.4f}")
        
        return signals


def generate_signal(predictions: np.ndarray, returns: pd.Series, cap: float = 3.0) -> np.ndarray:
    """Convenience function."""
    generator = SignalGenerator(cap=cap)
    return generator.generate(predictions, returns)