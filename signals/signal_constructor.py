"""
Convert model predictions into standardized, capped signals.
"""
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger

from config.settings import SIGNAL_CAP


class SignalConstructor:
    """
    Transform raw predictions into tradeable signals.
    
    Steps:
    1. Standardize by realized volatility: S_t = r̂_{t+20} / σ_20d
    2. Cap at ±3 (or configured cap)
    3. Smooth via EMA (optional)
    """
    
    def __init__(self, 
                 cap: float = SIGNAL_CAP,
                 vol_window: int = 20):
        self.cap = cap
        self.vol_window = vol_window
        
    def compute_realized_vol(self, returns: pd.Series) -> pd.Series:
        """
        Compute rolling realized volatility.
        
        σ_t = std(returns_{t-20:t}) * sqrt(252)  (annualized)
        """
        rolling_std = returns.rolling(window=self.vol_window).std()
        annualized_vol = rolling_std * np.sqrt(252)
        
        # Fill initial NaNs with expanding window
        annualized_vol = annualized_vol.fillna(
            returns.expanding().std() * np.sqrt(252)
        )
        
        return annualized_vol
    
    def standardize_signal(self, 
                          predictions: np.ndarray, 
                          returns: pd.Series) -> np.ndarray:
        """
        Standardize predictions by realized volatility.
        
        S_t = r̂_t / σ_t
        """
        realized_vol = self.compute_realized_vol(returns)
        
        # Align lengths
        if len(predictions) < len(realized_vol):
            realized_vol = realized_vol.iloc[-len(predictions):]
        elif len(predictions) > len(realized_vol):
            predictions = predictions[-len(realized_vol):]
        
        # Standardize
        with np.errstate(divide='ignore', invalid='ignore'):
            standardized = predictions / realized_vol.values
        
        # Handle any infinities or NaNs
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=self.cap, neginf=-self.cap)
        
        return standardized
    
    def cap_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Cap signal at ±cap.
        """
        return np.clip(signal, -self.cap, self.cap)
    
    def construct_signal(self, 
                        predictions: np.ndarray, 
                        returns: pd.Series) -> np.ndarray:
        """
        Full signal construction pipeline.
        
        Args:
            predictions: Raw model predictions (forward returns)
            returns: Historical returns series (for vol calculation)
            
        Returns:
            Standardized, capped signals
        """
        # Standardize
        signal = self.standardize_signal(predictions, returns)
        
        # Cap
        signal = self.cap_signal(signal)
        
        logger.info(f"Constructed signal: mean={signal.mean():.4f}, std={signal.std():.4f}")
        
        return signal
    
    def detect_regime_change(self, signal: np.ndarray, window: int = 60) -> np.ndarray:
        """
        Detect regime changes in signal (optional).
        
        Returns:
            Array of regime indicators (1 = regime change detected)
        """
        signal_series = pd.Series(signal)
        
        # Compute rolling mean and std
        rolling_mean = signal_series.rolling(window=window).mean()
        rolling_std = signal_series.rolling(window=window).std()
        
        # Detect when signal exceeds 2 std from rolling mean
        regime_change = np.abs(signal_series - rolling_mean) > (2 * rolling_std)
        
        return regime_change.astype(int).values


def construct_signals_for_commodity(predictions: np.ndarray,
                                    returns: pd.Series,
                                    commodity: str) -> np.ndarray:
    """
    Convenience function to construct signals for a commodity.
    """
    logger.info(f"Constructing signals for {commodity}")
    
    constructor = SignalConstructor()
    signals = constructor.construct_signal(predictions, returns)
    
    return signals