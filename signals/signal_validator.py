"""
Validate signal quality and stability.
"""
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class SignalValidator:
    """
    Check signal quality metrics.
    
    Validates:
    - No excessive NaNs
    - No extreme outliers
    - Reasonable turnover
    - Autocorrelation (persistence check)
    """
    
    def __init__(self):
        pass
    
    def check_nan_rate(self, signal: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Check if NaN rate is acceptable.
        
        Returns:
            True if NaN rate < threshold
        """
        nan_rate = np.isnan(signal).sum() / len(signal)
        
        if nan_rate > threshold:
            logger.warning(f"High NaN rate in signal: {nan_rate:.2%}")
            return False
        
        return True
    
    def check_outliers(self, signal: np.ndarray, threshold: float = 5.0) -> bool:
        """
        Check for extreme outliers (beyond ±threshold std).
        
        Returns:
            True if no extreme outliers
        """
        signal_clean = signal[~np.isnan(signal)]
        
        if len(signal_clean) == 0:
            return False
        
        z_scores = np.abs(stats.zscore(signal_clean))
        outlier_rate = (z_scores > threshold).sum() / len(signal_clean)
        
        if outlier_rate > 0.05:  # More than 5% outliers
            logger.warning(f"High outlier rate in signal: {outlier_rate:.2%}")
            return False
        
        return True
    
    def check_turnover(self, signal: np.ndarray, threshold: float = 2.0) -> bool:
        """
        Check signal turnover (daily changes).
        
        Turnover = Σ|Signal_t - Signal_{t-1}|
        
        High turnover = unstable signal = high transaction costs
        
        Returns:
            True if average daily change < threshold
        """
        signal_series = pd.Series(signal)
        daily_changes = signal_series.diff().abs()
        avg_change = daily_changes.mean()
        
        if avg_change > threshold:
            logger.warning(f"High signal turnover: {avg_change:.4f}")
            return False
        
        return True
    
    def check_autocorrelation(self, signal: np.ndarray, lag: int = 1, min_corr: float = 0.5) -> bool:
        """
        Check signal persistence (autocorrelation).
        
        Slow-moving signals should be autocorrelated.
        
        Returns:
            True if autocorrelation > min_corr
        """
        signal_clean = signal[~np.isnan(signal)]
        
        if len(signal_clean) < lag + 1:
            return False
        
        autocorr = pd.Series(signal_clean).autocorr(lag=lag)
        
        if autocorr < min_corr:
            logger.warning(f"Low signal persistence: autocorr({lag})={autocorr:.4f}")
            return False
        
        return True
    
    def check_signal_sign_changes(self, signal: np.ndarray, max_changes: int = 100) -> bool:
        """
        Check frequency of sign changes (long → short → long).
        
        Too many sign changes = noisy signal.
        
        Returns:
            True if sign changes < max_changes
        """
        signal_series = pd.Series(signal)
        sign_changes = (np.sign(signal_series) != np.sign(signal_series.shift())).sum()
        
        if sign_changes > max_changes:
            logger.warning(f"Excessive sign changes in signal: {sign_changes}")
            return False
        
        return True
    
    def validate_signal(self, signal: np.ndarray) -> dict:
        """
        Run all validation checks.
        
        Returns:
            Dict with validation results
        """
        results = {
            'nan_rate_ok': self.check_nan_rate(signal),
            'outliers_ok': self.check_outliers(signal),
            'turnover_ok': self.check_turnover(signal),
            'autocorr_ok': self.check_autocorrelation(signal),
            'sign_changes_ok': self.check_signal_sign_changes(signal),
        }
        
        results['all_passed'] = all(results.values())
        
        if results['all_passed']:
            logger.info("✓ Signal passed all validation checks")
        else:
            failed = [k for k, v in results.items() if not v and k != 'all_passed']
            logger.warning(f"✗ Signal failed validation: {failed}")
        
        return results


def validate_signal(signal: np.ndarray) -> bool:
    """
    Convenience function to validate a signal.
    
    Returns:
        True if signal passes all checks
    """
    validator = SignalValidator()
    results = validator.validate_signal(signal)
    
    return results['all_passed']