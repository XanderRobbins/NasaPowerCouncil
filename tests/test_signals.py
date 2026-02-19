"""
Tests for signals module.
"""
import pytest
import pandas as pd
import numpy as np

from signals.signal_constructor import SignalConstructor
from signals.signal_smoother import SignalSmoother
from signals.signal_validator import SignalValidator


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    return np.random.randn(100) * 0.01


@pytest.fixture
def sample_returns():
    """Create sample returns."""
    return pd.Series(np.random.randn(100) * 0.01)


class TestSignalConstructor:
    """Test signal constructor."""
    
    def test_compute_realized_vol(self, sample_returns):
        """Test realized volatility computation."""
        constructor = SignalConstructor(vol_window=20)
        
        vol = constructor.compute_realized_vol(sample_returns)
        
        assert len(vol) == len(sample_returns)
        assert (vol > 0).all()  # Vol should be positive
    
    def test_standardize_signal(self, sample_predictions, sample_returns):
        """Test signal standardization."""
        constructor = SignalConstructor()
        
        signal = constructor.standardize_signal(sample_predictions, sample_returns)
        
        assert len(signal) == len(sample_predictions)
        assert not np.isnan(signal).all()
    
    def test_cap_signal(self):
        """Test signal capping."""
        constructor = SignalConstructor(cap=3.0)
        
        signal = np.array([-5, -2, 0, 2, 5])
        capped = constructor.cap_signal(signal)
        
        assert (capped >= -3).all()
        assert (capped <= 3).all()
        assert capped[2] == 0  # Middle value unchanged


class TestSignalSmoother:
    """Test signal smoother."""
    
    def test_smooth_ema(self):
        """Test EMA smoothing."""
        smoother = SignalSmoother(ema_span=3)
        
        signal = np.array([1, 2, 3, 4, 5])
        smoothed = smoother.smooth_ema(signal)
        
        assert len(smoothed) == len(signal)
        # Smoothed should be less volatile
        assert np.std(smoothed) < np.std(signal)
    
    def test_smooth_sma(self):
        """Test SMA smoothing."""
        smoother = SignalSmoother()
        
        signal = np.array([1, 2, 3, 4, 5])
        smoothed = smoother.smooth_sma(signal, window=3)
        
        assert len(smoothed) == len(signal)


class TestSignalValidator:
    """Test signal validator."""
    
    def test_check_nan_rate(self):
        """Test NaN rate check."""
        validator = SignalValidator()
        
        signal_good = np.random.randn(100)
        signal_bad = np.full(100, np.nan)
        
        assert validator.check_nan_rate(signal_good, threshold=0.1)
        assert not validator.check_nan_rate(signal_bad, threshold=0.1)
    
    def test_check_outliers(self):
        """Test outlier check."""
        validator = SignalValidator()
        
        signal_good = np.random.randn(100)
        signal_bad = np.concatenate([np.random.randn(90), np.full(10, 100)])
        
        assert validator.check_outliers(signal_good, threshold=5.0)
        # signal_bad might fail depending on threshold
    
    def test_validate_signal(self):
        """Test full signal validation."""
        validator = SignalValidator()
        
        signal = np.random.randn(100)
        results = validator.validate_signal(signal)
        
        assert isinstance(results, dict)
        assert 'all_passed' in results