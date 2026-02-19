"""
Tests for models module.
"""
import pytest
import pandas as pd
import numpy as np

from models.ridge_model import RollingRidgeModel
from models.kalman_beta import DynamicBetaEstimator
from models.model_trainer import ModelTrainer


@pytest.fixture
def sample_features():
    """Create sample feature data."""
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    df = pd.DataFrame({
        'date': dates,
        'feature_1_agg': np.random.randn(n),
        'feature_2_agg': np.random.randn(n),
        'feature_3_agg': np.random.randn(n),
        'heat_stress_7d_agg': np.random.uniform(0, 10, n),
        'dry_stress_14d_agg': np.random.uniform(0, 10, n),
    })
    
    return df


@pytest.fixture
def sample_prices():
    """Create sample price data."""
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    
    # Random walk
    returns = np.random.randn(len(dates)) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    return pd.Series(prices, index=dates)


class TestRollingRidge:
    """Test rolling ridge model."""
    
    def test_prepare_features(self, sample_features):
        """Test feature preparation."""
        model = RollingRidgeModel()
        
        X, feature_names = model.prepare_features(sample_features)
        
        assert X.shape[0] == len(sample_features)
        assert X.shape[1] <= model.max_features
        assert len(feature_names) == X.shape[1]
    
    def test_compute_target(self, sample_prices):
        """Test target computation."""
        model = RollingRidgeModel(forward_horizon=20)
        
        target = model.compute_target(sample_prices)
        
        assert len(target) == len(sample_prices)
        assert target.isna().sum() >= 20  # Last 20 should be NaN
    
    def test_cross_validate_alpha(self, sample_features, sample_prices):
        """Test alpha cross-validation."""
        model = RollingRidgeModel()
        
        X, _ = model.prepare_features(sample_features)
        y = model.compute_target(sample_prices).values
        
        best_alpha = model.cross_val_alpha(X, y, alphas=[0.1, 1.0, 10.0])
        
        assert best_alpha in [0.1, 1.0, 10.0]
        assert best_alpha > 0


class TestKalmanBeta:
    """Test Kalman filter beta estimation."""
    
    def test_fit(self):
        """Test Kalman filter fitting."""
        observations = np.random.randn(100)
        
        estimator = DynamicBetaEstimator()
        betas = estimator.fit(observations)
        
        assert len(betas) == len(observations)
        assert not np.isnan(betas).all()
    
    def test_handles_nans(self):
        """Test handling of NaN values."""
        observations = np.random.randn(100)
        observations[10:20] = np.nan
        
        estimator = DynamicBetaEstimator()
        betas = estimator.fit(observations)
        
        assert len(betas) == len(observations)


class TestModelTrainer:
    """Test model trainer."""
    
    def test_train_commodity_model(self, sample_features, sample_prices):
        """Test single commodity training."""
        trainer = ModelTrainer(['corn'])
        
        result = trainer.train_commodity_model('corn', sample_features, sample_prices)
        
        assert 'commodity' in result
        assert 'best_alpha' in result
        assert 'r2' in result
        assert result['commodity'] == 'corn'