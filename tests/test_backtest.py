"""
Tests for backtest module.
"""
import pytest
import pandas as pd
import numpy as np

from backtest.metrics import PerformanceMetrics
from backtest.engine import BacktestEngine


@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Random walk with positive drift
    returns = np.random.randn(len(dates)) * 0.01 + 0.0001
    portfolio_value = 1_000_000 * np.exp(np.cumsum(returns))
    daily_pnl = np.diff(portfolio_value, prepend=portfolio_value[0])
    
    df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_value,
        'daily_pnl': daily_pnl,
        'portfolio_return': returns
    })
    
    return df


class TestPerformanceMetrics:
    """Test performance metrics."""
    
    def test_compute_all_metrics(self, sample_backtest_results):
        """Test all metrics computation."""
        metrics = PerformanceMetrics(sample_backtest_results)
        
        all_metrics = metrics.compute_all_metrics()
        
        assert 'total_return' in all_metrics
        assert 'cagr' in all_metrics
        assert 'sharpe_ratio' in all_metrics
        assert 'max_drawdown' in all_metrics
        assert 'win_rate' in all_metrics
    
    def test_sharpe_ratio(self, sample_backtest_results):
        """Test Sharpe ratio calculation."""
        metrics = PerformanceMetrics(sample_backtest_results)
        
        sharpe = metrics.sharpe_ratio()
        
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)
    
    def test_max_drawdown(self, sample_backtest_results):
        """Test max drawdown calculation."""
        metrics = PerformanceMetrics(sample_backtest_results)
        
        max_dd = metrics.max_drawdown()
        
        assert max_dd <= 0  # Drawdown should be negative
        assert max_dd >= -1  # Should be > -100%
    
    def test_win_rate(self, sample_backtest_results):
        """Test win rate calculation."""
        metrics = PerformanceMetrics(sample_backtest_results)
        
        win_rate = metrics.win_rate()
        
        assert 0 <= win_rate <= 1


class TestBacktestEngine:
    """Test backtest engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = BacktestEngine(
            start_date='2020-01-01',
            end_date='2020-12-31',
            commodities=['corn'],
            initial_capital=1_000_000
        )
        
        assert engine.initial_capital == 1_000_000
        assert 'corn' in engine.commodities