"""
Tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from features.seasonal_deviation import SeasonalDeviationCalculator
from features.stress_calculator import StressCalculator
from features.cumulative_stress import CumulativeStressCalculator
from features.regional_aggregator import RegionalAggregator
from features.feature_pipeline import FeaturePipeline


@pytest.fixture
def sample_climate_data():
    """Create sample climate data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'temp_avg': 20 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'temp_max': 25 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'temp_min': 15 + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'precipitation': np.random.exponential(2, len(dates)),
        'solar_radiation': 15 + 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365),
        'wind_speed': np.random.uniform(2, 8, len(dates)),
        'relative_humidity': np.random.uniform(40, 80, len(dates)),
        'region': 'Iowa'
    })
    
    return df


class TestSeasonalDeviation:
    """Test seasonal deviation calculator."""
    
    def test_compute_baseline(self, sample_climate_data):
        """Test baseline computation."""
        calculator = SeasonalDeviationCalculator(baseline_years=1)
        
        baseline = calculator.compute_baseline(sample_climate_data, 'temp_max')
        
        assert 'doy' in baseline.columns
        assert 'temp_max_mu' in baseline.columns
        assert 'temp_max_sigma' in baseline.columns
        assert len(baseline) <= 366  # At most 366 days
    
    def test_compute_deviation(self, sample_climate_data):
        """Test Z-score computation."""
        calculator = SeasonalDeviationCalculator(baseline_years=1)
        
        result = calculator.compute_deviation(sample_climate_data, 'temp_max', rolling=False)
        
        assert 'temp_max_z' in result.columns
        assert not result['temp_max_z'].isna().all()
        
        # Z-scores should have mean ~0, std ~1 (approximately)
        z_scores = result['temp_max_z'].dropna()
        assert abs(z_scores.mean()) < 0.5
        assert 0.5 < z_scores.std() < 1.5


class TestStressCalculator:
    """Test stress calculator."""
    
    def test_heat_stress(self, sample_climate_data):
        """Test heat stress calculation."""
        calculator = StressCalculator('corn')
        
        result = calculator.compute_heat_stress(sample_climate_data)
        
        assert 'heat_stress' in result.columns
        assert (result['heat_stress'] >= 0).all()  # Heat stress should be non-negative
    
    def test_dry_stress(self, sample_climate_data):
        """Test dry stress calculation."""
        calculator = StressCalculator('corn')
        
        result = calculator.compute_dry_stress(sample_climate_data)
        
        assert 'dry_stress' in result.columns
        assert (result['dry_stress'] >= 0).all()
    
    def test_gdd(self, sample_climate_data):
        """Test growing degree days."""
        calculator = StressCalculator('corn')
        
        result = calculator.compute_gdd(sample_climate_data)
        
        assert 'gdd' in result.columns
        assert 'gdd_cumsum' in result.columns
        assert (result['gdd'] >= 0).all()
        assert result['gdd_cumsum'].is_monotonic_increasing


class TestCumulativeStress:
    """Test cumulative stress calculator."""
    
    def test_cumulative_computation(self, sample_climate_data):
        """Test rolling sum computation."""
        # First add a stress column
        sample_climate_data['heat_stress'] = np.random.uniform(0, 5, len(sample_climate_data))
        
        calculator = CumulativeStressCalculator(windows=[7, 14, 30])
        
        result = calculator.compute_cumulative(sample_climate_data, 'heat_stress')
        
        assert 'heat_stress_7d' in result.columns
        assert 'heat_stress_14d' in result.columns
        assert 'heat_stress_30d' in result.columns
        
        # 30-day cumulative should be >= 7-day cumulative
        assert (result['heat_stress_30d'] >= result['heat_stress_7d']).all()


class TestRegionalAggregator:
    """Test regional aggregator."""
    
    def test_aggregate_feature(self, sample_climate_data):
        """Test feature aggregation."""
        # Create multi-region data
        sample_climate_data['heat_stress_7d'] = np.random.uniform(0, 10, len(sample_climate_data))
        
        region_dfs = {
            'Iowa': sample_climate_data.copy(),
            'Illinois': sample_climate_data.copy()
        }
        
        aggregator = RegionalAggregator('corn')
        
        result = aggregator.aggregate_feature(region_dfs, 'heat_stress_7d')
        
        assert 'heat_stress_7d_agg' in result.columns
        assert len(result) == len(sample_climate_data)


class TestFeaturePipeline:
    """Test full feature pipeline."""
    
    def test_pipeline_execution(self, sample_climate_data):
        """Test end-to-end pipeline."""
        region_dfs = {
            'Iowa': sample_climate_data,
            'Illinois': sample_climate_data.copy()
        }
        
        pipeline = FeaturePipeline('corn')
        
        result = pipeline.run(region_dfs)
        
        assert isinstance(result, pd.DataFrame)
        assert 'date' in result.columns
        assert 'commodity' in result.columns
        assert len(result) > 0