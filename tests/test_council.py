"""
Tests for council module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from council.data_integrity_agent import DataIntegrityAgent
from council.feature_drift_agent import FeatureDriftAgent
from council.model_stability_agent import ModelStabilityAgent
from council.regime_agent import RegimeAgent
from council.red_team_agent import RedTeamAgent
from council.capital_allocator_agent import CapitalAllocatorAgent
from council.council_orchestrator import CouncilOrchestrator


@pytest.fixture
def sample_context():
    """Create sample context for council."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    features = pd.DataFrame({
        'date': dates,
        'feature_1': np.random.randn(len(dates)),
        'temp_max': np.random.uniform(20, 35, len(dates))
    })
    
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)), index=dates)
    returns = prices.pct_change()
    
    return {
        'features': features,
        'prices': prices,
        'returns': returns,
        'predictions': np.random.randn(100) * 0.01,
        'actuals': np.random.randn(100) * 0.01,
        'raw_signal': 0.5,
        'commodity': 'corn',
        'current_date': datetime.now()
    }


class TestDataIntegrityAgent:
    """Test data integrity agent."""
    
    def test_evaluate(self, sample_context):
        """Test data integrity evaluation."""
        agent = DataIntegrityAgent()
        
        result = agent.evaluate(sample_context)
        
        assert 'score' in result
        assert 0 <= agent.get_score() <= 1
    
    def test_missing_data(self, sample_context):
        """Test detection of missing data."""
        agent = DataIntegrityAgent()
        
        # Add missing values
        sample_context['features'].loc[0:100, 'feature_1'] = np.nan
        
        result = agent.evaluate(sample_context)
        
        assert agent.get_score() < 1.0


class TestFeatureDriftAgent:
    """Test feature drift agent."""
    
    def test_evaluate(self, sample_context):
        """Test feature drift evaluation."""
        agent = FeatureDriftAgent(lookback=252)
        
        result = agent.evaluate(sample_context)
        
        assert 'score' in result
        assert 0 <= agent.get_score() <= 1


class TestModelStabilityAgent:
    """Test model stability agent."""
    
    def test_evaluate(self, sample_context):
        """Test model stability evaluation."""
        agent = ModelStabilityAgent(window=60)
        
        result = agent.evaluate(sample_context)
        
        assert 'score' in result
        assert 0 <= agent.get_score() <= 1


class TestRegimeAgent:
    """Test regime agent."""
    
    def test_evaluate(self, sample_context):
        """Test regime evaluation."""
        agent = RegimeAgent(lookback=252)
        
        result = agent.evaluate(sample_context)
        
        assert 'score' in result
        assert 'regime' in result
        assert 0 <= agent.get_score() <= 1


class TestRedTeamAgent:
    """Test red team agent."""
    
    def test_evaluate(self, sample_context):
        """Test red team evaluation."""
        agent = RedTeamAgent()
        
        result = agent.evaluate(sample_context)
        
        assert 'penalty' in result
        assert 'red_flags' in result
        assert 0 <= result['penalty'] <= 1


class TestCapitalAllocatorAgent:
    """Test capital allocator agent."""
    
    def test_evaluate(self):
        """Test capital allocation."""
        agent = CapitalAllocatorAgent()
        
        context = {
            'raw_signal': 0.8,
            'data_integrity_score': 1.0,
            'feature_drift_score': 0.9,
            'model_stability_score': 0.85,
            'regime_score': 0.9,
            'red_team_penalty': 0.1
        }
        
        result = agent.evaluate(context)
        
        assert 'final_weight' in result
        assert abs(result['final_weight']) <= 1.0


class TestCouncilOrchestrator:
    """Test council orchestrator."""
    
    def test_evaluate(self, sample_context):
        """Test full council evaluation."""
        orchestrator = CouncilOrchestrator()
        
        result = orchestrator.evaluate(sample_context)
        
        assert 'decision' in result
        assert 'final_weight' in result
        assert 'agent_results' in result
        assert result['decision'] in ['PROCEED', 'REDUCE', 'NO_TRADE', 'VETO']