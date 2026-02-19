"""
Capital Allocator Agent: Combines all agent inputs to determine final position size.
"""
import numpy as np
from typing import Dict, Any

from council.base_agent import BaseAgent


class CapitalAllocatorAgent(BaseAgent):
    """
    Final arbiter of position sizing.
    
    Combines:
    - Raw signal strength
    - Data integrity score
    - Feature drift score
    - Model stability score
    - Regime score
    - Red team penalty
    
    Formula:
    FinalWeight = RawSignal × DataIntegrity × FeatureDrift × ModelStability × Regime × (1 - RedTeamPenalty)
    """
    
    def __init__(self):
        super().__init__("CapitalAllocatorAgent")
        self.final_weight = 0.0
        self.component_scores = {}
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute final position weight.
        """
        # Get raw signal
        raw_signal = context.get('raw_signal', 0.0)
        
        # Get agent scores
        data_integrity_score = context.get('data_integrity_score', 1.0)
        feature_drift_score = context.get('feature_drift_score', 1.0)
        model_stability_score = context.get('model_stability_score', 1.0)
        regime_score = context.get('regime_score', 1.0)
        red_team_penalty = context.get('red_team_penalty', 0.0)
        
        # Store components
        self.component_scores = {
            'raw_signal': raw_signal,
            'data_integrity': data_integrity_score,
            'feature_drift': feature_drift_score,
            'model_stability': model_stability_score,
            'regime': regime_score,
            'red_team_penalty': red_team_penalty
        }
        
        # Compute final weight
        self.final_weight = (
            raw_signal 
            * data_integrity_score 
            * feature_drift_score 
            * model_stability_score 
            * regime_score 
            * (1 - red_team_penalty)
        )
        
        # Additional sanity checks
        if abs(self.final_weight) < 0.01:
            # Signal too weak
            self.final_weight = 0.0
        
        # Cap at ±1.0 (will be scaled by vol targeting later)
        self.final_weight = np.clip(self.final_weight, -1.0, 1.0)
        
        return {
            'final_weight': self.final_weight,
            'components': self.component_scores,
            'net_adjustment': self.final_weight / (raw_signal + 1e-8) if raw_signal != 0 else 0
        }
    
    def get_score(self) -> float:
        """Return final weight magnitude as score."""
        return abs(self.final_weight)
    
    def get_final_weight(self) -> float:
        """Get the final position weight."""
        return self.final_weight