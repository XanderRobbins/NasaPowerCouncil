"""
Council Orchestrator: Coordinates all agents and produces final decision.
"""
from typing import Dict, Any, List
from loguru import logger

from council.data_integrity_agent import DataIntegrityAgent
from council.feature_drift_agent import FeatureDriftAgent
from council.model_stability_agent import ModelStabilityAgent
from council.regime_agent import RegimeAgent
from council.red_team_agent import RedTeamAgent
from council.capital_allocator_agent import CapitalAllocatorAgent


class CouncilOrchestrator:
    """
    Orchestrate all council agents to produce final position sizing decision.
    
    Flow:
    1. Data Integrity Agent validates data
    2. Feature Drift Agent checks distribution stability
    3. Model Stability Agent checks performance
    4. Regime Agent assesses market conditions
    5. Red Team Agent finds potential failure modes
    6. Capital Allocator Agent combines all inputs
    """
    
    def __init__(self):
        self.agents = {
            'data_integrity': DataIntegrityAgent(),
            'feature_drift': FeatureDriftAgent(),
            'model_stability': ModelStabilityAgent(),
            'regime': RegimeAgent(),
            'red_team': RedTeamAgent(),
            'capital_allocator': CapitalAllocatorAgent()
        }
        
        self.evaluation_results = {}
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all agents and produce final decision.
        
        Args:
            context: Dict containing all relevant data
                - features: Feature DataFrame
                - prices: Price series
                - returns: Return series
                - predictions: Model predictions
                - actuals: Actual returns
                - raw_signal: Raw signal value
                - commodity: Commodity name
                - current_date: Current date
                - (optional) cot_positioning, dollar_surge, etc.
        
        Returns:
            Dict with final decision and all agent evaluations
        """
        logger.info("=" * 60)
        logger.info("COUNCIL EVALUATION STARTING")
        logger.info("=" * 60)
        
        # Step 1: Data Integrity
        logger.info("\n[1/6] Data Integrity Agent")
        data_integrity_result = self.agents['data_integrity'].evaluate(context)
        data_integrity_score = self.agents['data_integrity'].get_score()
        logger.info(f"Score: {data_integrity_score:.2f} | Recommendation: {self.agents['data_integrity'].get_recommendation()}")
        
        # Veto if data integrity fails
        if data_integrity_score < 0.3:
            logger.error("✗ DATA INTEGRITY VETO - Aborting trade")
            return {
                'decision': 'VETO',
                'reason': 'Data integrity failure',
                'final_weight': 0.0,
                'agent_results': {'data_integrity': data_integrity_result}
            }
        
        # Step 2: Feature Drift
        logger.info("\n[2/6] Feature Drift Agent")
        feature_drift_result = self.agents['feature_drift'].evaluate(context)
        feature_drift_score = self.agents['feature_drift'].get_score()
        logger.info(f"Score: {feature_drift_score:.2f} | Recommendation: {self.agents['feature_drift'].get_recommendation()}")
        
        # Step 3: Model Stability
        logger.info("\n[3/6] Model Stability Agent")
        model_stability_result = self.agents['model_stability'].evaluate(context)
        model_stability_score = self.agents['model_stability'].get_score()
        logger.info(f"Score: {model_stability_score:.2f} | Recommendation: {self.agents['model_stability'].get_recommendation()}")
        
        # Step 4: Regime
        logger.info("\n[4/6] Regime Agent")
        regime_result = self.agents['regime'].evaluate(context)
        regime_score = self.agents['regime'].get_score()
        logger.info(f"Score: {regime_score:.2f} | Recommendation: {self.agents['regime'].get_recommendation()}")
        
        # Step 5: Red Team
        logger.info("\n[5/6] Red Team Agent")
        red_team_result = self.agents['red_team'].evaluate(context)
        red_team_penalty = red_team_result['penalty']
        logger.info(f"Penalty: {red_team_penalty:.2f} | Red Flags: {red_team_result['red_flags']}")
        
        # Step 6: Capital Allocation
        logger.info("\n[6/6] Capital Allocator Agent")
        
        # Build context for capital allocator
        allocator_context = {
            'raw_signal': context.get('raw_signal', 0.0),
            'data_integrity_score': data_integrity_score,
            'feature_drift_score': feature_drift_score,
            'model_stability_score': model_stability_score,
            'regime_score': regime_score,
            'red_team_penalty': red_team_penalty
        }
        
        capital_result = self.agents['capital_allocator'].evaluate(allocator_context)
        final_weight = self.agents['capital_allocator'].get_final_weight()
        
        logger.info(f"Final Weight: {final_weight:.4f}")
        logger.info(f"Net Adjustment: {capital_result['net_adjustment']:.2%}")
        
        # Determine decision
        if abs(final_weight) < 0.01:
            decision = 'NO_TRADE'
            reason = 'Final weight below threshold'
        elif abs(final_weight) < abs(context.get('raw_signal', 0.0)) * 0.3:
            decision = 'REDUCE'
            reason = 'Significant risk adjustment applied'
        else:
            decision = 'PROCEED'
            reason = 'All checks passed'
        
        # Compile results
        results = {
            'decision': decision,
            'reason': reason,
            'final_weight': final_weight,
            'raw_signal': context.get('raw_signal', 0.0),
            'agent_results': {
                'data_integrity': data_integrity_result,
                'feature_drift': feature_drift_result,
                'model_stability': model_stability_result,
                'regime': regime_result,
                'red_team': red_team_result,
                'capital_allocator': capital_result
            },
            'component_scores': {
                'data_integrity': data_integrity_score,
                'feature_drift': feature_drift_score,
                'model_stability': model_stability_score,
                'regime': regime_score,
                'red_team_penalty': red_team_penalty
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info(f"FINAL DECISION: {decision}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Position Weight: {final_weight:.4f}")
        logger.info("=" * 60 + "\n")
        
        return results
    
    def get_summary(self, results: Dict[str, Any]) -> str:
        """Generate human-readable summary of council decision."""
        summary = f"""
COUNCIL DECISION SUMMARY
========================

Decision: {results['decision']}
Reason: {results['reason']}

Raw Signal: {results['raw_signal']:.4f}
Final Weight: {results['final_weight']:.4f}
Adjustment: {(results['final_weight'] / (results['raw_signal'] + 1e-8)):.2%}

Component Scores:
  - Data Integrity: {results['component_scores']['data_integrity']:.2f}
  - Feature Drift: {results['component_scores']['feature_drift']:.2f}
  - Model Stability: {results['component_scores']['model_stability']:.2f}
  - Regime: {results['component_scores']['regime']:.2f}
  - Red Team Penalty: {results['component_scores']['red_team_penalty']:.2f}

Red Flags: {results['agent_results']['red_team']['red_flags']}
"""
        return summary


def run_council(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to run the council.
    
    Args:
        context: Dict with all required data
        
    Returns:
        Council decision
    """
    orchestrator = CouncilOrchestrator()
    results = orchestrator.evaluate(context)
    
    return results