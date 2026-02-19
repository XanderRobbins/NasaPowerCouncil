"""
Base class for all council agents.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from loguru import logger


class BaseAgent(ABC):
    """
    Abstract base class for council agents.
    
    All agents must implement:
    - evaluate(): Assess the current state
    - get_score(): Return a score/weight (0-1)
    - get_recommendation(): Return action recommendation
    """
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate current state and return assessment.
        
        Args:
            context: Dict containing relevant data (signals, features, prices, etc.)
            
        Returns:
            Dict with evaluation results
        """
        pass
    
    @abstractmethod
    def get_score(self) -> float:
        """
        Return confidence/quality score (0-1).
        
        1.0 = full confidence
        0.0 = no confidence (veto)
        """
        pass
    
    def get_recommendation(self) -> str:
        """
        Return recommendation: 'proceed', 'reduce', or 'veto'.
        """
        score = self.get_score()
        
        if score >= 0.8:
            return 'proceed'
        elif score >= 0.5:
            return 'reduce'
        else:
            return 'veto'
    
    def log_evaluation(self, evaluation: Dict[str, Any]):
        """Log evaluation results."""
        logger.info(f"[{self.name}] Score: {self.get_score():.2f} | Recommendation: {self.get_recommendation()}")
        logger.debug(f"[{self.name}] Details: {evaluation}")