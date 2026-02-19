"""
Regime Agent: Detects macro regime changes that could invalidate weather signals.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.stats import hmm

from council.base_agent import BaseAgent


class RegimeAgent(BaseAgent):
    """
    Detect market regime changes.
    
    Regimes to monitor:
    - Volatility regime (low/high vol)
    - Trend regime (trending vs mean-reverting)
    - Correlation regime (correlations spike = risk-off)
    - Hurst exponent (persistence vs anti-persistence)
    
    Weather signals work best in certain regimes.
    """
    
    def __init__(self, lookback: int = 252):
        super().__init__("RegimeAgent")
        self.lookback = lookback
        self.score = 1.0
        self.current_regime = {}
        
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect current market regime.
        """
        prices = context.get('prices')
        returns = context.get('returns')
        
        if prices is None or returns is None:
            self.score = 0.5
            return {'score': self.score}
        
        if len(returns) < self.lookback:
            self.score = 0.8
            return {'score': self.score, 'note': 'Insufficient data'}
        
        recent_returns = returns.iloc[-self.lookback:]
        
        # 1. Volatility Regime
        vol = recent_returns.std() * np.sqrt(252)
        hist_vol = returns.std() * np.sqrt(252)
        vol_ratio = vol / hist_vol
        
        if vol_ratio > 1.5:
            vol_regime = 'high_vol'
            vol_penalty = 0.7  # Weather signals less reliable in high vol
        elif vol_ratio < 0.7:
            vol_regime = 'low_vol'
            vol_penalty = 1.0
        else:
            vol_regime = 'normal_vol'
            vol_penalty = 0.9
        
        # 2. Trend Regime (Hurst exponent)
        hurst = self._compute_hurst(recent_returns.values)
        
        if hurst > 0.55:
            trend_regime = 'trending'
            trend_penalty = 0.8  # Weather signals struggle in strong trends
        elif hurst < 0.45:
            trend_regime = 'mean_reverting'
            trend_penalty = 1.0  # Good for weather signals
        else:
            trend_regime = 'random_walk'
            trend_penalty = 0.9
        
        # 3. Drawdown Check
        prices_series = pd.Series(prices.values) if hasattr(prices, 'values') else pd.Series(prices)
        cumulative = (1 + recent_returns).cumprod()
        drawdown = (cumulative / cumulative.cummax() - 1).min()
        
        if drawdown < -0.15:
            drawdown_regime = 'deep_drawdown'
            drawdown_penalty = 0.5  # Reduce exposure in drawdown
        elif drawdown < -0.08:
            drawdown_regime = 'moderate_drawdown'
            drawdown_penalty = 0.7
        else:
            drawdown_regime = 'normal'
            drawdown_penalty = 1.0
        
        # 4. Dollar Strength (if available in context)
        dollar_surge = context.get('dollar_surge', False)
        if dollar_surge:
            dollar_penalty = 0.7  # Commodities inverse to dollar
        else:
            dollar_penalty = 1.0
        
        # Aggregate score
        self.score = vol_penalty * trend_penalty * drawdown_penalty * dollar_penalty
        
        self.current_regime = {
            'vol_regime': vol_regime,
            'trend_regime': trend_regime,
            'drawdown_regime': drawdown_regime,
            'hurst': hurst,
            'vol_ratio': vol_ratio,
            'drawdown': drawdown
        }
        
        return {
            'score': self.score,
            'regime': self.current_regime,
            'penalties': {
                'vol': vol_penalty,
                'trend': trend_penalty,
                'drawdown': drawdown_penalty,
                'dollar': dollar_penalty
            }
        }
    
    def _compute_hurst(self, returns: np.ndarray) -> float:
        """
        Compute Hurst exponent.
        
        H > 0.5: Trending (persistent)
        H = 0.5: Random walk
        H < 0.5: Mean-reverting (anti-persistent)
        """
        if len(returns) < 50:
            return 0.5
        
        # Remove NaNs
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 50:
            return 0.5
        
        # Compute cumulative deviate
        cumsum = np.cumsum(returns - np.mean(returns))
        
        # R/S analysis
        lags = range(2, min(100, len(returns) // 2))
        rs_values = []
        
        for lag in lags:
            # Split into chunks
            n_chunks = len(cumsum) // lag
            if n_chunks == 0:
                continue
            
            rs_chunk = []
            for i in range(n_chunks):
                chunk = cumsum[i*lag:(i+1)*lag]
                if len(chunk) == 0:
                    continue
                R = np.max(chunk) - np.min(chunk)
                S = np.std(returns[i*lag:(i+1)*lag])
                if S > 0:
                    rs_chunk.append(R / S)
            
            if rs_chunk:
                rs_values.append(np.mean(rs_chunk))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression: log(R/S) vs log(lag)
        log_lags = np.log([lag for lag in lags[:len(rs_values)]])
        log_rs = np.log(rs_values)
        
        # Simple linear regression
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        
        return np.clip(hurst, 0.0, 1.0)
    
    def get_score(self) -> float:
        return self.score