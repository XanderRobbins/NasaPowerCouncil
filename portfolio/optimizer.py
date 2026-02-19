"""
Portfolio optimization for risk parity and optimal allocation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy.optimize import minimize
from loguru import logger


class PortfolioOptimizer:
    """
    Optimize portfolio allocation.
    
    Methods:
    - Risk parity (equal risk contribution)
    - Mean-variance optimization
    - Maximum diversification
    - Minimum variance
    """
    
    def __init__(self, method: str = 'risk_parity'):
        """
        Args:
            method: Optimization method ('risk_parity', 'mean_variance', 'min_variance')
        """
        self.method = method
        
    def optimize(self,
                signals: Dict[str, float],
                returns: Dict[str, pd.Series],
                covariance: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights.
        
        Args:
            signals: Dict mapping commodity -> signal strength
            returns: Dict mapping commodity -> return series
            covariance: Optional covariance matrix
            
        Returns:
            Dict mapping commodity -> optimized weight
        """
        commodities = list(signals.keys())
        n = len(commodities)
        
        if n == 0:
            return {}
        
        # Compute covariance if not provided
        if covariance is None:
            returns_df = pd.DataFrame(returns)
            covariance = returns_df.cov().values
        
        # Initial weights from signals (normalized)
        signal_array = np.array([signals[c] for c in commodities])
        initial_weights = signal_array / np.sum(np.abs(signal_array)) if np.sum(np.abs(signal_array)) > 0 else np.ones(n) / n
        
        # Optimize based on method
        if self.method == 'risk_parity':
            optimized_weights = self._risk_parity_optimization(covariance, initial_weights)
        elif self.method == 'mean_variance':
            expected_returns = np.array([returns[c].mean() for c in commodities])
            optimized_weights = self._mean_variance_optimization(expected_returns, covariance, initial_weights)
        elif self.method == 'min_variance':
            optimized_weights = self._min_variance_optimization(covariance)
        else:
            logger.warning(f"Unknown optimization method: {self.method}, using signal weights")
            optimized_weights = initial_weights
        
        # Convert back to dict
        result = {commodity: weight for commodity, weight in zip(commodities, optimized_weights)}
        
        return result
    
    def _risk_parity_optimization(self, covariance: np.ndarray, initial_weights: np.ndarray) -> np.ndarray:
        """
        Risk parity optimization: equal risk contribution.
        
        Minimize: Σ(RC_i - RC_target)²
        where RC_i = w_i × (Σ × w)_i
        """
        n = len(initial_weights)
        
        def risk_contribution(weights):
            """Compute risk contribution for each asset."""
            portfolio_vol = np.sqrt(weights @ covariance @ weights)
            marginal_contrib = covariance @ weights
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def objective(weights):
            """Objective: minimize variance of risk contributions."""
            rc = risk_contribution(weights)
            target_rc = np.mean(rc)
            return np.sum((rc - target_rc) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds: allow long and short
        bounds = [(-0.5, 0.5) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")
            return initial_weights
        
        return result.x
    
    def _mean_variance_optimization(self,
                                   expected_returns: np.ndarray,
                                   covariance: np.ndarray,
                                   initial_weights: np.ndarray,
                                   risk_aversion: float = 1.0) -> np.ndarray:
        """
        Mean-variance optimization (Markowitz).
        
        Maximize: μ'w - λ/2 × w'Σw
        where λ is risk aversion parameter
        """
        def objective(weights):
            """Negative Sharpe-like objective."""
            portfolio_return = expected_returns @ weights
            portfolio_variance = weights @ covariance @ weights
            return -(portfolio_return - risk_aversion * portfolio_variance)
        
        n = len(initial_weights)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        bounds = [(-0.5, 0.5) for _ in range(n)]
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Mean-variance optimization did not converge: {result.message}")
            return initial_weights
        
        return result.x
    
    def _min_variance_optimization(self, covariance: np.ndarray) -> np.ndarray:
        """
        Minimum variance optimization.
        
        Minimize: w'Σw
        """
        n = covariance.shape[0]
        
        def objective(weights):
            return weights @ covariance @ weights
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        bounds = [(-0.5, 0.5) for _ in range(n)]
        
        initial_weights = np.ones(n) / n
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning(f"Minimum variance optimization did not converge: {result.message}")
            return initial_weights
        
        return result.x
    
    def compute_portfolio_metrics(self,
                                 weights: Dict[str, float],
                                 returns: Dict[str, pd.Series],
                                 covariance: Optional[np.ndarray] = None) -> Dict:
        """
        Compute portfolio metrics for given weights.
        
        Returns:
            Dict with expected return, volatility, Sharpe, etc.
        """
        commodities = list(weights.keys())
        weight_array = np.array([weights[c] for c in commodities])
        
        # Compute covariance if not provided
        if covariance is None:
            returns_df = pd.DataFrame(returns)
            covariance = returns_df.cov().values
        
        # Expected return
        expected_returns = np.array([returns[c].mean() for c in commodities])
        portfolio_return = expected_returns @ weight_array
        
        # Volatility
        portfolio_variance = weight_array @ covariance @ weight_array
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (annualized)
        sharpe = (portfolio_return * 252) / (portfolio_vol * np.sqrt(252)) if portfolio_vol > 0 else 0
        
        # Diversification ratio
        asset_vols = np.sqrt(np.diag(covariance))
        weighted_vol = np.abs(weight_array) @ asset_vols
        diversification = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        return {
            'expected_return_daily': portfolio_return,
            'expected_return_annual': portfolio_return * 252,
            'volatility_daily': portfolio_vol,
            'volatility_annual': portfolio_vol * np.sqrt(252),
            'sharpe_ratio': sharpe,
            'diversification_ratio': diversification
        }


class BlackLittermanOptimizer(PortfolioOptimizer):
    """
    Black-Litterman optimization.
    
    Combines market equilibrium with investor views (our signals).
    """
    
    def __init__(self, tau: float = 0.025, risk_aversion: float = 2.5):
        """
        Args:
            tau: Uncertainty in prior (typically 0.01 - 0.05)
            risk_aversion: Market risk aversion parameter
        """
        super().__init__(method='black_litterman')
        self.tau = tau
        self.risk_aversion = risk_aversion
        
    def optimize(self,
                signals: Dict[str, float],
                returns: Dict[str, pd.Series],
                covariance: Optional[np.ndarray] = None,
                market_caps: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Black-Litterman optimization.
        
        Args:
            signals: Investor views (our model signals)
            returns: Historical returns
            covariance: Covariance matrix
            market_caps: Market capitalizations (for equilibrium weights)
            
        Returns:
            Optimized weights
        """
        commodities = list(signals.keys())
        n = len(commodities)
        
        if covariance is None:
            returns_df = pd.DataFrame(returns)
            covariance = returns_df.cov().values
        
        # Market equilibrium weights (if not provided, use equal weight)
        if market_caps is None:
            market_weights = np.ones(n) / n
        else:
            total_cap = sum(market_caps.values())
            market_weights = np.array([market_caps.get(c, 0) / total_cap for c in commodities])
        
        # Implied equilibrium returns: Π = λ × Σ × w_mkt
        implied_returns = self.risk_aversion * (covariance @ market_weights)
        
        # Views (P matrix and Q vector)
        # For simplicity, assume we have a view on each asset
        P = np.eye(n)  # Identity: one view per asset
        Q = np.array([signals[c] for c in commodities])  # Our signal views
        
        # View uncertainty (Ω)
        # Proportional to variance of each asset
        omega = np.diag(np.diag(self.tau * covariance))
        
        # Black-Litterman formula
        # Posterior returns: μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹Π + P'Ω⁻¹Q]
        tau_sigma_inv = np.linalg.inv(self.tau * covariance)
        omega_inv = np.linalg.inv(omega)
        
        posterior_cov_inv = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_cov = np.linalg.inv(posterior_cov_inv)
        
        posterior_returns = posterior_cov @ (tau_sigma_inv @ implied_returns + P.T @ omega_inv @ Q)
        
        # Optimize with posterior returns
        optimized_weights = self._mean_variance_optimization(
            posterior_returns,
            covariance,
            market_weights,
            self.risk_aversion
        )
        
        result = {commodity: weight for commodity, weight in zip(commodities, optimized_weights)}
        
        return result


def optimize_portfolio(signals: Dict[str, float],
                      returns: Dict[str, pd.Series],
                      method: str = 'risk_parity') -> Dict[str, float]:
    """
    Convenience function for portfolio optimization.
    
    Args:
        signals: Signal strengths
        returns: Historical returns
        method: Optimization method
        
    Returns:
        Optimized weights
    """
    optimizer = PortfolioOptimizer(method=method)
    weights = optimizer.optimize(signals, returns)
    
    return weights