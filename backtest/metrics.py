"""
Compute backtest performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """
    Compute comprehensive performance metrics.
    """
    
    def __init__(self, results: pd.DataFrame):
        self.results = results
        self.returns = results['portfolio_return']
        
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all performance metrics."""
        metrics = {
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
            'win_rate': self.win_rate(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'profit_factor': self.profit_factor(),
            'volatility': self.volatility(),
        }
        
        return metrics
    
    def total_return(self) -> float:
        """Total return over the period."""
        return (self.results['portfolio_value'].iloc[-1] / self.results['portfolio_value'].iloc[0]) - 1
    
    def cagr(self) -> float:
        """Compound Annual Growth Rate."""
        total_ret = self.total_return()
        n_years = len(self.returns) / 252
        return (1 + total_ret) ** (1 / n_years) - 1
    
    def sharpe_ratio(self, rf_rate: float = 0.0) -> float:
        """Sharpe ratio (annualized)."""
        excess_returns = self.returns - rf_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def sortino_ratio(self, rf_rate: float = 0.0) -> float:
        """Sortino ratio (annualized)."""
        excess_returns = self.returns - rf_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return np.sqrt(252) * excess_returns.mean() / downside_std if downside_std > 0 else 0
    
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1)
        return drawdown.min()
    
    def calmar_ratio(self) -> float:
        """Calmar ratio (CAGR / Max Drawdown)."""
        max_dd = abs(self.max_drawdown())
        return self.cagr() / max_dd if max_dd > 0 else 0
    
    def win_rate(self) -> float:
        """Percentage of winning days."""
        return (self.returns > 0).sum() / len(self.returns)
    
    def avg_win(self) -> float:
        """Average return on winning days."""
        wins = self.returns[self.returns > 0]
        return wins.mean() if len(wins) > 0 else 0
    
    def avg_loss(self) -> float:
        """Average return on losing days."""
        losses = self.returns[self.returns < 0]
        return losses.mean() if len(losses) > 0 else 0
    
    def profit_factor(self) -> float:
        """Profit factor (total wins / total losses)."""
        wins = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        return wins / losses if losses > 0 else 0
    
    def volatility(self) -> float:
        """Annualized volatility."""
        return self.returns.std() * np.sqrt(252)
    
    def print_summary(self):
        """Print performance summary."""
        metrics = self.compute_all_metrics()
        
        print("\n" + "=" * 60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total Return:      {metrics['total_return']:>10.2%}")
        print(f"CAGR:              {metrics['cagr']:>10.2%}")
        print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:     {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:      {metrics['max_drawdown']:>10.2%}")
        print(f"Calmar Ratio:      {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:          {metrics['win_rate']:>10.2%}")
        print(f"Avg Win:           {metrics['avg_win']:>10.4f}")
        print(f"Avg Loss:          {metrics['avg_loss']:>10.4f}")
        print(f"Profit Factor:     {metrics['profit_factor']:>10.2f}")
        print(f"Volatility (Ann.): {metrics['volatility']:>10.2%}")
        print("=" * 60 + "\n")


def compute_metrics(results: pd.DataFrame) -> Dict[str, float]:
    """Convenience function."""
    metrics_calc = PerformanceMetrics(results)
    return metrics_calc.compute_all_metrics()