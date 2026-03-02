"""
Calculate backtest performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger

def compute_metrics(results: pd.DataFrame) -> Dict:
    if results.empty or 'portfolio_return' not in results.columns:
        logger.warning("Empty results or missing portfolio_return column")
        return {}
    returns = results['portfolio_return'].dropna()
    if len(returns) == 0:
        logger.warning("No valid returns")
        return {}

    active_returns = returns[returns.abs() > 1e-8]
    if len(active_returns) == 0:
        logger.warning("No active trading days found")
        return {}

    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    n_years = (results['date'].iloc[-1] - results['date'].iloc[0]).days / 365.25
    n_active_years = len(active_returns) / 252

    active_vol = active_returns.std() * np.sqrt(252)
    active_annualized_return = (1 + total_return) ** (1 / n_active_years) - 1

    sharpe = active_annualized_return / active_vol if active_vol > 0 else 0

    portfolio_values = results['portfolio_value']
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (active_returns > 0).mean()
    gains = active_returns[active_returns > 0].sum()
    losses = abs(active_returns[active_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    active_downside = active_returns[active_returns < 0]
    active_downside_std = active_downside.std() * np.sqrt(252)
    sortino = active_annualized_return / active_downside_std if active_downside_std > 0 else 0

    calmar = active_annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    avg_win = active_returns[active_returns > 0].mean() if (active_returns > 0).any() else 0
    avg_loss = active_returns[active_returns < 0].mean() if (active_returns < 0).any() else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    metrics = {
        'total_return': total_return,
        'annualized_return': active_annualized_return,
        'annualized_volatility': active_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'n_active_days': len(active_returns),
        'n_days_total': len(returns),
        'n_years': n_years,
        'n_active_years': n_active_years,
    }
    logger.info("Computed performance metrics")
    return metrics

def print_metrics(metrics: Dict):
    """Pretty print metrics."""
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE METRICS")
    print("=" * 60)

    print(f"\nReturns:")
    print(f"  Total Return:        {metrics['total_return']:>10.2%}")
    print(f"  Annualized Return:   {metrics['annualized_return']:>10.2%}")

    print(f"\nRisk:")
    print(f"  Annualized Vol:      {metrics['annualized_volatility']:>10.2%}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")

    print(f"\nRisk-Adjusted:")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

    print(f"\nTrading:")
    print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
    print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
    print(f"  Avg Win (daily):     {metrics['avg_win']:>10.4%}")
    print(f"  Avg Loss (daily):    {metrics['avg_loss']:>10.4%}")
    print(f"  Win/Loss Ratio:      {metrics['win_loss_ratio']:>10.2f}")
    print(f"  Active Trading Days: {metrics['n_active_days']:>10.0f}")
    print(f"  Total Calendar Days: {metrics['n_days_total']:>10.0f}")
    print(f"  Active Years:        {metrics['n_active_years']:>10.1f}")
    print(f"  Calendar Years:      {metrics['n_years']:>10.1f}")

    print("=" * 60 + "\n")