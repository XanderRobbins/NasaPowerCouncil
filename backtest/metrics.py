"""
Calculate backtest performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger


def compute_metrics(results: pd.DataFrame) -> Dict:
    """
    Compute comprehensive backtest metrics.

    Args:
        results: Backtest results DataFrame

    Returns:
        Dict of metrics
    """
    if results.empty or 'portfolio_return' not in results.columns:
        logger.warning("Empty results or missing portfolio_return column")
        return {}

    returns = results['portfolio_return'].dropna()

    if len(returns) == 0:
        logger.warning("No valid returns")
        return {}

    # FIX: Separate active days (position held) from zero/flat days
    active_returns = returns[returns != 0]

    if len(active_returns) == 0:
        logger.warning("No active trading days found")
        return {}

    # Basic metrics
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    n_years = len(returns) / 252  # Calendar years (for annualization)
    n_active_years = len(active_returns) / 252  # Active trading years
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Volatility — computed on ALL returns (zeros reduce vol correctly)
    daily_vol = returns.std()
    annualized_vol = daily_vol * np.sqrt(252)

    # Sharpe ratio (assume 0% risk-free rate)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Drawdown — computed on full equity curve
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # FIX: Win rate — only count days where a position was actually held
    win_rate = (active_returns > 0).mean()

    # Profit factor — on active days only
    gains = active_returns[active_returns > 0].sum()
    losses = abs(active_returns[active_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    # Sortino ratio — downside deviation on active days only
    downside_returns = active_returns[active_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = annualized_return / downside_std if downside_std > 0 else 0

    # Calmar ratio
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Average win / average loss
    avg_win = active_returns[active_returns > 0].mean() if len(active_returns[active_returns > 0]) > 0 else 0
    avg_loss = active_returns[active_returns < 0].mean() if len(active_returns[active_returns < 0]) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'n_active_days': len(active_returns),   # FIX: actual trading days
        'n_days_total': len(returns),            # Calendar days in backtest
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