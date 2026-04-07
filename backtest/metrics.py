"""
Calculate backtest performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
from config.settings import GROWING_SEASON_MONTHS


def compute_metrics(results: pd.DataFrame, commodity: str = None, trade_months: list = None) -> Dict:
    """
    Compute comprehensive backtest metrics.

    Only includes active trading periods (non-zero return days).
    All metrics are calculated and annualized based on ONLY trading days.

    Args:
        results: Backtest results DataFrame
        commodity: Optional commodity name to filter by trade months
        trade_months: Optional list of trade months (1-12) for the commodity

    Returns:
        Dict of metrics
    """
    if results.empty or 'portfolio_return' not in results.columns:
        logger.warning("Empty results or missing portfolio_return column")
        return {}

    # Filter to trade months if commodity specified
    results_filtered = results.copy()
    if commodity and trade_months:
        results_filtered = results[results['date'].dt.month.isin(trade_months)].copy()
        if results_filtered.empty:
            logger.warning(f"No results for {commodity} during trade months {trade_months}")
            return {}

    returns = results_filtered['portfolio_return'].dropna()

    if len(returns) == 0:
        logger.warning("No valid returns")
        return {}

    # Filter to only active trading days (non-zero returns)
    active_returns = returns[returns != 0]

    if len(active_returns) == 0:
        logger.warning("No active trading days found")
        return {}

    # Calculate total return from portfolio value (this captures the actual gains/losses)
    total_return = (results_filtered['portfolio_value'].iloc[-1] / results_filtered['portfolio_value'].iloc[0]) - 1

    # Annualize over filtered trading days
    n_active_years = len(results_filtered) / 252

    annualized_return = (1 + total_return) ** (1 / n_active_years) - 1 if n_active_years > 0 else 0

    # Volatility — computed on active returns only (excludes flat periods)
    daily_vol = active_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)

    # Sharpe ratio (assume 0% risk-free rate)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Drawdown — computed on active trading periods only
    # Exclude flat periods (no positions, no returns) to measure risk while actually invested
    # A position is "active" if any commodity has a non-zero position on that day OR if there's a return
    position_cols = [c for c in results_filtered.columns if c.endswith('_position')]
    any_position = (results_filtered[position_cols].abs() > 1e-6).any(axis=1) | (results_filtered['portfolio_return'] != 0)

    active_results = results_filtered[any_position].copy()

    if len(active_results) > 0:
        portfolio_values = active_results['portfolio_value'].values
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        max_drawdown = 0.0

    # Win rate
    win_rate = (active_returns > 0).mean()

    # Profit factor
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
        'n_active_days': len(active_returns),
        'n_active_years': n_active_years,
    }

    if commodity:
        logger.info(f"Computed performance metrics for {commodity} (active trading days only)")
    else:
        logger.info("Computed performance metrics (active trading days only)")

    return metrics


def compute_metrics_by_commodity(results: pd.DataFrame, commodity_trade_months: Dict) -> Dict:
    """
    Compute metrics for each commodity during its designated trade months.

    Args:
        results: Backtest results DataFrame
        commodity_trade_months: Dict mapping commodity names to lists of trade months

    Returns:
        Dict with structure: {'commodity_name': {metrics}}
    """
    all_metrics = {}

    for commodity, trade_months in commodity_trade_months.items():
        logger.info(f"\nComputing metrics for {commodity} (months {trade_months})...")
        metrics = compute_metrics(results, commodity=commodity, trade_months=trade_months)
        all_metrics[commodity] = metrics

    return all_metrics


def print_metrics_by_commodity(metrics_by_commodity: Dict):
    """Pretty print per-commodity metrics."""
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE METRICS (By Commodity)")
    print("=" * 60)

    for commodity, metrics in metrics_by_commodity.items():
        if not metrics:
            print(f"\n{commodity.upper()}: No data")
            continue

        print(f"\n{commodity.upper()}")
        print(f"-" * 60)

        print(f"Returns:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['annualized_return']:>10.2%}")

        print(f"Risk:")
        print(f"  Annualized Vol:      {metrics['annualized_volatility']:>10.2%}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")

        print(f"Risk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")

        print(f"Trading:")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"  Avg Win (daily):     {metrics['avg_win']:>10.4%}")
        print(f"  Avg Loss (daily):    {metrics['avg_loss']:>10.4%}")
        print(f"  Win/Loss Ratio:      {metrics['win_loss_ratio']:>10.2f}")
        print(f"  Active Trading Days: {metrics['n_active_days']:>10.0f}")
        print(f"  Growing Season Yrs:  {metrics['n_active_years']:>10.1f}")

    print("=" * 60 + "\n")


def print_metrics(metrics: Dict):
    """Pretty print metrics (active trading periods only)."""
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE METRICS (Active Trading Days Only)")
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
    print(f"  Growing Season Yrs:  {metrics['n_active_years']:>10.1f}")

    print("=" * 60 + "\n")