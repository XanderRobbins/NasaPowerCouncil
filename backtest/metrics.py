"""
Performance metrics for a seasonal strategy.

The strategy sits in cash outside its growing season, so metrics are computed on the
*in-season* rows only, and "years" means in-season days / 252. The definitions below are
deliberately the original ones, so numbers stay comparable between runs:

* annualized_return   CAGR over the in-season years
* sharpe_ratio        annualized return / annualized volatility of the *active-day* returns
                      (days with a non-zero return), 0% risk-free. This divides a compounded
                      return by volatility; a textbook Sharpe divides the mean daily return by
                      its volatility, so the two differ somewhat.
* sortino_ratio       annualized return / annualized std of the negative active-day returns
* max_drawdown        peak-to-trough fall of the equity curve on days when risk was held
* calmar_ratio        annualized return / |max drawdown|

Because capital would be redeployed outside the season, in-season figures are the relevant
ones for a sleeve in a portfolio -- but they are also easy to over-read: with ~10 seasons the
uncertainty around any Sharpe estimate is large.
"""
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(results: pd.DataFrame, commodity: str = None, trade_months: list = None) -> Dict:
    """
    Metrics for the rows of ``results`` that fall in ``trade_months`` (all rows if not given).

    ``commodity`` is only used for log messages. Returns an empty dict when there is nothing
    to measure.
    """
    if results.empty or 'portfolio_return' not in results.columns:
        logger.warning("Empty results or missing portfolio_return column")
        return {}

    frame = results.copy()
    if commodity and trade_months:
        frame = results[results['date'].dt.month.isin(trade_months)].copy()
        if frame.empty:
            logger.warning(f"No results for {commodity} during trade months {trade_months}")
            return {}

    returns = frame['portfolio_return'].dropna()
    if len(returns) == 0:
        logger.warning("No valid returns")
        return {}

    # "Active" days are those with a non-zero return (flat days earn exactly 0).
    active_returns = returns[returns != 0]
    if len(active_returns) == 0:
        logger.warning("No active trading days found")
        return {}

    # Total and annualised return come from the equity curve over the in-season rows.
    total_return = frame['portfolio_value'].iloc[-1] / frame['portfolio_value'].iloc[0] - 1
    n_season_years = len(frame) / TRADING_DAYS_PER_YEAR
    annualized_return = (1 + total_return) ** (1 / n_season_years) - 1 if n_season_years > 0 else 0

    annualized_vol = active_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Drawdown only over days when the strategy held a position or had a non-zero return.
    position_cols = [c for c in frame.columns if c.endswith('_position')]
    risk_on = (frame[position_cols].abs() > 1e-6).any(axis=1) | (frame['portfolio_return'] != 0)
    equity = frame.loc[risk_on, 'portfolio_value'].values
    if len(equity) > 0:
        running_max = np.maximum.accumulate(equity)
        max_drawdown = ((equity - running_max) / running_max).min()
    else:
        max_drawdown = 0.0

    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]

    gains_total, losses_total = wins.sum(), abs(losses.sum())
    profit_factor = gains_total / losses_total if losses_total > 0 else np.inf

    downside_std = losses.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sortino = annualized_return / downside_std if downside_std > 0 else 0
    calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    logger.info(f"Computed performance metrics{f' for {commodity}' if commodity else ''} (in-season rows)")
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': (active_returns > 0).mean(),
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'n_active_days': len(active_returns),
        'n_active_years': n_season_years,
    }


def compute_metrics_by_commodity(results: pd.DataFrame, commodity_trade_months: Dict) -> Dict:
    """Metrics per commodity, each over its own trade months: ``{commodity: metrics dict}``."""
    all_metrics = {}
    for commodity, trade_months in commodity_trade_months.items():
        logger.info(f"Computing metrics for {commodity} (months {trade_months})...")
        all_metrics[commodity] = compute_metrics(results, commodity=commodity, trade_months=trade_months)
    return all_metrics


def print_metrics_by_commodity(metrics_by_commodity: Dict):
    """Print the metrics table for each commodity."""
    print("\n" + "=" * 60)
    print("BACKTEST PERFORMANCE METRICS (By Commodity, in-season rows)")
    print("=" * 60)

    for commodity, m in metrics_by_commodity.items():
        if not m:
            print(f"\n{commodity.upper()}: No data")
            continue

        print(f"\n{commodity.upper()}")
        print("-" * 60)
        print("Returns:")
        print(f"  Total Return:        {m['total_return']:>10.2%}")
        print(f"  Annualized Return:   {m['annualized_return']:>10.2%}")
        print("Risk:")
        print(f"  Annualized Vol:      {m['annualized_volatility']:>10.2%}")
        print(f"  Max Drawdown:        {m['max_drawdown']:>10.2%}")
        print("Risk-Adjusted:")
        print(f"  Sharpe Ratio:        {m['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {m['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {m['calmar_ratio']:>10.2f}")
        print("Trading:")
        print(f"  Win Rate:            {m['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {m['profit_factor']:>10.2f}")
        print(f"  Avg Win (daily):     {m['avg_win']:>10.4%}")
        print(f"  Avg Loss (daily):    {m['avg_loss']:>10.4%}")
        print(f"  Win/Loss Ratio:      {m['win_loss_ratio']:>10.2f}")
        print(f"  Active Trading Days: {m['n_active_days']:>10.0f}")
        print(f"  Growing Season Yrs:  {m['n_active_years']:>10.1f}")

    print("=" * 60 + "\n")
