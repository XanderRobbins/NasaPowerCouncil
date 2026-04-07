"""
Calculate backtest performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
from config.settings import GROWING_SEASON_MONTHS


<<<<<<< HEAD
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
=======
def compute_metrics(results: pd.DataFrame) -> Dict:
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
    if results.empty or 'portfolio_return' not in results.columns:
        logger.warning("Empty results or missing portfolio_return column")
        return {}

<<<<<<< HEAD
    # Filter to trade months if commodity specified
    results_filtered = results.copy()
    if commodity and trade_months:
        results_filtered = results[results['date'].dt.month.isin(trade_months)].copy()
        if results_filtered.empty:
            logger.warning(f"No results for {commodity} during trade months {trade_months}")
            return {}

    returns = results_filtered['portfolio_return'].dropna()

=======
    returns = results['portfolio_return'].dropna()
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
    if len(returns) == 0:
        logger.warning("No valid returns")
        return {}

<<<<<<< HEAD
    # Filter to only active trading days (non-zero returns)
    active_returns = returns[returns != 0]

=======
    active_returns = returns[returns.abs() > 1e-8]
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
    if len(active_returns) == 0:
        logger.warning("No active trading days found")
        return {}

<<<<<<< HEAD
    # Calculate total return from portfolio value (this captures the actual gains/losses)
    total_return = (results_filtered['portfolio_value'].iloc[-1] / results_filtered['portfolio_value'].iloc[0]) - 1

    # Annualize over filtered trading days
    n_active_years = len(results_filtered) / 252

    annualized_return = (1 + total_return) ** (1 / n_active_years) - 1 if n_active_years > 0 else 0

    # Volatility — computed on active returns only (excludes flat periods)
    daily_vol = active_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)
=======
    total_return = (results['portfolio_value'].iloc[-1] / results['portfolio_value'].iloc[0]) - 1
    n_years = (results['date'].iloc[-1] - results['date'].iloc[0]).days / 365.25
    n_active_years = len(active_returns) / 252

    active_vol = active_returns.std() * np.sqrt(252)
    active_annualized_return = (1 + total_return) ** (1 / n_active_years) - 1
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b

    sharpe = active_annualized_return / active_vol if active_vol > 0 else 0

<<<<<<< HEAD
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
=======
    portfolio_values = results['portfolio_value']
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = (active_returns > 0).mean()
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
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
<<<<<<< HEAD
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
=======
        'n_days_total': len(returns),
        'n_years': n_years,
        'n_active_years': n_active_years,
    }

    logger.info("Computed performance metrics")
    return metrics


def compute_commodity_attribution(results: pd.DataFrame,
                                   commodities: List[str]) -> Dict[str, Dict]:
    """
    Compute per-commodity performance attribution.

    Strategy: isolate each commodity's contribution by computing
    the P&L it would have generated if it were the only position,
    using its position size and the portfolio's daily return series.

    For each commodity we have:
      - {commodity}_position: position size on that day
      - portfolio_return: total portfolio return

    We approximate each commodity's daily P&L contribution as:
        commodity_pnl_t = position_t * (price_t - price_{t-1}) / price_{t-1}

    Since we don't store per-commodity prices in results, we back out
    the contribution using the signal column as a proxy for activity,
    and isolate active days per commodity from the position column.
    """
    attribution = {}

    for commodity in commodities:
        pos_col = f'{commodity}_position'
        sig_col = f'{commodity}_signal'

        if pos_col not in results.columns:
            logger.warning(f"No position column for {commodity} — skipping attribution")
            continue

        # Active days: this commodity had a non-zero position
        active_mask = results[pos_col].abs() > 1e-6
        active_days = results[active_mask].copy()

        if len(active_days) == 0:
            attribution[commodity] = {'active_days': 0}
            continue

        # On active days, estimate this commodity's return contribution.
        # In magnitude mode [2], position sizing is:
        #   position = signal * (TARGET_VOL / commodity_vol)
        # With multiple commodities, each commodity's contribution to
        # portfolio_return is approximately:
        #   contrib_t ≈ (position_c / sum_of_all_positions) * portfolio_return_t
        #
        # We use position weight share as the attribution proxy.
        position_cols = [f'{c}_position' for c in commodities
                         if f'{c}_position' in results.columns]

        total_abs_position = results[position_cols].abs().sum(axis=1).replace(0, np.nan)
        commodity_weight = results[pos_col].abs() / total_abs_position

        # Attributed daily return = weight share * portfolio return
        attributed_returns = (commodity_weight * results['portfolio_return']).fillna(0)
        active_attributed = attributed_returns[active_mask]

        n_active = len(active_days)
        n_active_years = n_active / 252

        # Cumulative attributed return
        attributed_total = (1 + active_attributed).prod() - 1

        if n_active_years > 0:
            attributed_ann = (1 + attributed_total) ** (1 / n_active_years) - 1
        else:
            attributed_ann = 0.0

        attributed_vol = active_attributed.std() * np.sqrt(252)
        attributed_sharpe = attributed_ann / attributed_vol if attributed_vol > 0 else 0

        win_rate = (active_attributed > 0).mean()

        # Average signal strength on active days
        avg_signal = (
            active_days[sig_col].abs().mean()
            if sig_col in results.columns else np.nan
        )

        # Average position size
        avg_position = active_days[pos_col].abs().mean()

        # Contribution to total portfolio return (% of total gains)
        total_portfolio_return = results['portfolio_return'].sum()
        return_contribution_pct = (
            active_attributed.sum() / total_portfolio_return * 100
            if total_portfolio_return != 0 else 0
        )

        attribution[commodity] = {
            'active_days': n_active,
            'active_years': round(n_active_years, 2),
            'attributed_total_return': attributed_total,
            'attributed_annualized_return': attributed_ann,
            'attributed_vol': attributed_vol,
            'attributed_sharpe': attributed_sharpe,
            'win_rate': win_rate,
            'avg_signal_strength': avg_signal,
            'avg_position_size': avg_position,
            'return_contribution_pct': return_contribution_pct,
        }

    logger.info("Computed per-commodity attribution")
    return attribution


def print_metrics(metrics: Dict):
    """Pretty print metrics."""
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
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
<<<<<<< HEAD
    print(f"  Growing Season Yrs:  {metrics['n_active_years']:>10.1f}")
=======
    print(f"  Total Calendar Days: {metrics['n_days_total']:>10.0f}")
    print(f"  Active Years:        {metrics['n_active_years']:>10.1f}")
    print(f"  Calendar Years:      {metrics['n_years']:>10.1f}")
    print("=" * 60 + "\n")
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b


def print_attribution(attribution: Dict[str, Dict]):
    """Pretty print per-commodity attribution."""
    print("\n" + "=" * 60)
    print("PER-COMMODITY ATTRIBUTION")
    print("=" * 60)

    for commodity, stats in attribution.items():
        if stats.get('active_days', 0) == 0:
            print(f"\n  {commodity.upper()}: No active trading days")
            continue

        print(f"\n  {commodity.upper()}")
        print(f"  {'─' * 30}")
        print(f"  Active Days:         {stats['active_days']:>10.0f}")
        print(f"  Active Years:        {stats['active_years']:>10.2f}")
        print(f"  Return Contribution: {stats['return_contribution_pct']:>10.1f}%")
        print(f"  Attributed Ann Ret:  {stats['attributed_annualized_return']:>10.2%}")
        print(f"  Attributed Vol:      {stats['attributed_vol']:>10.2%}")
        print(f"  Attributed Sharpe:   {stats['attributed_sharpe']:>10.2f}")
        print(f"  Win Rate:            {stats['win_rate']:>10.2%}")
        print(f"  Avg Position Size:   {stats['avg_position_size']:>10.3f}")
        print(f"  Avg Signal Strength: {stats['avg_signal_strength']:>10.3f}")

    print("\n" + "=" * 60 + "\n")