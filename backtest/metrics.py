"""
Backtest performance metrics.

Conventions:
  - Vol / downside vol: computed on FULL return series (all calendar days)
    This is the honest denominator — it reflects real elapsed time including
    idle RF days. Using trading-only vol overstates annualized vol because
    non-consecutive days are treated as consecutive.

  - Sharpe / Sortino / Calmar: reported in two flavours:
      Raw    — annualized_return in numerator  (no RF subtraction)
      RF-Adj — excess_return = annualized_return - RISK_FREE_RATE

  - Smart Sharpe: uses trading-only skew/kurtosis (strategy quality signal,
    not RF-day distribution artifact), applied to RF-adj Sharpe.

  - Distribution stats: reported for both full series and trading days only.

  - Trading statistics (win rate, profit factor, etc.): trading days ONLY.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict
from loguru import logger
from config.settings import RISK_FREE_RATE


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_metrics(results: pd.DataFrame) -> Dict:
    # --- Guard rails ---
    if results.empty or 'portfolio_return' not in results.columns:
        logger.warning("Empty results or missing portfolio_return column")
        return {}

    returns = results['portfolio_return'].dropna()
    if len(returns) == 0:
        logger.warning("No valid returns")
        return {}

    # --- Split into buckets ---
    if 'is_risk_free' in results.columns:
        is_rf = results.loc[returns.index, 'is_risk_free'].astype(bool)
        trading_returns = returns[(returns.abs() > 1e-8) & (~is_rf)]
        rf_returns      = returns[is_rf]
    else:
        trading_returns = returns[returns.abs() > 1e-8]
        rf_returns      = pd.Series([], dtype=float)

    active_returns = returns[returns.abs() > 1e-8]

    if len(active_returns) == 0:
        logger.warning("No active trading days found")
        return {}
    if len(trading_returns) == 0:
        logger.warning("No pure trading days found — check is_risk_free column")
        return {}

    # --- Core returns ---
    total_return      = (results['portfolio_value'].iloc[-1] /
                         results['portfolio_value'].iloc[0]) - 1
    n_years           = (results['date'].iloc[-1] -
                         results['date'].iloc[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1 / n_years) - 1
    excess_return     = annualized_return - RISK_FREE_RATE   # RF-adj numerator

    # --- Volatility ---
    # Full series: honest calendar denominator (includes RF/idle days)
    full_vol     = returns.std() * np.sqrt(252)
    # Trading-only: reported for reference, NOT used in ratio denominators
    # (non-consecutive days artificially inflate annualized vol)
    trading_vol  = trading_returns.std() * np.sqrt(252)

    # --- Drawdown ---
    pv          = results['portfolio_value']
    running_max = pv.expanding().max()
    drawdown    = (pv - running_max) / running_max
    max_drawdown = drawdown.min()

    # --- Sharpe (full vol denominator — both flavours) ---
    sharpe_raw = annualized_return / full_vol if full_vol > 0 else 0.0
    sharpe_rf  = excess_return     / full_vol if full_vol > 0 else 0.0

    # --- Sortino (downside from full series — consistent denominator) ---
    downside     = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252)
    sortino_raw  = annualized_return / downside_std if downside_std > 0 else 0.0
    sortino_rf   = excess_return     / downside_std if downside_std > 0 else 0.0

    # --- Calmar ---
    calmar_raw = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    calmar_rf  = excess_return     / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # --- Distribution: full series ---
    full_skew = returns.skew()
    full_kurt = returns.kurtosis()

    # --- Distribution: trading days only ---
    trading_skew = trading_returns.skew()
    trading_kurt = trading_returns.kurtosis()

    # --- Smart Sharpe (Pézier & White) ---
    # Uses trading-only distribution + respective Sharpe flavour
    def _smart_sharpe(sr: float, skew: float, kurt: float) -> float:
        adj = 1 - (skew / 6) * sr + (kurt / 24) * sr ** 2
        return sr / np.sqrt(abs(adj)) if adj != 0 else sr

    smart_sharpe_raw = _smart_sharpe(sharpe_raw, trading_skew, trading_kurt)
    smart_sharpe_rf  = _smart_sharpe(sharpe_rf,  trading_skew, trading_kurt)

    # --- Sharpe p-value (Lo 2002) — uses RF-adj Sharpe ---
    n             = len(returns)
    t_stat        = sharpe_rf * np.sqrt(n / 252)
    sharpe_pvalue = 1 - stats.norm.cdf(t_stat)

    # --- Trading statistics (pure trading days ONLY) ---
    win_rate      = (trading_returns > 0).mean()
    gains         = trading_returns[trading_returns > 0].sum()
    losses        = trading_returns[trading_returns < 0].abs().sum()
    profit_factor = gains / losses if losses > 0 else np.inf
    avg_win       = (trading_returns[trading_returns > 0].mean()
                     if (trading_returns > 0).any() else 0.0)
    avg_loss      = (trading_returns[trading_returns < 0].mean()
                     if (trading_returns < 0).any() else 0.0)
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # --- Activity ---
    n_trading_days   = len(trading_returns)
    n_rf_days        = len(rf_returns)
    active_day_pct   = len(active_returns) / len(returns)
    trading_day_pct  = n_trading_days      / len(returns)

    metrics = {
        # Returns
        'total_return':             total_return,
        'annualized_return':        annualized_return,
        'excess_return':            excess_return,

        # Volatility
        'vol_full':                 full_vol,
        'vol_trading':              trading_vol,

        # Sharpe
        'sharpe_raw':               sharpe_raw,
        'sharpe_rf':                sharpe_rf,

        # Smart Sharpe
        'smart_sharpe_raw':         smart_sharpe_raw,
        'smart_sharpe_rf':          smart_sharpe_rf,

        # Sharpe significance
        'sharpe_pvalue':            sharpe_pvalue,

        # Sortino
        'sortino_raw':              sortino_raw,
        'sortino_rf':               sortino_rf,

        # Calmar
        'calmar_raw':               calmar_raw,
        'calmar_rf':                calmar_rf,

        # Drawdown
        'max_drawdown':             max_drawdown,

        # Distribution — full series
        'skewness_full':            full_skew,
        'excess_kurtosis_full':     full_kurt,

        # Distribution — trading days only
        'skewness_trading':         trading_skew,
        'excess_kurtosis_trading':  trading_kurt,

        # Trading stats
        'win_rate':                 win_rate,
        'profit_factor':            profit_factor,
        'avg_win':                  avg_win,
        'avg_loss':                 avg_loss,
        'win_loss_ratio':           win_loss_ratio,

        # Activity
        'n_active_days':            len(active_returns),
        'n_trading_days':           n_trading_days,
        'n_risk_free_days':         n_rf_days,
        'n_days_total':             len(returns),
        'n_years':                  n_years,
        'active_day_pct':           active_day_pct,
        'trading_day_pct':          trading_day_pct,
    }

    logger.info(
        f"Metrics computed | Trading days: {n_trading_days} | "
        f"RF days: {n_rf_days} | "
        f"Win rate (trading only): {win_rate:.1%}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_metrics(metrics: Dict):
    W = 10  # column width


    def row(label: str, raw, rf, fmt: str = '.2f'):
        raw_str = format(float(raw), fmt)
        rf_str  = format(float(rf),  fmt)
        print(f"  {label:<32} {raw_str:>{W}}   {rf_str:>{W}}")

    def pct_row(label: str, raw, rf):
        raw_str = format(float(raw), '.2%')
        rf_str  = format(float(rf),  '.2%')
        print(f"  {label:<32} {raw_str:>{W}}   {rf_str:>{W}}")

    sep = "=" * 72

    print(f"\n{sep}")
    print("BACKTEST PERFORMANCE METRICS")
    print(sep)
    print(f"  Risk-Free Rate (annual): {RISK_FREE_RATE:.2%}")
    print(f"\n  {'Metric':<32} {'Raw':>{W}}   {'RF-Adj':>{W}}")
    print(f"  {'-'*32} {'-'*W}   {'-'*W}")

    print(f"\nReturns:")
    print(f"  {'Total Return':<32} {metrics['total_return']:>{W}.2%}")
    pct_row('Annualized Return',   metrics['annualized_return'], metrics['excess_return'])

    print(f"\nVolatility:")
    pct_row('Vol (full series)',   metrics['vol_full'],          metrics['vol_full'])
    pct_row('Vol (trading only)',  metrics['vol_trading'],       metrics['vol_trading'])
    print(f"  {'  * trading vol for reference only':<32}")

    print(f"\nSharpe (full vol denominator):")
    row('Sharpe',                  metrics['sharpe_raw'],        metrics['sharpe_rf'])
    row('Smart Sharpe',            metrics['smart_sharpe_raw'],  metrics['smart_sharpe_rf'])
    print(f"  {'Sharpe p-value':<32} {metrics['sharpe_pvalue']:>{W}.4f}")

    print(f"\nSortino / Calmar:")
    row('Sortino',                 metrics['sortino_raw'],       metrics['sortino_rf'])
    row('Calmar',                  metrics['calmar_raw'],        metrics['calmar_rf'])
    print(f"  {'Max Drawdown':<32} {metrics['max_drawdown']:>{W}.2%}")

    print(f"\nDistribution (full series):")
    print(f"  {'Skewness':<32} {metrics['skewness_full']:>{W}.3f}")
    print(f"  {'Excess Kurtosis':<32} {metrics['excess_kurtosis_full']:>{W}.3f}")

    print(f"\nDistribution (trading days only):")
    print(f"  {'Skewness':<32} {metrics['skewness_trading']:>{W}.3f}")
    print(f"  {'Excess Kurtosis':<32} {metrics['excess_kurtosis_trading']:>{W}.3f}")

    print(f"\nTrading (pure trade days only):")
    print(f"  {'Win Rate':<32} {metrics['win_rate']:>{W}.2%}")
    print(f"  {'Profit Factor':<32} {metrics['profit_factor']:>{W}.2f}")
    print(f"  {'Avg Win (daily)':<32} {metrics['avg_win']:>{W}.4%}")
    print(f"  {'Avg Loss (daily)':<32} {metrics['avg_loss']:>{W}.4%}")
    print(f"  {'Win/Loss Ratio':<32} {metrics['win_loss_ratio']:>{W}.2f}")

    print(f"\nActivity:")
    print(f"  {'Active Day %':<32} {metrics['active_day_pct']:>{W}.1%}")
    print(f"  {'Trading Day %':<32} {metrics['trading_day_pct']:>{W}.1%}")
    print(f"  {'Active Days':<32} {metrics['n_active_days']:>{W}.0f}")
    print(f"  {'Trading Days':<32} {metrics['n_trading_days']:>{W}.0f}")
    print(f"  {'Risk-Free Days':<32} {metrics['n_risk_free_days']:>{W}.0f}")
    print(f"  {'Total Calendar Days':<32} {metrics['n_days_total']:>{W}.0f}")
    print(f"  {'Calendar Years':<32} {metrics['n_years']:>{W}.1f}")
    print(f"{sep}\n")