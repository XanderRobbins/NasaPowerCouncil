"""
Active-time compaction: strip idle (flat) days and stitch
only active trading periods into a continuous sequence.

A day is "active" if at least one commodity has a non-zero position
OR a non-zero daily PnL. This removes the long calendar gaps between
growing seasons and produces clean continuous equity/metric charts.
"""
import pandas as pd
import numpy as np
from typing import List


def compact_results(results: pd.DataFrame, commodities: List[str]) -> pd.DataFrame:
    """
    Strip idle rows (no position, no PnL) and return a compacted DataFrame
    with a synthetic 'trading_day' index column replacing real calendar dates.

    Args:
        results:     Full backtest results DataFrame from engine [2]
        commodities: List of commodity names (e.g. ['corn', 'cocoa', 'soybeans'])

    Returns:
        Compacted DataFrame where each row is an active trading day,
        indexed 0..N continuously. Real dates are preserved in 'real_date'.
        The 'date' column is replaced with the synthetic integer index
        so all downstream consumers (metrics, visualizer) work unchanged.
    """
    df = results.copy()

    # --- Identify active rows ---
    position_cols = [f'{c}_position' for c in commodities if f'{c}_position' in df.columns]

    has_position = (
        df[position_cols].abs().max(axis=1) > 1e-6
        if position_cols else pd.Series(False, index=df.index)
    )
    has_pnl = df['daily_pnl'].abs() > 1e-8

    active_mask = has_position | has_pnl
    compacted = df[active_mask].copy().reset_index(drop=True)

    if compacted.empty:
        return compacted

    # --- Preserve real dates, replace 'date' with synthetic trading day ---
    compacted['real_date'] = compacted['date']
    compacted['date'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(
        compacted.index, unit='D'
    )

    # --- Recompute portfolio_return from compacted PnL sequence ---
    # The original portfolio_return is still valid per-row; no recalculation needed.
    # But portfolio_value needs to be re-chained from initial value so the equity
    # curve reflects only active-day compounding (no flat lines between seasons).
    initial_value = compacted['portfolio_value'].iloc[0]

    # Rechain: each row's portfolio_value is previous * (1 + active_return)
    # We use the original daily_pnl but re-anchor to avoid the flat stretches
    # inflating/deflating the curve due to calendar gaps.
    rechained = [initial_value]
    for i in range(1, len(compacted)):
        prev = rechained[-1]
        pnl_return = compacted['portfolio_return'].iloc[i]
        rechained.append(prev * (1 + pnl_return))

    compacted['portfolio_value'] = rechained

    return compacted