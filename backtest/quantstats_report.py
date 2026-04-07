# backtest/quantstats_report.py
"""
Generate QuantStats HTML tearsheet from backtest results.
Uses full calendar results with idle days zeroed out — honest
representation of a seasonal strategy that sits in cash off-season.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from loguru import logger

try:
    import quantstats as qs
except ImportError:
    raise ImportError("Run: pip install quantstats")


def build_returns_series(results: pd.DataFrame) -> pd.Series:
    """
    Build a daily returns Series with real calendar dates.

    Uses full (non-compacted) results so QuantStats sees a proper
    calendar timeline. Idle (off-season) days are already 0.0 in
    portfolio_return from the engine [2], so no manipulation needed —
    we just extract, set the index to real dates, and hand it over.

    Args:
        results: Full (non-compacted) backtest results from engine [2]

    Returns:
        pd.Series of daily returns indexed by real datetime dates
    """
    returns = results.set_index('date')['portfolio_return'].copy()
    returns.index = pd.to_datetime(returns.index)

    # QuantStats chokes on NaN — fill with 0 (idle day = 0% return)
    returns = returns.fillna(0.0)

    # Drop any duplicate dates (shouldn't happen but defensive)
    returns = returns[~returns.index.duplicated(keep='first')]

    return returns


def generate_quantstats_report(
    results: pd.DataFrame,
    output_path: str,
    benchmark_ticker: Optional[str] = None,
    title: str = "NasaPowerCouncil — Weather-Based Commodity Strategy",
    commodities: Optional[List[str]] = None
):
    """
    Generate a full QuantStats HTML tearsheet.

    Args:
        results:          Full (non-compacted) backtest results DataFrame [2]
        output_path:      Path to save the HTML file
        benchmark_ticker: Optional ticker for benchmark comparison (e.g. 'DJP' for Bloomberg
                          Commodity Index ETF, or None to skip benchmark)
        title:            Report title
        commodities:      List of commodity names for subtitle annotation
    """
    logger.info("Generating QuantStats tearsheet...")

    # Build returns series
    returns = build_returns_series(results)

    logger.info(f"  Returns series: {len(returns)} days, "
                f"{returns.index[0].date()} → {returns.index[-1].date()}")
    logger.info(f"  Non-zero days: {(returns != 0).sum()} "
                f"({(returns != 0).mean():.1%} of calendar days)")

    # Extend QuantStats pandas methods
    qs.extend_pandas()

    # Build subtitle
    subtitle_parts = []
    if commodities:
        subtitle_parts.append(f"Commodities: {', '.join(c.capitalize() for c in commodities)}")
    subtitle_parts.append("Idle (off-season) days shown as 0% return")
    subtitle = " | ".join(subtitle_parts)

    output_path = str(output_path)

    try:
        if benchmark_ticker:
            logger.info(f"  Fetching benchmark: {benchmark_ticker}")
            qs.reports.html(
                returns,
                benchmark=benchmark_ticker,
                output=output_path,
                title=f"{title}<br><small>{subtitle}</small>",
                download_filename=output_path
            )
        else:
            qs.reports.html(
                returns,
                output=output_path,
                title=f"{title}<br><small>{subtitle}</small>",
                download_filename=output_path
            )

        logger.info(f"✓ QuantStats report saved to {output_path}")

    except Exception as e:
        logger.error(f"QuantStats report generation failed: {e}")
        raise