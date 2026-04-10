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
    Build a daily returns Series restricted to active trading days.

    A day is considered "active" if any {commodity}_position column is
    non-zero on that date. Idle (off-season / flat) days are dropped
    entirely so QuantStats computes Sharpe, CAGR, vol, etc. over the
    periods the strategy is actually taking risk — matching the
    per-commodity metrics printed to the terminal.

    Args:
        results: Full (non-compacted) backtest results from engine [2]

    Returns:
        pd.Series of daily returns indexed by real datetime dates,
        containing only dates where the strategy held a position.
    """
    df = results.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Identify active days via any non-zero position column
    position_cols = [c for c in df.columns if c.endswith('_position')]
    if position_cols:
        active_mask = (df[position_cols].fillna(0).abs().sum(axis=1) > 0)
        df = df[active_mask]

    returns = df.set_index('date')['portfolio_return']

    # QuantStats chokes on NaN
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
    subtitle_parts.append("Active trading days only (idle days excluded)")
    subtitle = " | ".join(subtitle_parts)

    output_path = str(output_path)

    # Pass a clean title to QuantStats — the subtitle is injected in post-processing
    # as a separate element because mixing <br><small> into the title collides with
    # QuantStats's own "(Compounded) <dt>dates</dt>" suffix and renders awkwardly.
    try:
        if benchmark_ticker:
            logger.info(f"  Fetching benchmark: {benchmark_ticker}")
            qs.reports.html(
                returns,
                benchmark=benchmark_ticker,
                output=output_path,
                title=title,
                download_filename=output_path
            )
        else:
            qs.reports.html(
                returns,
                output=output_path,
                title=title,
                download_filename=output_path
            )

        _clean_report_html(output_path, subtitle=subtitle)
        logger.info(f"✓ QuantStats report saved to {output_path}")

    except Exception as e:
        logger.error(f"QuantStats report generation failed: {e}")
        raise


def _clean_report_html(output_path: str, subtitle: Optional[str] = None):
    """
    Post-process the QuantStats-generated HTML to fix formatting issues:

    1. Remove the bottom-of-body global CSS override that breaks text wrapping.
    2. Re-add the "%" suffix to the EOY Returns "Cumulative" column
       (QuantStats's CSS rule strips it because it targets :last-of-type).
    3. Inject a clean subtitle as a dedicated element under the <h1>,
       instead of mangling the title string.
    4. Tighten header spacing.
    """
    path = Path(output_path)
    content = path.read_text()

    # 1. Strip the weird global white-space override QuantStats appends at </body>
    content = content.replace(
        '<style>*{white-space:auto !important;}</style>',
        ''
    )

    # 2. + 4. Inject our CSS overrides into <head>
    extra_css = """
    <style>
    /* Re-add % suffix to the Cumulative column in EOY Returns
       (QuantStats CSS empties :last-of-type which is the Cumulative col) */
    #eoy table td:last-of-type:after { content: "%" !important; }
    /* Clean subtitle element styling */
    .qs-subtitle { color: #666; font-size: 13px; margin: 6px 0 4px 0;
                   font-weight: 400; }
    /* Give the header a bit more breathing room */
    h1 { line-height: 1.3; }
    h1 dt { color: #666; }
    </style>
    """
    content = content.replace('</head>', extra_css + '</head>', 1)

    # 3. Inject subtitle as a sibling paragraph right after the <h1>...</h1>
    if subtitle:
        content = content.replace(
            '</h1>',
            f'</h1>\n        <p class="qs-subtitle">{subtitle}</p>',
            1
        )

    path.write_text(content)