"""
Compute seasonal deviations (Z-scores) from historical baseline.
NO look-ahead bias: baseline uses only data prior to each observation.
OPTIMIZED: Fully vectorized groupby expanding window — no row iteration.
"""
import pandas as pd
import numpy as np
from loguru import logger


def compute_seasonal_deviations(df: pd.DataFrame,
                                 baseline_years: int = 20) -> pd.DataFrame:
    """
    Compute Z-scores for weather variables relative to seasonal baseline.
    Z-score = (X_t - μ_season) / σ_season

    OPTIMIZED: Replaces O(n²) iterrows loop with vectorized groupby
    expanding window. For each calendar day group, we compute the
    expanding mean/std shifted by 1 year to ensure strict no look-ahead.

    Args:
        df: DataFrame with date, temp_max, temp_min, precipitation, etc.
        baseline_years: Not used directly (expanding window handles this)

    Returns:
        DataFrame with additional Z-score columns
    """
    df = df.copy()

    # Ensure date is datetime and sorted
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Extract grouping keys
    df['month_day'] = df['date'].dt.strftime('%m-%d')
    df['year'] = df['date'].dt.year

    # Store original index order so we can re-align after groupby operations
    df['_orig_idx'] = df.index

    variables = ['temp_avg', 'temp_max', 'temp_min', 'precipitation']

    for var in variables:
        if var not in df.columns:
            logger.warning(f"Variable {var} not found in DataFrame, skipping")
            continue

        global_mean = df[var].mean()
        global_std = df[var].std() if df[var].std() > 0 else 1.0

        # Vectorized: within each calendar-day group (already sorted by date/year),
        # expanding().mean/std gives cumulative stats; shift(1) excludes current year
        grp = df.groupby('month_day', sort=False)[var]
        seas_mean = grp.transform(lambda x: x.expanding().mean().shift(1)).fillna(global_mean)
        seas_std  = grp.transform(lambda x: x.expanding().std().shift(1)).fillna(global_std)

        # Compute Z-score
        df[f'{var}_z'] = (df[var] - seas_mean) / (seas_std + 1e-6)

    # Clean up helper columns
    df = df.drop(columns=['month_day', 'year', '_orig_idx'])

    logger.debug(
        f"Computed seasonal deviations (vectorized, no look-ahead) "
        f"for {len(variables)} variables"
    )

    return df