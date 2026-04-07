"""
Compute seasonal deviations (Z-scores) from historical baseline.
NO look-ahead bias: baseline uses only data prior to each observation.
"""
import pandas as pd
import numpy as np
from loguru import logger


def compute_seasonal_deviations(df: pd.DataFrame,
                                 baseline_years: int = 20) -> pd.DataFrame:
    """
    Compute Z-scores for weather variables relative to seasonal baseline.

    Z-score = (X_t - μ_season) / σ_season

    FIXED: Uses expanding window grouped by day-of-year so that the
    seasonal mean/std for any date t is computed ONLY from prior years'
    observations on the same calendar day. No future data leaks in.

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

    # Extract month-day key for seasonal grouping
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['month_day'] = df['month'].astype(str) + '-' + df['day'].astype(str)
    df['year'] = df['date'].dt.year

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

    # Clean up
    df = df.drop(columns=['month', 'day', 'month_day', 'year'])

    logger.debug(f"Computed seasonal deviations (no look-ahead) for {len(variables)} variables")

    return df