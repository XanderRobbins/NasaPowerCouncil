"""
Seasonal anomalies: how unusual is today's weather *for this calendar day*?

    z = (x - mean of the same calendar day in earlier years) / std of the same

The baseline for a given day only ever contains observations from strictly earlier years
(an expanding mean/std shifted by one year), so the value computed for day ``t`` can never
depend on data after ``t`` -- no look-ahead.

Warm-up: until a calendar day has two earlier years the baseline std is undefined and the
z-score is NaN. NaN is deliberately left as is: downstream code treats it as "no anomaly".
(An earlier version filled these with the mean and std of the *whole* sample, which leaked
future information into the first years.)
"""
import pandas as pd
from loguru import logger

# Weather variables that get a ``<name>_z`` anomaly column.
ANOMALY_VARIABLES = ['temp_avg', 'temp_max', 'temp_min', 'precipitation']


def compute_seasonal_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` (sorted by date) with a z-score column for each anomaly variable."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Group rows by calendar day (e.g. every 06-15) to build each day's multi-year baseline.
    month_day = df['date'].dt.strftime('%m-%d')

    for var in ANOMALY_VARIABLES:
        if var not in df.columns:
            logger.warning(f"Variable {var} not found in DataFrame, skipping")
            continue

        grouped = df.groupby(month_day, sort=False)[var]
        seasonal_mean = grouped.transform(lambda x: x.expanding().mean().shift(1))
        seasonal_std = grouped.transform(lambda x: x.expanding().std().shift(1))

        # The 1e-6 keeps the division finite when a baseline has zero variance.
        df[f'{var}_z'] = (df[var] - seasonal_mean) / (seasonal_std + 1e-6)

    logger.debug(f"Computed seasonal deviations for {len(ANOMALY_VARIABLES)} variables")
    return df
