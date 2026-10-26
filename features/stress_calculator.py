"""
Crop-stress indicators and their rolling accumulations, for one region.

Per day:

* ``heat_stress``  1 if max temperature exceeds the crop's heat threshold
* ``cold_stress``  1 if min temperature is below the crop's cold threshold
* ``dry_stress``   1 if rainfall over the trailing 7 days is below the crop's dry threshold
* ``*_stress_z``   the matching seasonal anomaly, one-sided: only hotter-than-normal,
                   colder-than-normal or drier-than-normal counts

Crop damage accumulates (three hot days in a week matter far more than one), so every
indicator is also summed over trailing 7, 14 and 30 day windows (e.g. ``heat_stress_7d``).
Those windowed columns are what the models eventually see.
"""
import pandas as pd
from loguru import logger

from config.thresholds import get_thresholds

# Trailing windows (days) over which each indicator is accumulated.
ROLLING_WINDOWS = (7, 14, 30)


def compute_stress_indicators(df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Add stress indicator columns (and their 7/14/30-day sums) to a region's weather frame."""
    df = df.copy()
    thresholds = get_thresholds(commodity)

    # Heat stress: day hotter than the crop's critical maximum.
    if 'temp_max' in df.columns:
        df['heat_stress'] = (df['temp_max'] > thresholds['temp_critical_high']).astype(float)
        if 'temp_max_z' in df.columns:
            df['heat_stress_z'] = df['temp_max_z'].clip(lower=0)  # only hotter-than-normal

    # Cold stress: day colder than the crop's critical minimum.
    if 'temp_min' in df.columns:
        df['cold_stress'] = (df['temp_min'] < thresholds['temp_critical_low']).astype(float)
        if 'temp_min_z' in df.columns:
            df['cold_stress_z'] = df['temp_min_z'].clip(upper=0).abs()  # only colder-than-normal

    # Dry stress: trailing-week rainfall below the crop's dryness threshold.
    if 'precipitation' in df.columns:
        df['precip_7d'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
        df['dry_stress'] = (df['precip_7d'] < thresholds['precip_threshold_dry']).astype(float)
        if 'precipitation_z' in df.columns:
            df['dry_stress_z'] = df['precipitation_z'].clip(upper=0).abs()  # only drier-than-normal

    # Accumulate every indicator over trailing windows. NaN anomalies (warm-up) count as 0
    # inside the window because min_periods=1 skips them.
    stress_vars = [col for col in df.columns if col.endswith('_stress') or col.endswith('_stress_z')]
    for var in stress_vars:
        for window in ROLLING_WINDOWS:
            df[f'{var}_{window}d'] = df[var].rolling(window=window, min_periods=1).sum()

    logger.debug(f"Computed stress indicators for {commodity}")
    return df
