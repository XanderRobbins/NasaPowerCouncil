"""
Compute seasonal deviations (Z-scores) from historical baseline.
"""
import pandas as pd
import numpy as np
from loguru import logger

def compute_seasonal_deviations(df: pd.DataFrame, 
                               baseline_years: int = 20) -> pd.DataFrame:
    """
    Compute Z-scores for weather variables relative to seasonal baseline.
    
    Z-score = (X_t - μ_season) / σ_season
    
    This captures "how unusual is today's weather relative to normal for this time of year?"
    
    Args:
        df: DataFrame with date, temp_max, temp_min, precipitation, etc.
        baseline_years: Number of years for baseline calculation
        
    Returns:
        DataFrame with additional Z-score columns
    """
    df = df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract month and day for seasonal grouping
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['month_day'] = df['month'].astype(str) + '-' + df['day'].astype(str)
    
    # Variables to compute Z-scores for
    variables = ['temp_avg', 'temp_max', 'temp_min', 'precipitation']
    
    for var in variables:
        if var not in df.columns:
            logger.warning(f"Variable {var} not found in DataFrame, skipping")
            continue
        
        # Compute seasonal baseline (mean and std for each day-of-year)
        seasonal_baseline = df.groupby('month_day')[var].agg(['mean', 'std']).reset_index()
        seasonal_baseline.columns = ['month_day', f'{var}_seasonal_mean', f'{var}_seasonal_std']
        
        # Merge back
        df = df.merge(seasonal_baseline, on='month_day', how='left')
        
        # Compute Z-score
        df[f'{var}_z'] = (df[var] - df[f'{var}_seasonal_mean']) / (df[f'{var}_seasonal_std'] + 1e-6)
        
        # Clean up intermediate columns
        df = df.drop(columns=[f'{var}_seasonal_mean', f'{var}_seasonal_std'])
    
    # Clean up
    df = df.drop(columns=['month_day'])
    
    logger.debug(f"Computed seasonal deviations for {len(variables)} variables")
    
    return df