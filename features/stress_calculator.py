"""
Calculate weather stress indicators (heat, cold, dry).
"""
import pandas as pd
import numpy as np
from loguru import logger
from config.thresholds import get_thresholds

def compute_stress_indicators(df: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """
    Compute stress indicators based on commodity-specific thresholds.
    
    Stress types:
    - Heat stress: Days above critical temperature
    - Cold stress: Days below critical temperature
    - Dry stress: Cumulative precipitation deficit
    
    Args:
        df: DataFrame with temp and precipitation data (and Z-scores)
        commodity: Commodity name
        
    Returns:
        DataFrame with stress indicator columns
    """
    df = df.copy()
    thresholds = get_thresholds(commodity)
    
    # Heat stress (temperature above critical high)
    if 'temp_max' in df.columns:
        df['heat_stress'] = (df['temp_max'] > thresholds['temp_critical_high']).astype(float)
        
        # Use Z-score for normalized heat stress
        if 'temp_max_z' in df.columns:
            df['heat_stress_z'] = df['temp_max_z'].clip(lower=0)  # Only positive deviations
    
    # Cold stress (temperature below critical low)
    if 'temp_min' in df.columns:
        df['cold_stress'] = (df['temp_min'] < thresholds['temp_critical_low']).astype(float)
        
        if 'temp_min_z' in df.columns:
            df['cold_stress_z'] = df['temp_min_z'].clip(upper=0).abs()  # Absolute value of negative deviations
    
    # Dry stress (precipitation below threshold)
    # FIX: Handle both possible threshold key names
    dry_threshold = thresholds.get('precip_threshold_dry', 25)  # Default to 25mm if not found
    
    if 'precipitation' in df.columns:
        # Compute weekly cumulative precipitation
        df['precip_7d'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
        df['dry_stress'] = (df['precip_7d'] < dry_threshold).astype(float)
        
        if 'precipitation_z' in df.columns:
            df['dry_stress_z'] = df['precipitation_z'].clip(upper=0).abs()
    
    # Cumulative stress indicators (7-day, 14-day, 30-day rolling sums)
    stress_vars = [col for col in df.columns if col.endswith('_stress') or col.endswith('_stress_z')]
    
    for var in stress_vars:
        # 7-day cumulative
        df[f'{var}_7d'] = df[var].rolling(window=7, min_periods=1).sum()
        
        # 14-day cumulative
        df[f'{var}_14d'] = df[var].rolling(window=14, min_periods=1).sum()
        
        # 30-day cumulative
        df[f'{var}_30d'] = df[var].rolling(window=30, min_periods=1).sum()
    
    logger.debug(f"Computed stress indicators for {commodity}")
    
    return df

