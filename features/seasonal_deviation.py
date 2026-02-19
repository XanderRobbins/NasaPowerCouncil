"""
Compute seasonal deviations (Z-scores) for weather variables.
This normalizes raw weather data relative to historical baselines.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger

from config.settings import BASELINE_YEARS


class SeasonalDeviationCalculator:
    """
    Compute Z-scores for weather variables relative to day-of-year baselines.
    
    Z_{r,t} = (X_{r,t} - μ_{r,doy}) / σ_{r,doy}
    
    Uses rolling N-year baseline to avoid look-ahead bias.
    """
    
    def __init__(self, baseline_years: int = BASELINE_YEARS):
        self.baseline_years = baseline_years
        self.baselines = {}  # Cache baselines
        
    def compute_baseline(self, df: pd.DataFrame, variable: str) -> pd.DataFrame:
        """
        Compute mean and std for each day-of-year from historical data.
        
        Args:
            df: DataFrame with 'date' and weather variables
            variable: Variable name (e.g., 'temp_max')
            
        Returns:
            DataFrame with doy, mean, std
        """
        df = df.copy()
        df['doy'] = pd.to_datetime(df['date']).dt.dayofyear
        
        # Group by day-of-year and compute statistics
        baseline = df.groupby('doy')[variable].agg(['mean', 'std']).reset_index()
        baseline.columns = ['doy', f'{variable}_mu', f'{variable}_sigma']
        
        # Handle zero std (rare, but possible)
        baseline[f'{variable}_sigma'] = baseline[f'{variable}_sigma'].replace(0, 1e-6)
        
        return baseline
    
    def compute_deviation(self, 
                         df: pd.DataFrame, 
                         variable: str,
                         rolling: bool = True) -> pd.DataFrame:
        """
        Compute Z-score deviations.
        
        Args:
            df: DataFrame with 'date' and weather variables
            variable: Variable name
            rolling: If True, use rolling baseline (no look-ahead)
            
        Returns:
            DataFrame with Z-score column added
        """
        df = df.copy()
        df['doy'] = pd.to_datetime(df['date']).dt.dayofyear
        df['year'] = pd.to_datetime(df['date']).dt.year
        
        if rolling:
            # Rolling baseline: for each date, use only prior years
            z_scores = []
            
            for idx, row in df.iterrows():
                current_year = row['year']
                current_doy = row['doy']
                
                # Get historical data for this DOY (excluding current year)
                historical = df[
                    (df['doy'] == current_doy) & 
                    (df['year'] < current_year) &
                    (df['year'] >= current_year - self.baseline_years)
                ]
                
                if len(historical) == 0:
                    # Not enough history
                    z_scores.append(np.nan)
                    continue
                
                mu = historical[variable].mean()
                sigma = historical[variable].std()
                
                if sigma == 0 or pd.isna(sigma):
                    sigma = 1e-6
                
                z = (row[variable] - mu) / sigma
                z_scores.append(z)
            
            df[f'{variable}_z'] = z_scores
        
        else:
            # Static baseline: use all data (ONLY for backtesting with held-out test set)
            baseline = self.compute_baseline(df, variable)
            df = df.merge(baseline, on='doy', how='left')
            df[f'{variable}_z'] = (df[variable] - df[f'{variable}_mu']) / df[f'{variable}_sigma']
        
        return df
    
    def compute_all_deviations(self, 
                              df: pd.DataFrame, 
                              variables: List[str]) -> pd.DataFrame:
        """Compute Z-scores for multiple variables."""
        for var in variables:
            if var in df.columns:
                df = self.compute_deviation(df, var, rolling=True)
                logger.debug(f"Computed Z-score for {var}")
        
        return df


def compute_seasonal_deviations_for_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to compute all standard deviations for a region.
    """
    calculator = SeasonalDeviationCalculator()
    
    variables = [
        'temp_avg', 'temp_max', 'temp_min',
        'precipitation', 'solar_radiation',
        'wind_speed', 'relative_humidity'
    ]
    
    df = calculator.compute_all_deviations(df, variables)
    
    return df