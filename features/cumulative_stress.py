"""
Compute cumulative stress over rolling windows.
Markets respond to accumulated stress, not single-day events.
"""
import pandas as pd
import numpy as np
from typing import List
from loguru import logger


class CumulativeStressCalculator:
    """
    Compute rolling sums of stress metrics over multiple time windows.
    
    Windows: 7, 14, 30 days (typical)
    """
    
    def __init__(self, windows: List[int] = [7, 14, 30]):
        self.windows = windows
        
    def compute_cumulative(self, df: pd.DataFrame, stress_col: str) -> pd.DataFrame:
        """
        Compute rolling sums for a single stress column.
        
        CStress_w = Σ(t-w to t) Stress_t
        """
        df = df.copy()
        
        for window in self.windows:
            col_name = f'{stress_col}_{window}d'
            df[col_name] = df[stress_col].rolling(window=window, min_periods=1).sum()
        
        return df
    
    def compute_cumulative_z(self, df: pd.DataFrame, stress_col: str) -> pd.DataFrame:
        """
        Compute rolling sums of Z-scored stress.
        """
        z_col = f'{stress_col}_z'
        
        if z_col in df.columns:
            for window in self.windows:
                col_name = f'{z_col}_{window}d'
                df[col_name] = df[z_col].rolling(window=window, min_periods=1).sum()
        
        return df
    
    def compute_all_cumulative(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cumulative stress for all stress columns in the DataFrame.
        """
        df = df.copy()
        
        # Identify stress columns
        stress_cols = [col for col in df.columns if 'stress' in col and '_weighted' in col]
        
        for col in stress_cols:
            df = self.compute_cumulative(df, col)
            df = self.compute_cumulative_z(df, col)
        
        logger.debug(f"Computed cumulative stress for {len(stress_cols)} variables")
        
        return df
    
    def compute_moving_average(self, df: pd.DataFrame, stress_col: str, window: int = 14) -> pd.DataFrame:
        """
        Compute moving average (alternative to sum).
        """
        df = df.copy()
        col_name = f'{stress_col}_ma{window}'
        df[col_name] = df[stress_col].rolling(window=window).mean()
        
        return df
    
    def compute_exponential_weighted(self, df: pd.DataFrame, stress_col: str, span: int = 14) -> pd.DataFrame:
        """
        Compute exponentially weighted stress (recent matters more).
        """
        df = df.copy()
        col_name = f'{stress_col}_ema{span}'
        df[col_name] = df[stress_col].ewm(span=span, adjust=False).mean()
        
        return df


def compute_cumulative_stress_for_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to compute all cumulative metrics for a region.
    """
    calculator = CumulativeStressCalculator()
    df = calculator.compute_all_cumulative(df)
    
    return df