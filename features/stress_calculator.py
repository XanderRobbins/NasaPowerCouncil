"""
Compute nonlinear stress transformations.
These capture threshold effects in crop physiology.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger

from config.thresholds import CRITICAL_THRESHOLDS
from config.calendars import get_current_stage, get_sensitivity_weight


class StressCalculator:
    """
    Compute stress metrics from weather variables.
    
    Stress types:
    - Heat stress: max(0, Temp - T_critical)
    - Cold stress: max(0, T_critical - Temp)
    - Dry stress: max(0, Precip_threshold - Precip)
    - Wet stress: max(0, Precip - Precip_threshold)
    - GDD (Growing Degree Days)
    """
    
    def __init__(self, commodity: str):
        self.commodity = commodity
        self.thresholds = CRITICAL_THRESHOLDS[commodity]
        

    def compute_heat_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Heat stress: HS = max(0, Temp_max - T_critical_high)
        """
        df = df.copy()
        
        # Check if threshold exists (not all commodities have this)
        if 'temp_critical_high' not in self.thresholds:
            return df
        
        t_crit = self.thresholds['temp_critical_high']
        df['heat_stress'] = np.maximum(0, df['temp_max'] - t_crit)
        
        # Also compute Z-score version
        if 'temp_max_z' in df.columns:
            df['heat_stress_z'] = np.maximum(0, df['temp_max_z'] - 
                                            (t_crit - df['temp_max'].mean()) / df['temp_max'].std())
        
        return df


    def compute_cold_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cold stress: CS = max(0, T_critical_low - Temp_min)
        """
        df = df.copy()
        
        # Check if threshold exists
        if 'temp_critical_low' not in self.thresholds:
            return df
        
        t_crit = self.thresholds['temp_critical_low']
        df['cold_stress'] = np.maximum(0, t_crit - df['temp_min'])
        
        if 'temp_min_z' in df.columns:
            df['cold_stress_z'] = np.maximum(0, 
                (t_crit - df['temp_min'].mean()) / df['temp_min'].std() - df['temp_min_z'])
        
        return df


    def compute_dry_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dry stress: DS = max(0, Precip_threshold - Precip)
        """
        df = df.copy()
        
        # Check if threshold exists
        if 'precip_threshold_dry' not in self.thresholds:
            return df
        
        precip_threshold = self.thresholds['precip_threshold_dry']
        
        # Convert to weekly if needed (NASA POWER gives daily)
        if 'precipitation' in df.columns:
            df['precip_7d'] = df['precipitation'].rolling(window=7).sum()
            df['dry_stress'] = np.maximum(0, precip_threshold - df['precip_7d'])
        
        return df

    def compute_wet_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wet stress (flooding): WS = max(0, Precip - Precip_threshold_wet)
        """
        df = df.copy()
        
        # Check if threshold exists
        if 'precip_threshold_wet' not in self.thresholds:
            return df
        
        precip_threshold = self.thresholds['precip_threshold_wet']
        
        if 'precipitation' in df.columns:
            df['precip_7d'] = df['precipitation'].rolling(window=7).sum()
            df['wet_stress'] = np.maximum(0, df['precip_7d'] - precip_threshold)
        
        return df



    def compute_gdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Growing Degree Days: GDD = max(0, (T_max + T_min)/2 - T_base)
        """
        df = df.copy()
        
        if 'gdd_base' in self.thresholds:
            t_base = self.thresholds['gdd_base']
            t_optimal = self.thresholds.get('gdd_optimal', 30)
            
            # Average temperature
            df['temp_avg_calc'] = (df['temp_max'] + df['temp_min']) / 2
            
            # Cap at optimal (heat above optimal doesn't help)
            df['temp_capped'] = df['temp_avg_calc'].clip(upper=t_optimal)
            
            # GDD
            df['gdd'] = np.maximum(0, df['temp_capped'] - t_base)
            
            # Cumulative GDD
            df['gdd_cumsum'] = df['gdd'].cumsum()
        
        return df
    
    def compute_frost_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Frost events (critical for coffee).
        """
        df = df.copy()
        
        if 'frost_threshold' in self.thresholds:
            frost_temp = self.thresholds['frost_threshold']
            df['frost_event'] = (df['temp_min'] <= frost_temp).astype(int)
        
        return df
    
    def compute_consecutive_dry_days(self, df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
        """
        Count consecutive days with precip < threshold.
        """
        df = df.copy()
        
        if 'precipitation' in df.columns:
            # Mark dry days
            df['is_dry'] = (df['precipitation'] < threshold).astype(int)
            
            # Count consecutive
            df['consecutive_dry'] = df['is_dry'].groupby(
                (df['is_dry'] != df['is_dry'].shift()).cumsum()
            ).cumsum()
        
        return df
    
    def compute_extreme_events(self, df: pd.DataFrame, percentile: float = 95) -> pd.DataFrame:
        """
        Flag extreme events (95th percentile exceedances).
        """
        df = df.copy()
        
        for var in ['temp_max', 'precipitation', 'wind_speed']:
            if var in df.columns:
                threshold = df[var].quantile(percentile / 100)
                df[f'{var}_extreme'] = (df[var] > threshold).astype(int)
        
        return df
    
    def compute_all_stress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all stress metrics for this commodity.
        """
        df = self.compute_heat_stress(df)
        df = self.compute_cold_stress(df)
        df = self.compute_dry_stress(df)
        df = self.compute_wet_stress(df)
        df = self.compute_gdd(df)
        df = self.compute_frost_events(df)
        df = self.compute_consecutive_dry_days(df)
        df = self.compute_extreme_events(df)
        
        logger.debug(f"Computed all stress metrics for {self.commodity}")
        
        return df
    
    def apply_seasonal_weighting(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """
        Weight stress by seasonal sensitivity.
        Stress during pollination matters more than stress during dormancy.
        """
        df = df.copy()
        
        stress_cols = [col for col in df.columns if 'stress' in col or col == 'gdd']
        
        for idx, row in df.iterrows():
            date = row['date']
            stage = get_current_stage(self.commodity, region, date)
            weight = get_sensitivity_weight(self.commodity, stage)
            
            for col in stress_cols:
                if col in df.columns:
                    df.loc[idx, f'{col}_weighted'] = row[col] * weight
        
        return df


def compute_stress_for_region(df: pd.DataFrame, commodity: str, region: str) -> pd.DataFrame:
    """
    Convenience function to compute all stress metrics for a region.
    """
    calculator = StressCalculator(commodity)
    df = calculator.compute_all_stress(df)
    df = calculator.apply_seasonal_weighting(df, region)
    
    return df