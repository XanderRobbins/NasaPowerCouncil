"""
Aggregate regional features into commodity-level signals using production weights.
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger
from config.regions import get_commodity_regions

def aggregate_regional_features(commodity: str, 
                                region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate regional features into commodity-level using production weights.
    
    Weighted average: F_commodity = Σ(w_i * F_region_i)
    
    Args:
        commodity: Commodity name
        region_dfs: Dict mapping region names to processed feature DataFrames
        
    Returns:
        Aggregated DataFrame with commodity-level features
    """
    regions = get_commodity_regions(commodity)
    
    # Get all dates (union of all regional dates)
    all_dates = pd.concat([df[['date']] for df in region_dfs.values()]).drop_duplicates().sort_values('date')
    
    # Initialize aggregated DataFrame
    aggregated = all_dates.copy()
    
    # Get stress feature columns (only aggregate stress indicators, not raw weather)
    stress_cols = []
    for df in region_dfs.values():
        stress_cols.extend([col for col in df.columns 
                          if ('stress' in col or 'gdd' in col) and col not in stress_cols])
    
    # Remove non-aggregatable columns
    stress_cols = [col for col in stress_cols if col not in ['date', 'month', 'day', 'region']]
    
    logger.info(f"Aggregating {len(stress_cols)} stress features across {len(region_dfs)} regions")
    
    # Aggregate each feature
    for col in stress_cols:
        aggregated[f'{col}_agg'] = 0.0
        
        for region_name, df in region_dfs.items():
            if col not in df.columns:
                continue
            
            weight = regions[region_name]['weight']
            
            # Merge regional data
            regional_data = df[['date', col]].rename(columns={col: f'{col}_temp'})
            aggregated = aggregated.merge(regional_data, on='date', how='left')
            
            # Add weighted contribution
            aggregated[f'{col}_agg'] += aggregated[f'{col}_temp'].fillna(0) * weight
            
            # Clean up temp column
            aggregated = aggregated.drop(columns=[f'{col}_temp'])
    
    # Forward fill missing values (FIXED for pandas 2.0+)
    for col in aggregated.columns:
        if col != 'date':
            aggregated[col] = aggregated[col].ffill().fillna(0)  # Changed from fillna(method='ffill')
    
    logger.info(f"Aggregated features shape: {aggregated.shape}")
    
    return aggregated