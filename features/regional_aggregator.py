"""
Aggregate regional stress into commodity-level signals.
Production-weighted aggregation.
"""
import pandas as pd
import numpy as np
from typing import Dict
from loguru import logger

from config.regions import COMMODITY_REGIONS


class RegionalAggregator:
    """
    Aggregate stress metrics across regions using production weights.
    
    Signal_commodity,t = Σ_r (w_r × CStress_r,t)
    """
    
    def __init__(self, commodity: str):
        self.commodity = commodity
        self.regions = COMMODITY_REGIONS[commodity]
        self.weights = {region: info['weight'] for region, info in self.regions.items()}
        
        # Validate weights sum to 1
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights for {commodity} sum to {total_weight}, normalizing...")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def aggregate_feature(self, 
                         region_dfs: Dict[str, pd.DataFrame], 
                         feature_col: str) -> pd.DataFrame:
        """
        Aggregate a single feature across regions.
        
        Args:
            region_dfs: Dict mapping region names to DataFrames
            feature_col: Feature column to aggregate
            
        Returns:
            DataFrame with aggregated feature
        """
        # Ensure all regions have the same date range
        all_dates = None
        
        for region, df in region_dfs.items():
            if all_dates is None:
                all_dates = pd.to_datetime(df['date'])
            else:
                # Ensure consistency
                assert len(df) == len(all_dates), f"Region {region} has different length"
        
        # Create result DataFrame
        result = pd.DataFrame({'date': all_dates})
        
        # Weighted sum
        weighted_values = None
        
        for region, df in region_dfs.items():
            weight = self.weights[region]
            
            if feature_col not in df.columns:
                logger.warning(f"Feature {feature_col} not found in {region}, skipping...")
                continue
            
            region_values = df[feature_col].values
            
            if weighted_values is None:
                weighted_values = weight * region_values
            else:
                weighted_values += weight * region_values
        
        result[f'{feature_col}_agg'] = weighted_values
        
        return result
    
    def aggregate_all_features(self, region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate all stress features across regions.
        
        Returns:
            DataFrame with all aggregated features
        """
        # Get list of features from first region
        first_region = list(region_dfs.values())[0]
        
        # Identify cumulative stress features (these are the ones we want to aggregate)
        feature_cols = [col for col in first_region.columns 
                       if ('stress' in col and ('_7d' in col or '_14d' in col or '_30d' in col)) 
                       or ('gdd' in col and ('_7d' in col or '_14d' in col or '_30d' in col))]
        
        logger.info(f"Aggregating {len(feature_cols)} features for {self.commodity}")
        
        # Start with dates
        result = pd.DataFrame({'date': pd.to_datetime(first_region['date'])})
        
        # Aggregate each feature
        for feature in feature_cols:
            feature_df = self.aggregate_feature(region_dfs, feature)
            result[f'{feature}_agg'] = feature_df[f'{feature}_agg']
        
        # Add metadata
        result['commodity'] = self.commodity
        
        return result
    
    def compute_regional_divergence(self, region_dfs: Dict[str, pd.DataFrame], feature_col: str) -> pd.DataFrame:
        """
        Compute standard deviation across regions (divergence signal).
        High divergence = regional heterogeneity (less reliable signal).
        """
        result = pd.DataFrame({'date': pd.to_datetime(list(region_dfs.values())[0]['date'])})
        
        # Stack regional values
        regional_values = []
        for region, df in region_dfs.items():
            if feature_col in df.columns:
                regional_values.append(df[feature_col].values)
        
        if regional_values:
            regional_values = np.array(regional_values)
            result[f'{feature_col}_divergence'] = np.std(regional_values, axis=0)
        
        return result


def aggregate_commodity_regions(commodity: str, 
                                region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convenience function to aggregate all regions for a commodity.
    
    Args:
        commodity: Commodity name
        region_dfs: Dict mapping region names to processed DataFrames
        
    Returns:
        Aggregated DataFrame with commodity-level signals
    """
    aggregator = RegionalAggregator(commodity)
    aggregated = aggregator.aggregate_all_features(region_dfs)
    
    logger.info(f"Aggregated {len(region_dfs)} regions for {commodity}")
    
    return aggregated