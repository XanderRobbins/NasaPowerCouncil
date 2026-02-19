"""
Orchestrate the full feature engineering pipeline.
"""
import pandas as pd
from typing import Dict, List
from loguru import logger

from features.seasonal_deviation import compute_seasonal_deviations_for_region
from features.stress_calculator import compute_stress_for_region
from features.cumulative_stress import compute_cumulative_stress_for_region
from features.regional_aggregator import aggregate_commodity_regions


class FeaturePipeline:
    """
    Full feature engineering pipeline from raw climate data to aggregated signals.
    
    Steps:
    1. Seasonal deviation (Z-scores)
    2. Stress calculations (heat, dry, etc.)
    3. Cumulative stress (rolling windows)
    4. Regional aggregation (production-weighted)
    """
    
    def __init__(self, commodity: str):
        self.commodity = commodity
        
    def process_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """
        Process a single region through the feature pipeline.
        
        Args:
            df: Raw climate data for region
            region: Region name
            
        Returns:
            Processed DataFrame with all features
        """
        logger.info(f"Processing features for {self.commodity} - {region}")
        
        # Step 1: Seasonal deviations
        df = compute_seasonal_deviations_for_region(df)
        
        # Step 2: Stress calculations
        df = compute_stress_for_region(df, self.commodity, region)
        
        # Step 3: Cumulative stress
        df = compute_cumulative_stress_for_region(df)
        
        logger.debug(f"Generated {len(df.columns)} features for {region}")
        
        return df
    
    def process_all_regions(self, region_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process all regions for this commodity.
        
        Args:
            region_dfs: Dict mapping region names to raw climate DataFrames
            
        Returns:
            Dict mapping region names to processed DataFrames
        """
        processed = {}
        
        for region, df in region_dfs.items():
            processed[region] = self.process_region(df, region)
        
        return processed
    
    def aggregate(self, processed_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate processed regional data into commodity-level signals.
        
        Args:
            processed_dfs: Dict mapping region names to processed DataFrames
            
        Returns:
            Aggregated commodity-level DataFrame
        """
        logger.info(f"Aggregating regions for {self.commodity}")
        
        aggregated = aggregate_commodity_regions(self.commodity, processed_dfs)
        
        return aggregated
    
    def run(self, region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Run the full pipeline: raw data → aggregated signals.
        
        Args:
            region_dfs: Dict mapping region names to raw climate DataFrames
            
        Returns:
            Commodity-level DataFrame with aggregated features
        """
        logger.info(f"Running feature pipeline for {self.commodity}")
        
        # Process each region
        processed = self.process_all_regions(region_dfs)
        
        # Aggregate
        aggregated = self.aggregate(processed)
        
        logger.info(f"Feature pipeline complete for {self.commodity}: {aggregated.shape}")
        
        return aggregated


def run_feature_pipeline_for_commodity(commodity: str, 
                                       region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Convenience function to run the full feature pipeline for a commodity.
    """
    pipeline = FeaturePipeline(commodity)
    return pipeline.run(region_dfs)