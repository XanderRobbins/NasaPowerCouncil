"""
Raw weather -> model features for one commodity.

Steps, per region and then across regions:

1. seasonal z-scores         (``seasonal_deviation``)  how unusual is today for this date?
2. stress indicators         (``stress_calculator``)   heat / cold / dry, plus 7/14/30-day sums
3. production-weighted mean  (``regional_aggregator``) one commodity-level feature table
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import pandas as pd
from loguru import logger

from features.regional_aggregator import aggregate_regional_features
from features.seasonal_deviation import compute_seasonal_deviations
from features.stress_calculator import compute_stress_indicators


class FeaturePipeline:
    """Turns a commodity's raw regional weather into its aggregated feature table."""

    def __init__(self, commodity: str):
        self.commodity = commodity

    def process_region(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Seasonal z-scores, then stress indicators, for one region."""
        logger.info(f"Processing features for {self.commodity} - {region}")
        df = compute_seasonal_deviations(df)
        return compute_stress_indicators(df, self.commodity)

    def run(self, region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Args:
            region_dfs: region name -> raw daily weather frame (as returned by the climate fetcher).

        Returns:
            Commodity-level DataFrame: ``date`` plus the aggregated ``*_agg`` stress features.
        """
        logger.info(f"Running feature pipeline for {self.commodity}")

        # Regions are independent, so process them in parallel.
        with ThreadPoolExecutor() as executor:
            futures = {region: executor.submit(self.process_region, df, region)
                       for region, df in region_dfs.items()}
            processed = {region: future.result() for region, future in futures.items()}

        aggregated = aggregate_regional_features(self.commodity, processed)
        logger.info(f"Feature pipeline complete for {self.commodity}: {aggregated.shape}")
        return aggregated
