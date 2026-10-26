"""
Combine the per-region stress features into one commodity-level feature set, using
production weights:

    F_commodity(t) = sum over regions r of  weight_r * F_r(t)

A region with missing data on a day contributes 0 ("no stress") so one region's data hole
cannot erase the others. Weights are applied as given and are not renormalised.
"""
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from config.regions import get_commodity_regions

# Suffix marking a commodity-level (aggregated) column, e.g. ``heat_stress_7d_agg``.
# The models select their features by this suffix.
AGG_SUFFIX = '_agg'


def aggregate_regional_features(commodity: str, region_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Args:
        commodity: commodity name (selects the production weights).
        region_dfs: region name -> frame with ``date`` and the stress columns.

    Returns:
        One row per date (union of all regions' dates) with a ``<column>_agg`` column per
        stress feature (raw weather columns are not aggregated).
    """
    regions = get_commodity_regions(commodity)

    # Union of every region's dates, oldest first.
    all_dates = pd.concat([df[['date']] for df in region_dfs.values()]).drop_duplicates().sort_values('date')
    aggregated = all_dates.copy()

    # Aggregate only stress columns, not the raw weather variables.
    stress_cols = []
    for df in region_dfs.values():
        stress_cols.extend(c for c in df.columns if 'stress' in c and c not in stress_cols)
    logger.info(f"Aggregating {len(stress_cols)} stress features across {len(region_dfs)} regions")

    for col in stress_cols:
        weighted_sum = np.zeros(len(all_dates))
        for region_name, df in region_dfs.items():
            if col not in df.columns:
                continue
            # Align this region to the common dates; days it lacks count as 0 stress.
            values = all_dates[['date']].merge(df[['date', col]], on='date', how='left')[col]
            weighted_sum += values.fillna(0).values * regions[region_name]['weight']
        aggregated[f'{col}{AGG_SUFFIX}'] = weighted_sum

    # Carry the last known value forward over any remaining gaps, and treat a missing start as 0.
    for col in aggregated.columns:
        if col != 'date':
            aggregated[col] = aggregated[col].ffill().fillna(0)

    logger.info(f"Aggregated features shape: {aggregated.shape}")
    return aggregated
