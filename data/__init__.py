"""
Data fetching and management module.
"""

from data.climate_fetcher import NASAPowerFetcher, fetch_commodity_regions
from data.market_fetcher import MarketDataFetcher, fetch_market_data
from data.cot_fetcher import COTFetcher, fetch_cot_data
from data.data_validator import DataValidator, validate_data
from data.cache_manager import CacheManager, cache_manager

__all__ = [
    'NASAPowerFetcher',
    'MarketDataFetcher',
    'COTFetcher',
    'DataValidator',
    'CacheManager',
    'cache_manager',
    'fetch_commodity_regions',
    'fetch_market_data',
    'fetch_cot_data',
    'validate_data',
]