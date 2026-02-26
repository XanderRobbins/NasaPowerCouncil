"""Configuration module."""
from config.settings import *
from config.regions import COMMODITY_REGIONS, get_commodity_regions
from config.thresholds import CRITICAL_THRESHOLDS, get_thresholds

__all__ = [
    'COMMODITY_REGIONS', 'get_commodity_regions',
    'CRITICAL_THRESHOLDS', 'get_thresholds',
]