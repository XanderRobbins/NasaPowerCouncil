"""
NASA POWER API client for fetching climate data.
"""
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time
from loguru import logger

from config.settings import NASA_POWER_BASE_URL, RAW_DATA_PATH

class NASAPowerFetcher:
    """
    Fetch daily climate data from NASA POWER API.
    
    Parameters available:
    - T2M: Temperature at 2 meters (°C)
    - T2M_MAX: Maximum temperature (°C)
    - T2M_MIN: Minimum temperature (°C)
    - PRECTOTCORR: Precipitation (mm/day)
    - ALLSKY_SFC_SW_DWN: Solar radiation (MJ/m²/day)
    - WS2M: Wind speed at 2 meters (m/s)
    - RH2M: Relative humidity at 2 meters (%)
    """
    
    PARAMETERS = [
        'T2M',                  # Avg temp
        'T2M_MAX',             # Max temp
        'T2M_MIN',             # Min temp
        'PRECTOTCORR',         # Precipitation
        'ALLSKY_SFC_SW_DWN',   # Solar radiation
        'WS2M',                # Wind speed
        'RH2M'                 # Relative humidity
    ]
    
    def __init__(self, base_url: str = NASA_POWER_BASE_URL):
        self.base_url = base_url
        self.cache_dir = RAW_DATA_PATH / 'climate'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_region(self,
                    lat: float,
                    lon: float,
                    start_date: str,
                    end_date: str,
                    region_name: str) -> pd.DataFrame:
        """
        Fetch climate data for a specific region and date range.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            region_name: Name for caching
            
        Returns:
            DataFrame with daily climate data
        """
        # Check cache first
        cache_file = self.cache_dir / f"{region_name}_{start_date}_{end_date}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached data for {region_name}")
            return pd.read_pickle(cache_file)
        
        # Format dates
        start_date_fmt = pd.to_datetime(start_date).strftime('%Y%m%d')
        end_date_fmt = pd.to_datetime(end_date).strftime('%Y%m%d')
        
        # Build URL
        params = ','.join(self.PARAMETERS)
        url = (f"{self.base_url}?"
               f"parameters={params}&"
               f"community=AG&"
               f"longitude={lon}&"
               f"latitude={lat}&"
               f"start={start_date_fmt}&"
               f"end={end_date_fmt}&"
               f"format=JSON")
        
        logger.info(f"Fetching NASA POWER data for {region_name} ({lat}, {lon})")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            df = self._parse_response(data, region_name)
            
            # Cache
            df.to_pickle(cache_file)
            logger.info(f"Cached data for {region_name}")
            
            # Rate limit: NASA POWER has limits
            time.sleep(1)
            
            return df
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data for {region_name}: {e}")
            raise
    
    def _parse_response(self, data: dict, region_name: str) -> pd.DataFrame:
        """Parse NASA POWER JSON response into DataFrame."""
        parameters = data['properties']['parameter']
        
        # Extract time series
        records = []
        for param_name, param_data in parameters.items():
            for date_str, value in param_data.items():
                # Find or create record for this date
                date = pd.to_datetime(date_str, format='%Y%m%d')
                
                # Find existing record
                record = next((r for r in records if r['date'] == date), None)
                if record is None:
                    record = {'date': date, 'region': region_name}
                    records.append(record)
                
                # Map parameter name to simpler name
                param_map = {
                    'T2M': 'temp_avg',
                    'T2M_MAX': 'temp_max',
                    'T2M_MIN': 'temp_min',
                    'PRECTOTCORR': 'precipitation',
                    'ALLSKY_SFC_SW_DWN': 'solar_radiation',
                    'WS2M': 'wind_speed',
                    'RH2M': 'relative_humidity'
                }
                
                record[param_map.get(param_name, param_name)] = value
        
        df = pd.DataFrame(records)
        df = df.sort_values('date').reset_index(drop=True)
        
        # Replace missing value indicators (-999) with NaN
        df = df.replace(-999, np.nan)
        
        return df
    
    def fetch_commodity_regions(self,
                               commodity: str,
                               start_date: str,
                               end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch climate data for all regions of a commodity.
        
        Returns:
            Dict mapping region names to DataFrames
        """
        from config.regions import get_commodity_regions
        
        regions = get_commodity_regions(commodity)
        region_data = {}
        
        for region_name, region_info in regions.items():
            lat = region_info['lat']
            lon = region_info['lon']
            
            df = self.fetch_region(lat, lon, start_date, end_date,
                                  f"{commodity}_{region_name}")
            region_data[region_name] = df
        
        logger.info(f"Fetched climate data for {len(region_data)} regions of {commodity}")
        return region_data


def fetch_commodity_regions(commodity: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch climate data for all regions of a commodity.
    
    Args:
        commodity: Commodity name
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Dict mapping region names to DataFrames
    """
    fetcher = NASAPowerFetcher()
    return fetcher.fetch_commodity_regions(commodity, start_date, end_date)