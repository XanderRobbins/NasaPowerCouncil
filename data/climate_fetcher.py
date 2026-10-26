"""
NASA POWER client: daily point weather for each of a commodity's production regions.

Downloaded history never changes, so each region/date-range is cached forever as a pickle
in ``data_storage/raw/climate/`` (delete the file to force a re-download).

Note on timing: NASA POWER publishes with a 1-2+ day delay. The backtest accounts for this
with ``FEATURE_LAG_DAYS`` (see ``backtest/engine.py``); this module only downloads.
"""
import time
from typing import Dict

import numpy as np
import pandas as pd
import requests
from loguru import logger

from config.regions import get_commodity_regions
from config.settings import NASA_POWER_BASE_URL, RAW_DATA_PATH

# NASA's API parameter names -> the short column names used throughout this project.
PARAMETER_COLUMNS = {
    'T2M': 'temp_avg',                  # mean temperature at 2 m (deg C)
    'T2M_MAX': 'temp_max',              # max temperature (deg C)
    'T2M_MIN': 'temp_min',              # min temperature (deg C)
    'PRECTOTCORR': 'precipitation',     # precipitation (mm/day)
    'ALLSKY_SFC_SW_DWN': 'solar_radiation',
    'WS2M': 'wind_speed',
    'RH2M': 'relative_humidity',
}
MISSING_VALUE = -999  # NASA's "no data" marker

# Sanity bound on average rainfall (mm/day). Crop regions average 1-6; the wettest places
# on Earth about 30. NASA's near-real-time data has been seen returning rainfall multiplied
# by 24 for a whole request while still labelled mm/day, which would silently corrupt every
# dryness feature, so such responses are rejected instead of cached.
MAX_PLAUSIBLE_MEAN_PRECIP = 25.0


class NASAPowerFetcher:
    """Fetch (and cache) daily climate data from the NASA POWER API."""

    def __init__(self, base_url: str = NASA_POWER_BASE_URL):
        self.base_url = base_url
        self.cache_dir = RAW_DATA_PATH / 'climate'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_region(self, lat: float, lon: float, start_date: str, end_date: str,
                     region_name: str) -> pd.DataFrame:
        """
        Daily weather for one point between ``start_date`` and ``end_date`` (YYYY-MM-DD).

        ``region_name`` is only used to name the cache file and the ``region`` column.
        """
        cache_file = self.cache_dir / f"{region_name}_{start_date}_{end_date}.pkl"
        if cache_file.exists():
            logger.info(f"Loading cached data for {region_name}")
            return pd.read_pickle(cache_file)

        params = {
            'parameters': ','.join(PARAMETER_COLUMNS),
            'community': 'AG',  # agroclimatology community
            'longitude': lon,
            'latitude': lat,
            'start': pd.to_datetime(start_date).strftime('%Y%m%d'),
            'end': pd.to_datetime(end_date).strftime('%Y%m%d'),
            'format': 'JSON',
        }
        logger.info(f"Fetching NASA POWER data for {region_name} ({lat}, {lon})")

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            df = self._parse_response(response.json(), region_name)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch data for {region_name}: {e}")
            raise

        self._check_precipitation_units(df, region_name)
        df.to_pickle(cache_file)
        logger.info(f"Cached data for {region_name}")
        time.sleep(1)  # NASA POWER is rate limited; be polite after each live call
        return df

    @staticmethod
    def _parse_response(data: dict, region_name: str) -> pd.DataFrame:
        """
        Turn the JSON payload ``{parameter: {YYYYMMDD: value}}`` into one row per day with a
        column per variable, plus ``date`` and ``region``. -999 becomes NaN.
        """
        wide = pd.DataFrame(data['properties']['parameter'])
        wide.index = pd.to_datetime(wide.index, format='%Y%m%d')
        wide = wide.rename(columns=PARAMETER_COLUMNS).replace(MISSING_VALUE, np.nan)

        df = wide.rename_axis('date').sort_index().reset_index()
        df.insert(1, 'region', region_name)
        return df

    @staticmethod
    def _check_precipitation_units(df: pd.DataFrame, region_name: str) -> None:
        """Refuse a response whose rainfall is in the wrong unit (see MAX_PLAUSIBLE_MEAN_PRECIP)."""
        if 'precipitation' not in df or len(df) < 60:
            return
        mean = df['precipitation'].mean()
        if mean > MAX_PLAUSIBLE_MEAN_PRECIP:
            raise ValueError(
                f"NASA POWER precipitation for {region_name} averages {mean:.1f} mm/day, which is "
                f"implausible (known API quirk near the newest data). Request a range ending earlier."
            )

    def fetch_commodity_regions(self, commodity: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Weather frames keyed by region name for every production region of ``commodity``."""
        region_data = {}
        for region_name, info in get_commodity_regions(commodity).items():
            region_data[region_name] = self.fetch_region(
                info['lat'], info['lon'], start_date, end_date, f"{commodity}_{region_name}"
            )
        logger.info(f"Fetched climate data for {len(region_data)} regions of {commodity}")
        return region_data
