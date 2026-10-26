"""
Local on-disk cache for downloaded market data, so repeated runs do not hit Yahoo again.

Entries are pickled DataFrames keyed by an MD5 hash of the request parameters, stored under
``data_storage/raw/<data_type>/``. A cached file is only used while it is younger than its
time-to-live; older files are ignored (and overwritten by the next successful fetch).

Note: NASA POWER climate data uses its own simple file cache in ``climate_fetcher.py``
because historical weather never goes stale.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from config.settings import RAW_DATA_PATH


class CacheManager:
    """Pickle-backed cache with a time-to-live (TTL) per lookup."""

    def __init__(self, cache_dir: Path = RAW_DATA_PATH, default_ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.default_ttl_days = default_ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, params: dict) -> str:
        """Hash the request parameters (sorted, so key order does not matter)."""
        return hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()

    def _get_cache_path(self, cache_key: str, data_type: str) -> Path:
        subdir = self.cache_dir / data_type
        subdir.mkdir(exist_ok=True)
        return subdir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path: Path, ttl_days: Optional[int] = None) -> bool:
        """True if the file exists and was written less than ``ttl_days`` ago."""
        if not cache_path.exists():
            return False
        ttl = ttl_days if ttl_days is not None else self.default_ttl_days
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age < timedelta(days=ttl)

    def get(self, params: dict, data_type: str = 'market',
            ttl_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Return the cached DataFrame for ``params``, or None on a miss / expired / unreadable file."""
        cache_key = self._generate_cache_key(params)
        cache_path = self._get_cache_path(cache_key, data_type)

        if not self._is_cache_valid(cache_path, ttl_days):
            logger.debug(f"Cache MISS: {cache_key}")
            return None
        try:
            logger.debug(f"Cache HIT: {cache_key}")
            return pd.read_pickle(cache_path)
        except Exception as e:  # corrupt file: treat as a miss
            logger.warning(f"Failed to read cache {cache_key}: {e}")
            return None

    def set(self, params: dict, data: pd.DataFrame, data_type: str = 'market'):
        """Write ``data`` to the cache; a failed write is logged but never stops the run."""
        cache_key = self._generate_cache_key(params)
        cache_path = self._get_cache_path(cache_key, data_type)
        try:
            data.to_pickle(cache_path)
            logger.debug(f"Cached data: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to cache data {cache_key}: {e}")


# Shared instance used by the market fetcher.
cache_manager = CacheManager()
