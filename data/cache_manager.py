"""
Local caching layer for API data to reduce redundant requests.
"""
import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, timedelta
from loguru import logger

from config.settings import RAW_DATA_PATH


class CacheManager:
    """
    Manage local caching of API responses.
    
    Caches data to Parquet files with TTL (time-to-live).
    """
    
    def __init__(self, cache_dir: Path = RAW_DATA_PATH, default_ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.default_ttl_days = default_ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, params: dict) -> str:
        """Generate cache key from parameters."""
        # Sort params for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        hash_key = hashlib.md5(sorted_params.encode()).hexdigest()
        return hash_key
        
    def _get_cache_path(self, cache_key: str, data_type: str = 'climate') -> Path:
        """Get cache file path."""
        cache_subdir = self.cache_dir / data_type
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.pkl"  # Changed from .parquet to .pkl

    def _is_cache_valid(self, cache_path: Path, ttl_days: Optional[int] = None) -> bool:
        """Check if cache is still valid (not expired)."""
        if not cache_path.exists():
            return False
        
        ttl = ttl_days if ttl_days is not None else self.default_ttl_days
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mod_time
        
        return age < timedelta(days=ttl)
    
    def get(self, params: dict, data_type: str = 'climate', ttl_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get data from cache if available and valid.
        
        Args:
            params: Dict of parameters used to fetch data
            data_type: Type of data ('climate', 'market', 'cot')
            ttl_days: Override default TTL
            
        Returns:
            Cached DataFrame if available, else None
        """
        cache_key = self._generate_cache_key(params)
        cache_path = self._get_cache_path(cache_key, data_type)
        
        if self._is_cache_valid(cache_path, ttl_days):
            logger.debug(f"Cache HIT: {cache_key}")
            try:
                df = pd.read_pickle(cache_path)
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache {cache_key}: {e}")
                return None
        
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    def set(self, params: dict, data: pd.DataFrame, data_type: str = 'climate'):
        """
        Save data to cache.
        
        Args:
            params: Dict of parameters
            data: DataFrame to cache
            data_type: Type of data
        """
        cache_key = self._generate_cache_key(params)
        cache_path = self._get_cache_path(cache_key, data_type)
        
        try:
            data.to_pickle(cache_path)
            logger.debug(f"Cached data: {cache_key}")
        except Exception as e:
            logger.error(f"Failed to cache data {cache_key}: {e}")
    
    def invalidate(self, params: dict, data_type: str = 'climate'):
        """Invalidate (delete) cache entry."""
        cache_key = self._generate_cache_key(params)
        cache_path = self._get_cache_path(cache_key, data_type)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Invalidated cache: {cache_key}")
    
    def clear_expired(self, data_type: Optional[str] = None):
        """Clear all expired cache entries."""
        if data_type:
            dirs_to_check = [self.cache_dir / data_type]
        else:
            dirs_to_check = [d for d in self.cache_dir.iterdir() if d.is_dir()]
        
        cleared_count = 0
        for cache_dir in dirs_to_check:
            for cache_file in cache_dir.glob("*.parquet"):
                if not self._is_cache_valid(cache_file):
                    cache_file.unlink()
                    cleared_count += 1
        
        logger.info(f"Cleared {cleared_count} expired cache entries")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            'total_entries': 0,
            'total_size_mb': 0.0,
            'by_type': {}
        }
        
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in files)
                
                stats['by_type'][subdir.name] = {
                    'count': len(files),
                    'size_mb': total_size / (1024 * 1024)
                }
                
                stats['total_entries'] += len(files)
                stats['total_size_mb'] += total_size / (1024 * 1024)
        
        return stats


# Global cache instance
cache_manager = CacheManager()