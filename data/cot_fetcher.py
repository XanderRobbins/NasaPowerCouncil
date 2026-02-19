"""
Fetch COT (Commitments of Traders) positioning data.
"""
import pandas as pd
import requests
from typing import Optional
from datetime import datetime
from loguru import logger

from data.cache_manager import cache_manager


class COTFetcher:
    """
    Fetch COT reports from CFTC.
    
    COT reports show positioning of:
    - Commercial hedgers
    - Large speculators (non-commercial)
    - Small speculators
    """
    
    # CFTC API endpoints
    BASE_URL = "https://publicreporting.cftc.gov/resource/"
    FUTURES_ONLY = "jun7-fc8e.json"  # Futures only legacy report
    
    # Commodity codes (CFTC uses specific codes)
    COMMODITY_CODES = {
        'corn': '002602',
        'soybeans': '005602',
        'wheat': '001602',
        'coffee': '083731',
        'natural_gas': '023651',
        'cotton': '033661',
        'sugar': '080732'
    }
    
    def __init__(self):
        self.session = requests.Session()
        
    def fetch_cot(self, commodity: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch COT data for a commodity.
        
        Args:
            commodity: Commodity name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with COT positioning data
        """
        # Check cache
        cache_params = {
            'commodity': commodity,
            'start_date': start_date,
            'end_date': end_date,
            'source': 'cftc_cot'
        }
        
        cached = cache_manager.get(cache_params, data_type='cot', ttl_days=7)
        if cached is not None:
            return cached
        
        # Get commodity code
        commodity_code = self.COMMODITY_CODES.get(commodity)
        if commodity_code is None:
            logger.warning(f"COT data not available for {commodity}")
            return pd.DataFrame()
        
        # Build query
        url = f"{self.BASE_URL}{self.FUTURES_ONLY}"
        
        # CFTC uses YYYY-MM-DD format
        params = {
            'cftc_contract_market_code': commodity_code,
            '$where': f"report_date_as_yyyy_mm_dd between '{start_date}' and '{end_date}'",
            '$order': 'report_date_as_yyyy_mm_dd ASC',
            '$limit': 10000
        }
        
        logger.info(f"Fetching COT data for {commodity} from CFTC...")
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No COT data returned for {commodity}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            
            # Parse and clean
            df = self._parse_cot_data(df)
            
            # Cache
            cache_manager.set(cache_params, df, data_type='cot')
            
            logger.info(f"Fetched {len(df)} COT records for {commodity}")
            
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch COT data: {e}")
            return pd.DataFrame()
    
    def _parse_cot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and clean COT data."""
        # Select relevant columns
        columns_to_keep = [
            'report_date_as_yyyy_mm_dd',
            'open_interest_all',
            'noncomm_positions_long_all',
            'noncomm_positions_short_all',
            'comm_positions_long_all',
            'comm_positions_short_all',
            'nonrept_positions_long_all',
            'nonrept_positions_short_all'
        ]
        
        df = df[[col for col in columns_to_keep if col in df.columns]].copy()
        
        # Rename for clarity
        df = df.rename(columns={
            'report_date_as_yyyy_mm_dd': 'date',
            'open_interest_all': 'open_interest',
            'noncomm_positions_long_all': 'large_spec_long',
            'noncomm_positions_short_all': 'large_spec_short',
            'comm_positions_long_all': 'commercial_long',
            'comm_positions_short_all': 'commercial_short',
            'nonrept_positions_long_all': 'small_spec_long',
            'nonrept_positions_short_all': 'small_spec_short'
        })
        
        # Convert to numeric
        numeric_cols = [col for col in df.columns if col != 'date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'])
        
        # Compute net positions
        df['large_spec_net'] = df['large_spec_long'] - df['large_spec_short']
        df['commercial_net'] = df['commercial_long'] - df['commercial_short']
        df['small_spec_net'] = df['small_spec_long'] - df['small_spec_short']
        
        # Compute percentiles (for extreme positioning detection)
        for col in ['large_spec_net', 'commercial_net']:
            df[f'{col}_percentile'] = df[col].rank(pct=True) * 100
        
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def get_latest_positioning(self, commodity: str) -> Optional[dict]:
        """Get most recent COT positioning."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        
        df = self.fetch_cot(commodity, start_date, end_date)
        
        if df.empty:
            return None
        
        latest = df.iloc[-1]
        
        return {
            'date': latest['date'],
            'large_spec_net': latest['large_spec_net'],
            'commercial_net': latest['commercial_net'],
            'net_position_percentile': latest['large_spec_net_percentile']
        }


def fetch_cot_data(commodity: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function."""
    fetcher = COTFetcher()
    return fetcher.fetch_cot(commodity, start_date, end_date)