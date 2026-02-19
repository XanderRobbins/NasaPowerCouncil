"""
Fetch market data (prices, volume) for futures contracts.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from datetime import datetime
from loguru import logger

from data.cache_manager import cache_manager
from config.settings import MARKET_DATA_API_KEY, MARKET_DATA_PROVIDER


class MarketDataFetcher:
    """
    Fetch futures price data.
    
    This is a placeholder that generates synthetic data.
    In production, connect to your data provider:
    - Quandl
    - Polygon.io
    - Interactive Brokers API
    - Bloomberg
    - Refinitiv
    etc.
    """
    
    def __init__(self, provider: str = MARKET_DATA_PROVIDER, api_key: Optional[str] = MARKET_DATA_API_KEY):
        self.provider = provider
        self.api_key = api_key
        
        if self.api_key is None:
            logger.warning("No market data API key configured. Using synthetic data.")
    
    def fetch_futures_data(self, 
                          commodity: str, 
                          contract: str,
                          start_date: str, 
                          end_date: str) -> pd.DataFrame:
        """
        Fetch futures price data.
        
        Args:
            commodity: Commodity name
            contract: Contract symbol (e.g., 'ZCZ24')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache
        cache_params = {
            'commodity': commodity,
            'contract': contract,
            'start_date': start_date,
            'end_date': end_date,
            'provider': self.provider
        }
        
        cached = cache_manager.get(cache_params, data_type='market', ttl_days=1)
        if cached is not None:
            return cached
        
        # Fetch from provider
        if self.provider == 'quandl' and self.api_key:
            df = self._fetch_from_quandl(commodity, contract, start_date, end_date)
        elif self.provider == 'polygon' and self.api_key:
            df = self._fetch_from_polygon(commodity, contract, start_date, end_date)
        else:
            # Fallback: synthetic data
            logger.warning(f"Using synthetic data for {commodity} {contract}")
            df = self._generate_synthetic_data(start_date, end_date)
        
        # Cache
        if not df.empty:
            cache_manager.set(cache_params, df, data_type='market')
        
        return df
    
    def _fetch_from_quandl(self, commodity: str, contract: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch from Quandl.
        
        NOTE: Quandl futures data typically accessed via CHRIS database.
        Example: CHRIS/CME_C1 for corn front month
        """
        try:
            import quandl
            quandl.ApiConfig.api_key = self.api_key
            
            # Map commodity to Quandl symbols (you'll need to expand this)
            quandl_codes = {
                'corn': 'CHRIS/CME_C',
                'soybeans': 'CHRIS/CME_S',
                'wheat': 'CHRIS/CME_W',
                'coffee': 'CHRIS/ICE_KC',
                'natural_gas': 'CHRIS/CME_NG'
            }
            
            code = quandl_codes.get(commodity)
            if code is None:
                logger.error(f"Quandl code not found for {commodity}")
                return pd.DataFrame()
            
            # Fetch continuous contract (front month)
            df = quandl.get(f"{code}1", start_date=start_date, end_date=end_date)
            
            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Last': 'close',
                'Volume': 'volume',
                'Open Interest': 'open_interest'
            })
            
            df = df.reset_index().rename(columns={'Date': 'date'})
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch from Quandl: {e}")
            return pd.DataFrame()
    
    def _fetch_from_polygon(self, commodity: str, contract: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch from Polygon.io.
        
        NOTE: Polygon uses different ticker formats.
        """
        try:
            import requests
            
            # Map to Polygon ticker (example)
            ticker_map = {
                'corn': 'C',
                'soybeans': 'S',
                'natural_gas': 'NG'
            }
            
            ticker = ticker_map.get(commodity, commodity.upper())
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {'apiKey': self.api_key}
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            
            # Rename columns
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch from Polygon: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_data(self, start_date: str, end_date: str, initial_price: float = 100.0) -> pd.DataFrame:
        """
        Generate synthetic price data for testing.
        
        Uses geometric Brownian motion.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Parameters
        mu = 0.0001  # Drift
        sigma = 0.015  # Volatility
        
        # Generate returns
        returns = np.random.normal(mu, sigma, n)
        
        # Generate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Add intraday variation
        high = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
        open_price = np.roll(prices, 1)
        open_price[0] = initial_price
        
        # Volume (random around mean)
        volume = np.random.randint(10000, 50000, n)
        
        df = pd.DataFrame({
            'date': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        })
        
        return df
    
    def get_latest_price(self, commodity: str, contract: str) -> Optional[float]:
        """Get most recent close price."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
        
        df = self.fetch_futures_data(commodity, contract, start_date, end_date)
        
        if df.empty:
            return None
        
        return df['close'].iloc[-1]


def fetch_market_data(commodity: str, contract: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function."""
    fetcher = MarketDataFetcher()
    return fetcher.fetch_futures_data(commodity, contract, start_date, end_date)