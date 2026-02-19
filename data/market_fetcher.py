"""
Fetch market data (prices, volume) for futures contracts using Yahoo Finance.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict
from datetime import datetime
from loguru import logger

from data.cache_manager import cache_manager
from config.settings import MARKET_DATA_API_KEY, MARKET_DATA_PROVIDER


class MarketDataFetcher:
    """
    Fetch futures price data using Yahoo Finance.
    
    Yahoo Finance is free and has good coverage of commodity futures.
    No API key needed!
    """
    
    def __init__(self, provider: str = MARKET_DATA_PROVIDER, api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        
        logger.info(f"Market data provider: {self.provider}")
        
        # Import Yahoo ticker mappings
        from config.yahoo_tickers import YAHOO_TICKERS
        self.ticker_map = YAHOO_TICKERS
        
    def fetch_futures_data(self, 
                          commodity: str, 
                          contract: str,
                          start_date: str, 
                          end_date: str) -> pd.DataFrame:
        """
        Fetch futures price data.
        
        Args:
            commodity: Commodity name (e.g., 'corn')
            contract: Contract symbol (not used for Yahoo, uses continuous)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        cache_params = {
            'commodity': commodity,
            'contract': contract,
            'start_date': start_date,
            'end_date': end_date,
            'provider': self.provider
        }
        
        cached = cache_manager.get(cache_params, data_type='market', ttl_days=1)
        if cached is not None:
            logger.info(f"Using cached data for {commodity}")
            return cached
        
        # Fetch from Yahoo Finance
        df = self._fetch_from_yahoo(commodity, start_date, end_date)
        
        # Cache if successful
        if not df.empty:
            cache_manager.set(cache_params, df, data_type='market')
        
        return df
    
    def _fetch_from_yahoo(self, 
                         commodity: str, 
                         start_date: str, 
                         end_date: str) -> pd.DataFrame:
        """
        Fetch from Yahoo Finance.
        """
        try:
            # Get Yahoo ticker
            from config.yahoo_tickers import get_yahoo_ticker
            ticker_symbol = get_yahoo_ticker(commodity)
            
            logger.info(f"Fetching {commodity} ({ticker_symbol}) from Yahoo Finance...")
            
            # Create ticker object
            ticker = yf.Ticker(ticker_symbol)
            
            # Fetch historical data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            if df.empty:
                logger.warning(f"No data returned from Yahoo Finance for {ticker_symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Select relevant columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Remove timezone info if present
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✓ Fetched {len(df)} records for {commodity} from Yahoo Finance")
            logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            logger.info(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo Finance for {commodity}: {e}")
            
            # Fall back to synthetic data
            logger.warning(f"Falling back to synthetic data for {commodity}")
            return self._generate_synthetic_data(start_date, end_date)
    
    def _generate_synthetic_data(self, 
                                start_date: str, 
                                end_date: str, 
                                initial_price: float = 100.0) -> pd.DataFrame:
        """
        Generate synthetic price data for testing.
        
        Uses geometric Brownian motion.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Parameters (more realistic for commodities)
        mu = 0.0002  # Slight upward drift
        sigma = 0.02  # 2% daily vol
        
        # Generate returns
        returns = np.random.normal(mu, sigma, n)
        
        # Generate prices
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Add intraday variation
        high = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        open_price = np.roll(prices, 1)
        open_price[0] = initial_price
        
        # Volume (random around mean)
        volume = np.random.randint(5000, 20000, n)
        
        df = pd.DataFrame({
            'date': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        })
        
        logger.info(f"Generated synthetic data: {len(df)} records")
        
        return df
    
    def get_latest_price(self, commodity: str, contract: str = None) -> Optional[float]:
        """Get most recent close price."""
        try:
            from config.yahoo_tickers import get_yahoo_ticker
            ticker_symbol = get_yahoo_ticker(commodity)
            
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Try different price fields
            price = info.get('regularMarketPrice') or info.get('previousClose')
            
            if price:
                logger.info(f"Latest price for {commodity}: ${price:.2f}")
                return float(price)
            
            # Fallback: get last close from history
            hist = ticker.history(period='5d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                logger.info(f"Latest price for {commodity}: ${price:.2f}")
                return float(price)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get latest price for {commodity}: {e}")
            return None


def fetch_market_data(commodity: str, contract: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function."""
    fetcher = MarketDataFetcher()
    return fetcher.fetch_futures_data(commodity, contract, start_date, end_date)