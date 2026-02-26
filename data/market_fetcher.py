"""
Fetch market data (prices, volume) for futures contracts using Yahoo Finance.
REAL DATA ONLY - No synthetic fallback.
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
from datetime import datetime
from loguru import logger

from data.cache_manager import cache_manager


class MarketDataFetcher:
    """
    Fetch futures price data using Yahoo Finance.
    
    Yahoo Finance is free and has good coverage of commodity futures.
    No API key needed!
    
    IMPORTANT: This fetcher requires real market data. If Yahoo Finance
    fails, the system will raise an error rather than using synthetic data.
    """
    
    # Yahoo Finance ticker mappings
    YAHOO_TICKERS = {
        'corn': 'ZC=F',
        'soybeans': 'ZS=F',
        'wheat': 'ZW=F',
        'coffee': 'KC=F',
        'sugar': 'SB=F',
        'cotton': 'CT=F',
        'natural_gas': 'NG=F',
    }
    
    def __init__(self):
        logger.info("Market data provider: Yahoo Finance (REAL DATA ONLY)")
    
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
            
        Raises:
            RuntimeError: If Yahoo Finance fetch fails
        """
        # Check cache first
        cache_params = {
            'commodity': commodity,
            'start_date': start_date,
            'end_date': end_date,
            'provider': 'yahoo'
        }
        
        cached = cache_manager.get(cache_params, data_type='market', ttl_days=1)
        if cached is not None:
            logger.info(f"Using cached data for {commodity}")
            return cached
        
        # Fetch from Yahoo Finance (will raise error if it fails)
        df = self._fetch_from_yahoo(commodity, start_date, end_date)
        
        # Cache if successful
        if not df.empty:
            cache_manager.set(cache_params, df, data_type='market')
        else:
            raise RuntimeError(f"Yahoo Finance returned empty DataFrame for {commodity}")
        
        return df
    
    def _fetch_from_yahoo(self,
                         commodity: str,
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        Fetch from Yahoo Finance.
        
        Raises:
            RuntimeError: If fetch fails for any reason
        """
        try:
            # Get Yahoo ticker
            ticker_symbol = self.YAHOO_TICKERS.get(commodity.lower())
            if not ticker_symbol:
                raise ValueError(f"No Yahoo ticker mapping for {commodity}")
            
            logger.info(f"Fetching {commodity} ({ticker_symbol}) from Yahoo Finance...")
            
            # Create ticker object
            ticker = yf.Ticker(ticker_symbol)
            
            # Fetch historical data
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            # Check if data was returned
            if df is None or df.empty:
                raise RuntimeError(
                    f"No data returned from Yahoo Finance for {ticker_symbol}. "
                    f"Date range: {start_date} to {end_date}. "
                    f"This could be due to: (1) network issues, (2) Yahoo Finance API changes, "
                    f"(3) invalid date range, or (4) ticker symbol issues."
                )
            
            # Standardize column names
            df = df.reset_index()
            
            # Handle potential column name variations
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Verify required columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise RuntimeError(
                    f"Yahoo Finance data missing required columns: {missing_cols}. "
                    f"Available columns: {df.columns.tolist()}"
                )
            
            # Select relevant columns
            df = df[required_cols]
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Remove timezone info if present
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✓ Fetched {len(df)} records for {commodity}")
            logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            logger.info(f"  Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo Finance for {commodity}: {e}")
            logger.error("No fallback data available - real data required!")
            raise RuntimeError(
                f"Could not fetch real market data for {commodity}. "
                f"Error: {str(e)}. "
                f"Check your internet connection and try: pip install --upgrade yfinance"
            ) from e
    
    def get_latest_price(self, commodity: str) -> Optional[float]:
        """
        Get most recent close price.
        
        Returns None if fetch fails (non-critical operation).
        """
        try:
            ticker_symbol = self.YAHOO_TICKERS.get(commodity.lower())
            if not ticker_symbol:
                logger.warning(f"No Yahoo ticker for {commodity}")
                return None
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Get last close from history
            hist = ticker.history(period='5d')
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                logger.info(f"Latest price for {commodity}: ${price:.2f}")
                return float(price)
            
            logger.warning(f"No recent price data for {commodity}")
            return None
        
        except Exception as e:
            logger.error(f"Failed to get latest price for {commodity}: {e}")
            return None


def fetch_market_data(commodity: str, contract: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Convenience function to fetch market data.
    
    Raises:
        RuntimeError: If real data cannot be fetched
    """
    fetcher = MarketDataFetcher()
    return fetcher.fetch_futures_data(commodity, contract, start_date, end_date)