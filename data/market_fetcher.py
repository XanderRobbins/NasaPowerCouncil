"""
Futures price download from Yahoo Finance (continuous front-month contracts).

Real data only: any failure raises instead of falling back to synthetic prices, because
a backtest on made-up prices is worse than no backtest.

IMPORTANT: Yahoo's continuous series (e.g. ``ZC=F``) is NOT adjusted for contract rolls.
On the day it switches from the expiring contract to the next one, the close jumps by the
calendar spread, and a backtest holding through that day books the jump as profit or loss
even though no one earned it. Keep this in mind when reading results.
"""
import pandas as pd
import yfinance as yf
from loguru import logger

from data.cache_manager import cache_manager

# Yahoo Finance continuous-contract tickers for the commodities that have regions and
# stress thresholds configured (config/regions.py, config/thresholds.py).
YAHOO_TICKERS = {
    'corn': 'ZC=F',
    'soybeans': 'ZS=F',
    'wheat': 'ZW=F',
    'coffee': 'KC=F',
    'sugar': 'SB=F',
    'cotton': 'CT=F',
    'live_cattle': 'LE=F',
    'lean_hogs': 'HE=F',
    'cocoa': 'CC=F',
}

REQUIRED_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume']


class MarketDataFetcher:
    """Download daily OHLCV futures prices, cached for one day."""

    def __init__(self):
        logger.info("Market data provider: Yahoo Finance (REAL DATA ONLY)")

    def fetch_futures_data(self, commodity: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Daily OHLCV for ``commodity`` between ``start_date`` and ``end_date`` (YYYY-MM-DD).

        Returns a DataFrame with columns date, open, high, low, close, volume, sorted by date.
        Raises RuntimeError if Yahoo returns nothing or the download fails.
        """
        cache_params = {
            'commodity': commodity,
            'start_date': start_date,
            'end_date': end_date,
            'provider': 'yahoo',
        }

        cached = cache_manager.get(cache_params, data_type='market', ttl_days=1)
        if cached is not None:
            logger.info(f"Using cached data for {commodity}")
            return cached

        df = self._fetch_from_yahoo(commodity, start_date, end_date)
        cache_manager.set(cache_params, df, data_type='market')
        return df

    def _fetch_from_yahoo(self, commodity: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and normalise one price history; raises RuntimeError on any failure."""
        try:
            ticker_symbol = YAHOO_TICKERS.get(commodity.lower())
            if not ticker_symbol:
                raise ValueError(f"No Yahoo ticker mapping for {commodity}")

            logger.info(f"Fetching {commodity} ({ticker_symbol}) from Yahoo Finance...")
            df = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date, interval='1d')
            if df is None or df.empty:
                raise RuntimeError(
                    f"No data returned from Yahoo Finance for {ticker_symbol} ({start_date} to {end_date})"
                )

            # Yahoo's capitalised column names -> lowercase names used everywhere else.
            df = df.reset_index().rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high',
                'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            })
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                raise RuntimeError(
                    f"Yahoo Finance data missing required columns: {missing}. "
                    f"Available columns: {df.columns.tolist()}"
                )

            df = df[REQUIRED_COLUMNS].copy()
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].dt.tz is not None:  # the rest of the pipeline uses timezone-naive dates
                df['date'] = df['date'].dt.tz_localize(None)
            df = df.sort_values('date').reset_index(drop=True)

            logger.info(f"Fetched {len(df)} records for {commodity}: "
                        f"{df['date'].min().date()} to {df['date'].max().date()}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch from Yahoo Finance for {commodity}: {e}")
            raise RuntimeError(
                f"Could not fetch real market data for {commodity}. Error: {e}. "
                f"Check your internet connection and try: pip install --upgrade yfinance"
            ) from e
