"""
Date and time utility functions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from loguru import logger


def parse_date(date_input) -> pd.Timestamp:
    """
    Parse various date formats into pandas Timestamp.
    
    Args:
        date_input: String, datetime, or pd.Timestamp
        
    Returns:
        pd.Timestamp
    """
    if isinstance(date_input, pd.Timestamp):
        return date_input
    elif isinstance(date_input, datetime):
        return pd.Timestamp(date_input)
    elif isinstance(date_input, str):
        return pd.to_datetime(date_input)
    else:
        raise ValueError(f"Cannot parse date from type {type(date_input)}")


def get_business_days(start_date, end_date, freq='D') -> pd.DatetimeIndex:
    """
    Get business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D' for daily, 'B' for business days)
        
    Returns:
        DatetimeIndex of business days
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    dates = pd.bdate_range(start=start, end=end, freq=freq)
    return dates


def get_month_start_end(date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the first and last day of the month for a given date.
    
    Returns:
        (month_start, month_end)
    """
    date = parse_date(date)
    month_start = date.replace(day=1)
    month_end = (month_start + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    
    return month_start, month_end


def get_quarter_start_end(date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the first and last day of the quarter for a given date.
    
    Returns:
        (quarter_start, quarter_end)
    """
    date = parse_date(date)
    quarter = (date.month - 1) // 3 + 1
    quarter_start = pd.Timestamp(year=date.year, month=(quarter - 1) * 3 + 1, day=1)
    quarter_end = (quarter_start + pd.DateOffset(months=3)) - pd.Timedelta(days=1)
    
    return quarter_start, quarter_end


def get_year_start_end(date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the first and last day of the year for a given date.
    
    Returns:
        (year_start, year_end)
    """
    date = parse_date(date)
    year_start = pd.Timestamp(year=date.year, month=1, day=1)
    year_end = pd.Timestamp(year=date.year, month=12, day=31)
    
    return year_start, year_end


def is_business_day(date) -> bool:
    """Check if date is a business day."""
    date = parse_date(date)
    return bool(len(pd.bdate_range(date, date)))


def next_business_day(date, n: int = 1) -> pd.Timestamp:
    """
    Get the nth next business day.
    
    Args:
        date: Starting date
        n: Number of business days to advance
        
    Returns:
        Next business day
    """
    date = parse_date(date)
    return date + pd.offsets.BDay(n)


def previous_business_day(date, n: int = 1) -> pd.Timestamp:
    """
    Get the nth previous business day.
    
    Args:
        date: Starting date
        n: Number of business days to go back
        
    Returns:
        Previous business day
    """
    date = parse_date(date)
    return date - pd.offsets.BDay(n)


def get_trading_days_between(start_date, end_date) -> int:
    """
    Count trading days between two dates.
    
    Returns:
        Number of trading days
    """
    start = parse_date(start_date)
    end = parse_date(end_date)
    
    trading_days = pd.bdate_range(start=start, end=end)
    return len(trading_days)


def align_dates(dates1: pd.DatetimeIndex, dates2: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Find common dates between two date indices.
    
    Returns:
        Intersection of dates
    """
    return dates1.intersection(dates2)


def fill_missing_dates(df: pd.DataFrame, date_column: str = 'date', method: str = 'ffill') -> pd.DataFrame:
    """
    Fill missing dates in a DataFrame.
    
    Args:
        df: DataFrame with dates
        date_column: Name of date column
        method: Fill method ('ffill', 'bfill', 'interpolate')
        
    Returns:
        DataFrame with filled dates
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column)
    
    # Create complete date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(date_range)
    
    # Fill missing values
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate(method='linear')
    
    df = df.reset_index().rename(columns={'index': date_column})
    
    return df


def get_expiry_date(contract_code: str) -> Optional[pd.Timestamp]:
    """
    Parse expiry date from futures contract code.
    
    Example: 'ZCZ24' -> December 2024
    
    Args:
        contract_code: Contract code (e.g., 'ZCZ24')
        
    Returns:
        Expiry date (3rd Friday of month)
    """
    if len(contract_code) < 4:
        return None
    
    # Month codes
    month_codes = {
        'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
        'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
    }
    
    try:
        month_code = contract_code[-3]
        year_code = contract_code[-2:]
        
        month = month_codes.get(month_code)
        if month is None:
            return None
        
        # Assume 20XX for 00-49, 19XX for 50-99
        year = 2000 + int(year_code) if int(year_code) < 50 else 1900 + int(year_code)
        
        # Third Friday of the month (typical expiry)
        first_day = pd.Timestamp(year=year, month=month, day=1)
        first_friday = first_day + pd.DateOffset(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + pd.DateOffset(weeks=2)
        
        return third_friday
        
    except Exception as e:
        logger.warning(f"Could not parse expiry from {contract_code}: {e}")
        return None


def days_until_expiry(contract_code: str, current_date: Optional[datetime] = None) -> Optional[int]:
    """
    Calculate days until contract expiry.
    
    Returns:
        Number of days until expiry
    """
    expiry = get_expiry_date(contract_code)
    if expiry is None:
        return None
    
    if current_date is None:
        current_date = datetime.now()
    
    current_date = parse_date(current_date)
    
    return (expiry - current_date).days


def is_within_trading_hours(dt: Optional[datetime] = None, 
                           market_open: str = '09:30',
                           market_close: str = '16:00') -> bool:
    """
    Check if given time is within trading hours.
    
    Args:
        dt: Datetime to check (default: now)
        market_open: Market open time (HH:MM)
        market_close: Market close time (HH:MM)
        
    Returns:
        True if within trading hours
    """
    if dt is None:
        dt = datetime.now()
    
    open_time = datetime.strptime(market_open, '%H:%M').time()
    close_time = datetime.strptime(market_close, '%H:%M').time()
    
    return open_time <= dt.time() <= close_time


def get_roll_date(contract_code: str, days_before_expiry: int = 5) -> pd.Timestamp:
    """
    Get recommended roll date (typically 5 days before expiry).
    
    Returns:
        Recommended roll date
    """
    expiry = get_expiry_date(contract_code)
    if expiry is None:
        raise ValueError(f"Cannot determine expiry for {contract_code}")
    
    roll_date = expiry - pd.Timedelta(days=days_before_expiry)
    return roll_date


def format_date_for_api(date, format: str = '%Y-%m-%d') -> str:
    """
    Format date for API calls.
    
    Args:
        date: Date to format
        format: Date format string
        
    Returns:
        Formatted date string
    """
    date = parse_date(date)
    return date.strftime(format)


def get_season(date, hemisphere: str = 'northern') -> str:
    """
    Get season for a given date.
    
    Args:
        date: Date
        hemisphere: 'northern' or 'southern'
        
    Returns:
        Season name
    """
    date = parse_date(date)
    month = date.month
    
    if hemisphere == 'northern':
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    else:  # southern hemisphere
        if month in [12, 1, 2]:
            return 'summer'
        elif month in [3, 4, 5]:
            return 'fall'
        elif month in [6, 7, 8]:
            return 'winter'
        else:
            return 'spring'