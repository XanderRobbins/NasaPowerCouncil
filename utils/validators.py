"""
Input validation utilities.
"""
import pandas as pd
import numpy as np
from typing import Any, List, Union, Optional
from datetime import datetime


class ValidationError(Exception):
    """Custom validation error."""
    pass


def validate_date(date: Any, allow_none: bool = False) -> pd.Timestamp:
    """
    Validate and convert date input.
    
    Args:
        date: Date input
        allow_none: Whether to allow None values
        
    Returns:
        pd.Timestamp
        
    Raises:
        ValidationError if invalid
    """
    if date is None and allow_none:
        return None
    
    try:
        return pd.to_datetime(date)
    except Exception as e:
        raise ValidationError(f"Invalid date format: {date}. Error: {e}")


def validate_date_range(start_date: Any, end_date: Any) -> tuple:
    """
    Validate date range.
    
    Returns:
        (start_date, end_date) as pd.Timestamps
        
    Raises:
        ValidationError if invalid
    """
    start = validate_date(start_date)
    end = validate_date(end_date)
    
    if start > end:
        raise ValidationError(f"Start date {start} is after end date {end}")
    
    return start, end


def validate_number(value: Any, 
                   min_value: Optional[float] = None,
                   max_value: Optional[float] = None,
                   allow_none: bool = False,
                   name: str = "value") -> float:
    """
    Validate numeric input.
    
    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether to allow None
        name: Name for error messages
        
    Returns:
        Validated float
        
    Raises:
        ValidationError if invalid
    """
    if value is None and allow_none:
        return None
    
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{name} must be finite, got {value}")
    
    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")
    
    return value


def validate_percentage(value: Any, name: str = "percentage") -> float:
    """
    Validate percentage (0-100).
    
    Returns:
        Validated percentage
    """
    return validate_number(value, min_value=0, max_value=100, name=name)


def validate_probability(value: Any, name: str = "probability") -> float:
    """
    Validate probability (0-1).
    
    Returns:
        Validated probability
    """
    return validate_number(value, min_value=0, max_value=1, name=name)


def validate_string(value: Any, 
                   allowed_values: Optional[List[str]] = None,
                   allow_none: bool = False,
                   name: str = "value") -> str:
    """
    Validate string input.
    
    Args:
        value: Value to validate
        allowed_values: List of allowed values
        allow_none: Whether to allow None
        name: Name for error messages
        
    Returns:
        Validated string
        
    Raises:
        ValidationError if invalid
    """
    if value is None and allow_none:
        return None
    
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be string, got {type(value)}")
    
    if allowed_values and value not in allowed_values:
        raise ValidationError(f"{name} must be one of {allowed_values}, got {value}")
    
    return value


def validate_dataframe(df: Any, 
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 0,
                      name: str = "DataFrame") -> pd.DataFrame:
    """
    Validate DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows
        name: Name for error messages
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValidationError if invalid
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} must be pandas DataFrame, got {type(df)}")
    
    if df.empty and min_rows > 0:
        raise ValidationError(f"{name} is empty")
    
    if len(df) < min_rows:
        raise ValidationError(f"{name} must have at least {min_rows} rows, got {len(df)}")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValidationError(f"{name} missing required columns: {missing_cols}")
    
    return df


def validate_array(arr: Any,
                  shape: Optional[tuple] = None,
                  dtype: Optional[type] = None,
                  allow_nan: bool = False,
                  name: str = "array") -> np.ndarray:
    """
    Validate numpy array.
    
    Args:
        arr: Array to validate
        shape: Expected shape
        dtype: Expected dtype
        allow_nan: Whether to allow NaN values
        name: Name for error messages
        
    Returns:
        Validated array
        
    Raises:
        ValidationError if invalid
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr)
        except Exception as e:
            raise ValidationError(f"Cannot convert {name} to numpy array: {e}")
    
    if shape is not None and arr.shape != shape:
        raise ValidationError(f"{name} shape must be {shape}, got {arr.shape}")
    
    if dtype is not None and arr.dtype != dtype:
        raise ValidationError(f"{name} dtype must be {dtype}, got {arr.dtype}")
    
    if not allow_nan and np.isnan(arr).any():
        raise ValidationError(f"{name} contains NaN values")
    
    return arr


def validate_dict(d: Any,
                 required_keys: Optional[List[str]] = None,
                 allow_empty: bool = True,
                 name: str = "dict") -> dict:
    """
    Validate dictionary.
    
    Args:
        d: Dictionary to validate
        required_keys: List of required keys
        allow_empty: Whether to allow empty dict
        name: Name for error messages
        
    Returns:
        Validated dict
        
    Raises:
        ValidationError if invalid
    """
    if not isinstance(d, dict):
        raise ValidationError(f"{name} must be dict, got {type(d)}")
    
    if not allow_empty and not d:
        raise ValidationError(f"{name} cannot be empty")
    
    if required_keys:
        missing_keys = set(required_keys) - set(d.keys())
        if missing_keys:
            raise ValidationError(f"{name} missing required keys: {missing_keys}")
    
    return d


def validate_commodity(commodity: str) -> str:
    """
    Validate commodity name.
    
    Returns:
        Validated commodity name (lowercase)
    """
    from config.regions import COMMODITY_REGIONS
    
    commodity = commodity.lower()
    
    if commodity not in COMMODITY_REGIONS:
        raise ValidationError(f"Unknown commodity: {commodity}. "
                            f"Available: {list(COMMODITY_REGIONS.keys())}")
    
    return commodity


def validate_signal(signal: float, name: str = "signal") -> float:
    """
    Validate signal value (typically -3 to +3).
    
    Returns:
        Validated signal
    """
    signal = validate_number(signal, name=name)
    
    if abs(signal) > 10:
        raise ValidationError(f"{name} seems unusually large: {signal}")
    
    return signal


def validate_position(position: float, 
                     max_position: float = 1.0,
                     name: str = "position") -> float:
    """
    Validate position size.
    
    Returns:
        Validated position
    """
    position = validate_number(position, name=name)
    
    if abs(position) > max_position:
        raise ValidationError(f"{name} exceeds maximum: {abs(position)} > {max_position}")
    
    return position


def validate_config(config: dict) -> dict:
    """
    Validate configuration dictionary.
    
    Returns:
        Validated config
    """
    required_keys = ['commodities', 'start_date', 'end_date']
    config = validate_dict(config, required_keys=required_keys, name="config")
    
    # Validate dates
    validate_date_range(config['start_date'], config['end_date'])
    
    # Validate commodities
    if not isinstance(config['commodities'], list) or not config['commodities']:
        raise ValidationError("config['commodities'] must be non-empty list")
    
    for commodity in config['commodities']:
        validate_commodity(commodity)
    
    return config


class Validator:
    """
    Chainable validator class.
    
    Example:
        value = Validator(x).is_number().in_range(0, 100).get()
    """
    
    def __init__(self, value: Any, name: str = "value"):
        self.value = value
        self.name = name
        
    def is_number(self):
        """Validate that value is numeric."""
        self.value = validate_number(self.value, name=self.name)
        return self
    
    def in_range(self, min_val: float, max_val: float):
        """Validate that value is in range."""
        self.value = validate_number(self.value, min_value=min_val, max_value=max_val, name=self.name)
        return self
    
    def is_positive(self):
        """Validate that value is positive."""
        self.value = validate_number(self.value, min_value=0, name=self.name)
        return self
    
    def is_string(self, allowed_values: Optional[List[str]] = None):
        """Validate that value is string."""
        self.value = validate_string(self.value, allowed_values=allowed_values, name=self.name)
        return self
    
    def is_dataframe(self, required_columns: Optional[List[str]] = None):
        """Validate that value is DataFrame."""
        self.value = validate_dataframe(self.value, required_columns=required_columns, name=self.name)
        return self
    
    def not_empty(self):
        """Validate that value is not empty."""
        if isinstance(self.value, (list, dict, str, pd.DataFrame)):
            if not self.value or (isinstance(self.value, pd.DataFrame) and self.value.empty):
                raise ValidationError(f"{self.name} cannot be empty")
        return self
    
    def get(self):
        """Get validated value."""
        return self.value