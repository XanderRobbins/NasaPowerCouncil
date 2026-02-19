"""
Data quality validation and checks.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger


class DataValidator:
    """
    Validate data quality.
    
    Checks:
    - Missing values
    - Outliers
    - Duplicates
    - Data types
    - Date continuity
    """
    
    def __init__(self, strict: bool = False):
        self.strict = strict  # If True, raise errors; if False, log warnings
        self.validation_results = []
        
    def validate_dataframe(self, df: pd.DataFrame, name: str = "DataFrame") -> Tuple[bool, List[str]]:
        """
        Run all validation checks on a DataFrame.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        
        # Check 1: Empty DataFrame
        if df.empty:
            issues.append(f"{name} is empty")
            return False, issues
        
        # Check 2: Missing values
        missing_issues = self._check_missing_values(df, name)
        issues.extend(missing_issues)
        
        # Check 3: Duplicates
        duplicate_issues = self._check_duplicates(df, name)
        issues.extend(duplicate_issues)
        
        # Check 4: Outliers (for numeric columns)
        outlier_issues = self._check_outliers(df, name)
        issues.extend(outlier_issues)
        
        # Check 5: Date continuity (if date column exists)
        if 'date' in df.columns:
            date_issues = self._check_date_continuity(df, name)
            issues.extend(date_issues)
        
        # Check 6: Data types
        dtype_issues = self._check_data_types(df, name)
        issues.extend(dtype_issues)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Validation issues found in {name}:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.debug(f"✓ {name} passed all validation checks")
        
        return is_valid, issues
    
    def _check_missing_values(self, df: pd.DataFrame, name: str) -> List[str]:
        """Check for excessive missing values."""
        issues = []
        
        missing_pct = df.isnull().sum() / len(df) * 100
        
        for col, pct in missing_pct.items():
            if pct > 50:
                issues.append(f"{name}.{col}: {pct:.1f}% missing (> 50% threshold)")
            elif pct > 10:
                logger.debug(f"{name}.{col}: {pct:.1f}% missing")
        
        return issues
    
    def _check_duplicates(self, df: pd.DataFrame, name: str) -> List[str]:
        """Check for duplicate rows."""
        issues = []
        
        # Check for full row duplicates
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            issues.append(f"{name}: {n_duplicates} duplicate rows found")
        
        # Check for duplicate dates (if date column exists)
        if 'date' in df.columns:
            n_date_dupes = df['date'].duplicated().sum()
            if n_date_dupes > 0:
                issues.append(f"{name}: {n_date_dupes} duplicate dates found")
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame, name: str, z_threshold: float = 5.0) -> List[str]:
        """Check for extreme outliers using Z-score."""
        issues = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in ['date', 'year', 'month', 'day']:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            z_scores = np.abs((values - values.mean()) / values.std())
            n_outliers = (z_scores > z_threshold).sum()
            
            if n_outliers > 0:
                pct_outliers = n_outliers / len(values) * 100
                if pct_outliers > 5:  # More than 5% outliers
                    issues.append(f"{name}.{col}: {n_outliers} extreme outliers ({pct_outliers:.1f}%)")
        
        return issues
    
    def _check_date_continuity(self, df: pd.DataFrame, name: str) -> List[str]:
        """Check for gaps in date sequence."""
        issues = []
        
        dates = pd.to_datetime(df['date']).sort_values()
        date_diffs = dates.diff()
        
        # Check for large gaps (> 7 days)
        large_gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
        
        if len(large_gaps) > 0:
            issues.append(f"{name}: {len(large_gaps)} date gaps > 7 days found")
        
        return issues
    
    def _check_data_types(self, df: pd.DataFrame, name: str) -> List[str]:
        """Check for unexpected data types."""
        issues = []
        
        # Expected types for common columns
        expected_types = {
            'date': ['datetime64[ns]', 'object'],
            'temperature': ['float64', 'int64'],
            'precipitation': ['float64', 'int64'],
            'price': ['float64'],
            'volume': ['int64', 'float64']
        }
        
        for col, expected in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type not in expected:
                    issues.append(f"{name}.{col}: unexpected type {actual_type} (expected {expected})")
        
        return issues
    
    def validate_climate_data(self, df: pd.DataFrame, region: str) -> Tuple[bool, List[str]]:
        """Specialized validation for climate data."""
        issues = []
        
        # Required columns
        required_cols = ['date', 'temp_max', 'temp_min', 'precipitation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Physical constraints
        if 'temp_max' in df.columns:
            invalid_temps = df[(df['temp_max'] < -50) | (df['temp_max'] > 60)]
            if len(invalid_temps) > 0:
                issues.append(f"Invalid temperatures: {len(invalid_temps)} records outside [-50, 60]°C")
        
        if 'precipitation' in df.columns:
            invalid_precip = df[df['precipitation'] < 0]
            if len(invalid_precip) > 0:
                issues.append(f"Negative precipitation: {len(invalid_precip)} records")
        
        # General validation
        is_valid, general_issues = self.validate_dataframe(df, f"Climate-{region}")
        issues.extend(general_issues)
        
        return len(issues) == 0, issues
    
    def validate_price_data(self, df: pd.DataFrame, commodity: str) -> Tuple[bool, List[str]]:
        """Specialized validation for price data."""
        issues = []
        
        # Required columns
        required_cols = ['date', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Price constraints
        if 'close' in df.columns:
            if (df['close'] <= 0).any():
                issues.append("Non-positive prices detected")
            
            # Check for extreme price jumps (> 20% in one day)
            returns = df['close'].pct_change()
            extreme_moves = returns[returns.abs() > 0.20]
            if len(extreme_moves) > 5:  # Allow a few
                issues.append(f"Excessive extreme price moves: {len(extreme_moves)} days > 20%")
        
        # General validation
        is_valid, general_issues = self.validate_dataframe(df, f"Price-{commodity}")
        issues.extend(general_issues)
        
        return len(issues) == 0, issues


def validate_data(df: pd.DataFrame, data_type: str = 'general', **kwargs) -> Tuple[bool, List[str]]:
    """
    Convenience function to validate data.
    
    Args:
        df: DataFrame to validate
        data_type: 'general', 'climate', or 'price'
        **kwargs: Additional arguments (e.g., region, commodity)
    """
    validator = DataValidator()
    
    if data_type == 'climate':
        return validator.validate_climate_data(df, kwargs.get('region', 'unknown'))
    elif data_type == 'price':
        return validator.validate_price_data(df, kwargs.get('commodity', 'unknown'))
    else:
        return validator.validate_dataframe(df, kwargs.get('name', 'DataFrame'))