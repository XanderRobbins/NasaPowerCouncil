"""
Simple data validation utilities.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from loguru import logger

class DataValidator:
    """Basic data quality checks."""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[bool, List[str]]:
        """
        Check if missing values exceed threshold.
        
        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        missing_pct = df.isnull().sum() / len(df)
        
        for col, pct in missing_pct.items():
            if pct > threshold:
                issues.append(f"Column '{col}' has {pct:.1%} missing values")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def check_date_continuity(df: pd.DataFrame, max_gap_days: int = 7) -> Tuple[bool, List[str]]:
        """Check for large gaps in date sequence."""
        if 'date' not in df.columns:
            return True, []
        
        issues = []
        dates = pd.to_datetime(df['date']).sort_values()
        gaps = dates.diff()
        
        large_gaps = gaps[gaps > pd.Timedelta(days=max_gap_days)]
        if len(large_gaps) > 0:
            issues.append(f"Found {len(large_gaps)} date gaps > {max_gap_days} days")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> Tuple[bool, List[str]]:
        """Run all validation checks."""
        all_issues = []
        
        # Check 1: Empty
        if df.empty:
            return False, [f"{name} is empty"]
        
        # Check 2: Missing values
        valid, issues = DataValidator.check_missing_values(df)
        all_issues.extend(issues)
        
        # Check 3: Date continuity
        valid, issues = DataValidator.check_date_continuity(df)
        all_issues.extend(issues)
        
        is_valid = len(all_issues) == 0
        
        if not is_valid:
            logger.warning(f"Validation issues in {name}: {all_issues}")
        else:
            logger.debug(f"✓ {name} passed validation")
        
        return is_valid, all_issues


def validate_data(df: pd.DataFrame, name: str = "DataFrame") -> Tuple[bool, List[str]]:
    """Convenience function."""
    validator = DataValidator()
    return validator.validate_dataframe(df, name)