"""
Configuration module.
"""

from config.settings import (
    PROJECT_ROOT,
    DATA_STORAGE_PATH,
    TARGET_PORTFOLIO_VOL,
    MAX_SINGLE_COMMODITY,
    MAX_DRAWDOWN_THRESHOLD,
    PHASE_1_COMMODITIES,
)

from config.regions import (
    COMMODITY_REGIONS,
    get_commodity_regions,
    validate_weights,
)

from config.thresholds import (
    CRITICAL_THRESHOLDS,
    get_thresholds,
)

from config.calendars import (
    GROWING_CALENDARS,
    SEASONAL_SENSITIVITY,
    get_current_stage,
    get_sensitivity_weight,
)

from config.contracts import (
    CONTRACT_SPECS,
    get_contract_spec,
    get_active_contract_month,
)

__all__ = [
    # Settings
    'PROJECT_ROOT',
    'DATA_STORAGE_PATH',
    'TARGET_PORTFOLIO_VOL',
    'MAX_SINGLE_COMMODITY',
    'MAX_DRAWDOWN_THRESHOLD',
    'PHASE_1_COMMODITIES',
    
    # Regions
    'COMMODITY_REGIONS',
    'get_commodity_regions',
    'validate_weights',
    
    # Thresholds
    'CRITICAL_THRESHOLDS',
    'get_thresholds',
    
    # Calendars
    'GROWING_CALENDARS',
    'SEASONAL_SENSITIVITY',
    'get_current_stage',
    'get_sensitivity_weight',
    
    # Contracts
    'CONTRACT_SPECS',
    'get_contract_spec',
    'get_active_contract_month',
]