"""
Futures contract specifications.
"""

CONTRACT_SPECS = {
    'corn': {
        'exchange': 'CBOT',
        'symbol': 'ZC',
        'contract_months': ['H', 'K', 'N', 'U', 'Z'],  # Mar, May, Jul, Sep, Dec
        'month_codes': {
            'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12
        },
        'tick_size': 0.25,          # cents per bushel
        'tick_value': 12.50,        # $ per tick (5000 bushels)
        'point_value': 50,          # $ per cent
        'typical_volume_threshold': 10000,  # Liquidity filter
    },
    
    'soybeans': {
        'exchange': 'CBOT',
        'symbol': 'ZS',
        'contract_months': ['F', 'H', 'K', 'N', 'Q', 'U', 'X'],
        'month_codes': {
            'F': 1, 'H': 3, 'K': 5, 'N': 7, 'Q': 8, 'U': 9, 'X': 11
        },
        'tick_size': 0.25,
        'tick_value': 12.50,
        'point_value': 50,
        'typical_volume_threshold': 8000,
    },
    
    'coffee': {
        'exchange': 'ICE',
        'symbol': 'KC',
        'contract_months': ['H', 'K', 'N', 'U', 'Z'],
        'month_codes': {
            'H': 3, 'K': 5, 'N': 7, 'U': 9, 'Z': 12
        },
        'tick_size': 0.05,          # cents per pound
        'tick_value': 18.75,        # $ per tick (37500 lbs)
        'point_value': 375,
        'typical_volume_threshold': 5000,
    },
    
    'natural_gas': {
        'exchange': 'NYMEX',
        'symbol': 'NG',
        'contract_months': ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'],
        'month_codes': {
            'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
            'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
        },
        'tick_size': 0.001,         # $ per MMBtu
        'tick_value': 10,           # $ per tick (10000 MMBtu)
        'point_value': 10000,
        'typical_volume_threshold': 20000,
    }
}


def get_contract_spec(commodity: str) -> dict:
    """Get contract specifications."""
    if commodity not in CONTRACT_SPECS:
        raise ValueError(f"Contract spec for {commodity} not defined")
    return CONTRACT_SPECS[commodity]


def get_active_contract_month(commodity: str, stress_timing: str, current_date) -> str:
    """
    Select which contract month to trade based on stress timing.
    
    Rules:
    - Current season stress → new crop contract
    - Storage/deferred stress → 6-12 month out
    - Trend stress → front month
    """
    import pandas as pd
    from dateutil.relativedelta import relativedelta
    
    spec = CONTRACT_SPECS[commodity]
    current_date = pd.to_datetime(current_date)
    
    if stress_timing == 'current_season':
        # Trade the next harvest contract
        if commodity in ['corn', 'soybeans']:
            # Northern hemisphere: Dec contract for fall harvest
            target_month = 12 if current_date.month < 10 else 12  # Next year's Dec
            target_year = current_date.year if current_date.month < 10 else current_date.year + 1
        else:
            # Default: 3 months out
            target_date = current_date +relativedelta(months=3)
            target_month = target_date.month
            target_year = target_date.year
    
    elif stress_timing == 'storage':
        # 6 months deferred
        target_date = current_date + relativedelta(months=6)
        target_month = target_date.month
        target_year = target_date.year
    
    else:  # 'immediate'
        # Front month
        target_date = current_date + relativedelta(months=1)
        target_month = target_date.month
        target_year = target_date.year
    
    # Find nearest traded month
    month_codes = spec['month_codes']
    available_months = sorted(month_codes.values())
    
    # Find closest month
    closest_month = min(available_months, key=lambda x: abs(x - target_month))
    
    # Get month code
    month_code = [k for k, v in month_codes.items() if v == closest_month][0]
    
    return f"{spec['symbol']}{month_code}{str(target_year)[-2:]}"