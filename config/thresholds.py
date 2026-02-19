"""
Critical weather thresholds per commodity.
Based on crop science literature.
"""

CRITICAL_THRESHOLDS = {
    'corn': {
        'temp_critical_high': 32,      # °C - Above this, heat stress
        'temp_critical_low': 10,       # °C - Below this, cold stress
        'precip_threshold_dry': 25,    # mm/week - Below this, dry stress
        'precip_threshold_wet': 150,   # mm/week - Above this, flood risk
        'gdd_base': 10,                # °C - Base for Growing Degree Days
        'gdd_optimal': 30,             # °C - Optimal temperature
    },
    
    'soybeans': {
        'temp_critical_high': 35,
        'temp_critical_low': 15,
        'precip_threshold_dry': 30,
        'precip_threshold_wet': 130,
        'gdd_base': 10,
        'gdd_optimal': 30,
    },
    
    'coffee': {
        'temp_critical_high': 30,
        'temp_critical_low': 15,
        'precip_threshold_dry': 50,    # mm/week
        'precip_threshold_wet': 200,
        'frost_threshold': 0,          # °C - Frost damage
    },
    
    'natural_gas': {
        'heating_threshold': 18,       # °C - Below this, heating demand
        'cooling_threshold': 22,       # °C - Above this, cooling demand
    }
}


def get_thresholds(commodity: str) -> dict:
    """Get thresholds for a specific commodity."""
    if commodity not in CRITICAL_THRESHOLDS:
        raise ValueError(f"Thresholds for {commodity} not defined")
    return CRITICAL_THRESHOLDS[commodity]