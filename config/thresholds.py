"""
Critical weather thresholds per commodity.
"""

CRITICAL_THRESHOLDS = {
    'corn': {
        'temp_critical_high': 32,      # °C - Heat stress
        'temp_critical_low': 10,       # °C - Cold stress
        'temp_optimal': 21,            # °C - Optimal growth
        'precip_threshold_dry': 25,    # mm/week - Dry stress (CONSISTENT NAME)
        'precip_threshold_wet': 150,   # mm/week - Flood risk
        'gdd_base': 10,                # °C - Growing degree day base
        'gdd_optimal': 30,             # °C - Optimal temperature
    },
    
    'soybeans': {
        'temp_critical_high': 35,
        'temp_critical_low': 15,
        'temp_optimal': 25,
        'precip_threshold_dry': 30,
        'precip_threshold_wet': 130,
        'gdd_base': 10,
        'gdd_optimal': 30,
    },
}

def get_thresholds(commodity: str) -> dict:
    """Get thresholds for a specific commodity."""
    if commodity not in CRITICAL_THRESHOLDS:
        raise ValueError(f"Thresholds for {commodity} not defined")
    return CRITICAL_THRESHOLDS[commodity]