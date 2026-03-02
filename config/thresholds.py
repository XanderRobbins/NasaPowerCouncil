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
        'wheat': {
        'temp_critical_high': 34,
        'temp_critical_low': -5,
        'temp_optimal': 18,
        'precip_threshold_dry': 20,
        'precip_threshold_wet': 120,
        'gdd_base': 0,
        'gdd_optimal': 25,
    },

    'cotton': {
        'temp_critical_high': 38,
        'temp_critical_low': 15,
        'temp_optimal': 28,
        'precip_threshold_dry': 15,
        'precip_threshold_wet': 100,
        'gdd_base': 15,
        'gdd_optimal': 30,
    },

    'coffee': {
        'temp_critical_high': 30,
        'temp_critical_low': 5,
        'temp_optimal': 20,
        'precip_threshold_dry': 30,
        'precip_threshold_wet': 200,
        'gdd_base': 10,
        'gdd_optimal': 25,
    },

    'sugar': {
        'temp_critical_high': 38,
        'temp_critical_low': 10,
        'temp_optimal': 28,
        'precip_threshold_dry': 25,
        'precip_threshold_wet': 180,
        'gdd_base': 10,
        'gdd_optimal': 32,
    },

    'live_cattle': {
        'temp_critical_high': 35,
        'temp_critical_low': -10,
        'temp_optimal': 20,
        'precip_threshold_dry': 20,
        'precip_threshold_wet': 120,
        'gdd_base': 5,
        'gdd_optimal': 25,
    },

    'lean_hogs': {
        'temp_critical_high': 32,
        'temp_critical_low': -5,
        'temp_optimal': 18,
        'precip_threshold_dry': 20,
        'precip_threshold_wet': 120,
        'gdd_base': 5,
        'gdd_optimal': 25,
    },

    'soybean_oil': {
        'temp_critical_high': 35,
        'temp_critical_low': 15,
        'temp_optimal': 25,
        'precip_threshold_dry': 30,
        'precip_threshold_wet': 130,
        'gdd_base': 10,
        'gdd_optimal': 30,
    },

    'soybean_meal': {
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