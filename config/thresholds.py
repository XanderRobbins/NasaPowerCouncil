"""Weather-stress thresholds per commodity.

Three numbers per crop drive the stress features (``features/stress_calculator.py``):

* ``temp_critical_high`` (deg C): a day with max temperature above this is a heat-stress day.
* ``temp_critical_low``  (deg C): a day with min temperature below this is a cold-stress day.
* ``precip_threshold_dry`` (mm): 7-day rainfall below this marks a dry-stress day.

Only commodities that also have production regions in ``config/regions.py`` are listed,
because a commodity needs both to be usable.
"""

CRITICAL_THRESHOLDS = {
    'corn': {'temp_critical_high': 32, 'temp_critical_low': 10, 'precip_threshold_dry': 25},
    'soybeans': {'temp_critical_high': 35, 'temp_critical_low': 15, 'precip_threshold_dry': 30},
    'wheat': {'temp_critical_high': 34, 'temp_critical_low': -5, 'precip_threshold_dry': 20},
    'cotton': {'temp_critical_high': 38, 'temp_critical_low': 15, 'precip_threshold_dry': 15},
    'coffee': {'temp_critical_high': 30, 'temp_critical_low': 5, 'precip_threshold_dry': 30},
    'sugar': {'temp_critical_high': 38, 'temp_critical_low': 10, 'precip_threshold_dry': 25},
    'live_cattle': {'temp_critical_high': 35, 'temp_critical_low': -10, 'precip_threshold_dry': 20},
    'lean_hogs': {'temp_critical_high': 32, 'temp_critical_low': -5, 'precip_threshold_dry': 20},
    'cocoa': {'temp_critical_high': 32, 'temp_critical_low': 18, 'precip_threshold_dry': 30},
}


def get_thresholds(commodity: str) -> dict:
    """Return the stress thresholds for ``commodity`` (raises if it is not defined)."""
    if commodity not in CRITICAL_THRESHOLDS:
        raise ValueError(f"Thresholds for {commodity} not defined")
    return CRITICAL_THRESHOLDS[commodity]
