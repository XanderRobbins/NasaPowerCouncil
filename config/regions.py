"""Production region definitions and weights."""

COMMODITY_REGIONS = {
    'corn': {
        'Iowa': {
            'lat': 42.0,
            'lon': -93.5,
            'weight': 0.35,
        },
        'Illinois': {
            'lat': 40.0,
            'lon': -89.0,
            'weight': 0.30,
        },
        'Nebraska': {
            'lat': 41.5,
            'lon': -99.8,
            'weight': 0.20,
        },
        'Indiana': {
            'lat': 40.3,
            'lon': -86.1,
            'weight': 0.15,
        },
    },
    'soybeans': {
        'Iowa': {
            'lat': 42.0,
            'lon': -93.5,
            'weight': 0.30,
        },
        'Illinois': {
            'lat': 40.0,
            'lon': -89.0,
            'weight': 0.30,
        },
        'Minnesota': {
            'lat': 45.0,
            'lon': -94.0,
            'weight': 0.20,
        },
        'Indiana': {
            'lat': 40.3,
            'lon': -86.1,
            'weight': 0.20,
        },
    },
}

def get_commodity_regions(commodity: str) -> dict:
    """Get regions for a specific commodity."""
    if commodity not in COMMODITY_REGIONS:
        raise ValueError(f"Commodity {commodity} not defined")
    return COMMODITY_REGIONS[commodity]

def validate_weights(commodity: str) -> bool:
    """Ensure weights sum to 1.0."""
    regions = COMMODITY_REGIONS[commodity]
    total_weight = sum(r['weight'] for r in regions.values())
    return abs(total_weight - 1.0) < 0.01