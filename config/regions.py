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
        'wheat': {
        'Kansas': {'lat': 38.5, 'lon': -98.0, 'weight': 0.40},
        'Oklahoma': {'lat': 35.5, 'lon': -97.5, 'weight': 0.30},
        'Texas': {'lat': 33.0, 'lon': -97.0, 'weight': 0.30},
    },

    'cotton': {
        'Texas': {'lat': 33.0, 'lon': -101.0, 'weight': 0.45},
        'Georgia': {'lat': 32.0, 'lon': -83.5, 'weight': 0.30},
        'Mississippi': {'lat': 32.5, 'lon': -89.5, 'weight': 0.25},
    },

    'coffee': {
        'Minas_Gerais': {'lat': -19.0, 'lon': -44.0, 'weight': 0.45},
        'Sao_Paulo': {'lat': -22.0, 'lon': -47.0, 'weight': 0.35},
        'Espirito_Santo': {'lat': -20.0, 'lon': -41.0, 'weight': 0.20},
    },

    'sugar': {
        'Sao_Paulo_Sugar': {'lat': -21.5, 'lon': -48.0, 'weight': 0.50},
        'Louisiana': {'lat': 30.0, 'lon': -91.5, 'weight': 0.25},
        'Florida': {'lat': 26.5, 'lon': -80.5, 'weight': 0.25},
    },

    'live_cattle': {
        'Texas_Cattle': {'lat': 31.5, 'lon': -99.0, 'weight': 0.40},
        'Kansas_Cattle': {'lat': 38.0, 'lon': -98.5, 'weight': 0.35},
        'Nebraska_Cattle': {'lat': 41.5, 'lon': -99.8, 'weight': 0.25},
    },

    'lean_hogs': {
        'Iowa_Hogs': {'lat': 42.0, 'lon': -93.5, 'weight': 0.40},
        'Minnesota_Hogs': {'lat': 44.0, 'lon': -94.0, 'weight': 0.35},
        'Illinois_Hogs': {'lat': 40.0, 'lon': -89.0, 'weight': 0.25},
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