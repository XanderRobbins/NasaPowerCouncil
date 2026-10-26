"""Production regions per commodity: a representative lat/lon and a production weight.

NASA POWER is queried at each point, and the regional stress features are combined with
``weight`` in ``features/regional_aggregator.py``. The aggregator applies the weights as
given and does not renormalise them, so they should sum to 1 for the features to be on the
same scale across commodities (coffee sums to 0.8 because a third region was commented out).
"""

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
        #'Espirito_Santo': {'lat': -20.0, 'lon': -41.0, 'weight': 0.20},
    },

    'cocoa': {
        'Ivory_Coast': {'lat': 6.8, 'lon': -5.3, 'weight': 0.45},
        'Ghana': {'lat': 7.9, 'lon': -1.0, 'weight': 0.35},
        'Nigeria': {'lat': 6.5, 'lon': 3.4, 'weight': 0.20},
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
