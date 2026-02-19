"""
Production region definitions and weights.
Weights should sum to 1.0 per commodity.
"""

COMMODITY_REGIONS = {
    'corn': {
        'Iowa': {
            'lat': 42.0,
            'lon': -93.5,
            'weight': 0.35,
            'production_mt': 14_500_000  # metric tons
        },
        'Illinois': {
            'lat': 40.0,
            'lon': -89.0,
            'weight': 0.30,
            'production_mt': 12_400_000
        },
        'Nebraska': {
            'lat': 41.5,
            'lon': -99.8,
            'weight': 0.20,
            'production_mt': 8_300_000
        },
        'Argentina_Pampas': {
            'lat': -34.0,
            'lon': -61.0,
            'weight': 0.10,
            'production_mt': 4_100_000
        },
        'Brazil_MatoGrosso': {
            'lat': -12.5,
            'lon': -55.5,
            'weight': 0.05,
            'production_mt': 2_000_000
        }
    },
    
    'soybeans': {
        'Iowa': {
            'lat': 42.0,
            'lon': -93.5,
            'weight': 0.25,
            'production_mt': 13_800_000
        },
        'Illinois': {
            'lat': 40.0,
            'lon': -89.0,
            'weight': 0.25,
            'production_mt': 13_500_000
        },
        'Brazil_MatoGrosso': {
            'lat': -12.5,
            'lon': -55.5,
            'weight': 0.30,
            'production_mt': 16_500_000
        },
        'Argentina_Pampas': {
            'lat': -34.0,
            'lon': -61.0,
            'weight': 0.20,
            'production_mt': 11_000_000
        }
    },
    
    'coffee': {
        'Brazil_MinasGerais': {
            'lat': -19.9,
            'lon': -44.0,
            'weight': 0.50,
            'production_mt': 1_500_000
        },
        'Colombia': {
            'lat': 4.5,
            'lon': -75.5,
            'weight': 0.30,
            'production_mt': 900_000
        },
        'Vietnam_CentralHighlands': {
            'lat': 12.5,
            'lon': 108.0,
            'weight': 0.20,
            'production_mt': 600_000
        }
    },
    
    'natural_gas': {
        # Population-weighted heating/cooling demand regions
        'US_Northeast': {
            'lat': 40.7,
            'lon': -74.0,
            'weight': 0.35,
            'population': 56_000_000
        },
        'US_Midwest': {
            'lat': 41.9,
            'lon': -87.6,
            'weight': 0.30,
            'population': 68_000_000
        },
        'US_South': {
            'lat': 29.7,
            'lon': -95.4,
            'weight': 0.35,
            'population': 125_000_000
        }
    }
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