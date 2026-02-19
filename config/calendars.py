"""
Growing season calendars.
Format: (start_month, start_day, end_month, end_day)
"""

GROWING_CALENDARS = {
    'corn': {
        'Northern_Hemisphere': {
            'planting': (4, 15, 6, 15),        # April 15 - June 15
            'emergence': (5, 1, 6, 30),
            'vegetative': (6, 1, 7, 31),
            'pollination': (7, 1, 8, 15),      # CRITICAL WINDOW
            'grain_fill': (8, 1, 9, 15),
            'maturation': (9, 1, 10, 31),
            'harvest': (10, 1, 11, 30),
        },
        'Southern_Hemisphere': {
            'planting': (10, 1, 12, 15),
            'pollination': (1, 1, 2, 15),      # CRITICAL WINDOW
            'harvest': (4, 1, 6, 30),
        }
    },
    
    'soybeans': {
        'Northern_Hemisphere': {
            'planting': (5, 1, 6, 30),
            'flowering': (7, 1, 8, 31),        # CRITICAL WINDOW
            'pod_fill': (8, 15, 10, 15),
            'harvest': (10, 1, 11, 30),
        },
        'Southern_Hemisphere': {
            'planting': (11, 1, 12, 31),
            'flowering': (1, 1, 3, 15),        # CRITICAL WINDOW
            'harvest': (4, 1, 6, 30),
        }
    },
    
    'coffee': {
        'Brazil': {
            'flowering': (9, 1, 11, 30),       # Spring
            'fruit_development': (12, 1, 3, 31),
            'ripening': (4, 1, 6, 30),         # CRITICAL - frost risk
            'harvest': (5, 1, 9, 30),
        },
        'Colombia': {
            # Two harvests per year
            'main_harvest': (9, 1, 12, 31),
            'mitaca_harvest': (4, 1, 6, 30),
        },
        'Vietnam': {
            'flowering': (2, 1, 4, 30),
            'harvest': (10, 1, 12, 31),
        }
    },
    
    'natural_gas': {
        'Global': {
            'winter_heating': (11, 1, 3, 31),
            'summer_cooling': (6, 1, 9, 30),
        }
    }
}


# Seasonal sensitivity weights (0.0 - 1.0)
# Defines how much weather stress matters in each growth stage
SEASONAL_SENSITIVITY = {
    'corn': {
        'planting': 0.3,
        'emergence': 0.4,
        'vegetative': 0.6,
        'pollination': 1.0,      # Maximum sensitivity
        'grain_fill': 0.8,
        'maturation': 0.4,
        'harvest': 0.2,
    },
    'soybeans': {
        'planting': 0.3,
        'flowering': 1.0,
        'pod_fill': 0.9,
        'harvest': 0.2,
    },
    'coffee': {
        'flowering': 0.7,
        'fruit_development': 0.8,
        'ripening': 1.0,         # Frost critical
        'harvest': 0.3,
    }
}


def get_current_stage(commodity: str, region: str, date) -> str:
    """
    Determine current growth stage for a commodity/region/date.
    """
    import pandas as pd
    
    # Determine hemisphere
    from config.regions import COMMODITY_REGIONS
    lat = COMMODITY_REGIONS[commodity][region]['lat']
    hemisphere = 'Northern_Hemisphere' if lat > 0 else 'Southern_Hemisphere'
    
    if commodity not in GROWING_CALENDARS:
        return 'unknown'
    
    # Try hemisphere first, then region directly
    calendar = GROWING_CALENDARS[commodity].get(hemisphere)
    if calendar is None:
        # Try region name directly (for special cases like Brazil, Colombia, Vietnam)
        calendar = GROWING_CALENDARS[commodity].get(region)
    
    if calendar is None:
        return 'unknown'
    
    date = pd.to_datetime(date)
    month, day = date.month, date.day
    
    for stage, (start_m, start_d, end_m, end_d) in calendar.items():
        start_date = pd.Timestamp(year=date.year, month=start_m, day=start_d)
        end_date = pd.Timestamp(year=date.year, month=end_m, day=end_d)
        
        if start_date <= date <= end_date:
            return stage
    
    return 'dormant'



def get_sensitivity_weight(commodity: str, stage: str) -> float:
    """Get sensitivity weight for a growth stage."""
    if commodity not in SEASONAL_SENSITIVITY:
        return 0.5  # Default
    
    return SEASONAL_SENSITIVITY[commodity].get(stage, 0.5)