"""Global settings and configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_STORAGE_PATH = Path(os.getenv('DATA_STORAGE_PATH', PROJECT_ROOT / 'data_storage'))
RAW_DATA_PATH = DATA_STORAGE_PATH / 'raw'
PROCESSED_DATA_PATH = DATA_STORAGE_PATH / 'processed'
MODEL_PATH = DATA_STORAGE_PATH / 'models'
RESULTS_PATH = DATA_STORAGE_PATH / 'results'
LOG_PATH = DATA_STORAGE_PATH / 'logs'

# Create directories
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, RESULTS_PATH, LOG_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# API Settings
NASA_POWER_BASE_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'

# Strategy Settings
STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'directional')  # 'directional' or 'magnitude'
MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', 0.01))
FIXED_POSITION_SIZE = float(os.getenv('FIXED_POSITION_SIZE', 0.5))
ONLY_TRADE_GROWING_SEASON = os.getenv('ONLY_TRADE_GROWING_SEASON', 'true').lower() == 'true'

# Model Parameters
TRAIN_WINDOW_YEARS = int(os.getenv('TRAIN_WINDOW_YEARS', 3))
FORWARD_RETURN_DAYS = int(os.getenv('FORWARD_RETURN_DAYS', 10))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', 15))
RIDGE_ALPHA = float(os.getenv('RIDGE_ALPHA', 1.0))

# Risk Management
TARGET_PORTFOLIO_VOL = float(os.getenv('TARGET_PORTFOLIO_VOL', 0.99))
MAX_SINGLE_POSITION = float(os.getenv('MAX_SINGLE_POSITION', 1.0))
SIGNAL_CAP = float(os.getenv('SIGNAL_CAP', 90.0))

# Backtest Settings
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2010-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2023-01-01')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1_000_000))

# Commodities
PHASE_1_COMMODITIES = ['corn']

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO') or 'INFO'

<<<<<<< HEAD
# Per-commodity trade months (loaded from .env)
COMMODITY_TRADE_MONTHS = {}
for commodity in PHASE_1_COMMODITIES:
    months_str = os.getenv(f'{commodity.upper()}_TRADE_MONTHS', '4,5,6')
    try:
        COMMODITY_TRADE_MONTHS[commodity] = [int(m.strip()) for m in months_str.split(',')]
    except ValueError:
        COMMODITY_TRADE_MONTHS[commodity] = [4, 5, 6]  # Fallback

# Per-commodity volatility regime thresholds
COMMODITY_VOL_THRESHOLDS = {}
for commodity in PHASE_1_COMMODITIES:
    COMMODITY_VOL_THRESHOLDS[commodity] = float(os.getenv(f'{commodity.upper()}_VOL_REGIME_THRESHOLD', 0.3))

# Union of all trade months (used for outer loop gating)
GROWING_SEASON_MONTHS = sorted(list(set([m for months in COMMODITY_TRADE_MONTHS.values() for m in months])))
=======
# Growing Season Months (Northern Hemisphere)# Growing Season Months (Northern Hemisphere)
# Parses comma-separated list from .env e.g. "6,7,8,9,10"
# Fallback growing season months
GROWING_SEASON_MONTHS = [
    int(m.strip())
    for m in os.getenv('GROWING_SEASON_MONTHS', '6,7,8,9,10').split(',')
]

def get_vol_regime_threshold(commodity: str) -> float:
    env_key = f"{commodity.upper()}_VOL_REGIME_THRESHOLD"
    raw = os.getenv(env_key, None)
    if raw:
        return float(raw)
    # Fall back to global, then hardcoded default
    global_raw = os.getenv('VOL_REGIME_THRESHOLD', None)
    if global_raw:
        return float(global_raw)
    # Hardcoded last resort — set this to whatever makes sense for your system
    return 0.25

# Per-commodity trade months — falls back to GROWING_SEASON_MONTHS if not set
def get_trade_months(commodity: str) -> list:
    """Get trade months for a specific commodity from environment."""
    env_key = f"{commodity.upper()}_TRADE_MONTHS"
    raw = os.getenv(env_key, None)
    if raw:
        return [int(m.strip()) for m in raw.split(',')]
    return GROWING_SEASON_MONTHS


# Commodities — now driven by .env
PHASE_1_COMMODITIES = [
    c.strip().lower()
    for c in os.getenv('COMMODITIES', 'corn,soybeans').split(',')
]

VOL_REGIME_THRESHOLD = float(os.getenv('VOL_REGIME_THRESHOLD', 0.45))
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
