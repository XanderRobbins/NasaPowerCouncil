"""Global settings and configuration."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------
NASA_POWER_BASE_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'

# ---------------------------------------------------------------
# Strategy Settings
# ---------------------------------------------------------------
STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'directional')   # 'directional' or 'magnitude'
MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', 0.5))
FIXED_POSITION_SIZE = float(os.getenv('FIXED_POSITION_SIZE', 0.10))
ONLY_TRADE_GROWING_SEASON = os.getenv('ONLY_TRADE_GROWING_SEASON', 'true').lower() == 'true'

# ---------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------
TRAIN_WINDOW_YEARS = int(os.getenv('TRAIN_WINDOW_YEARS', 10))
FORWARD_RETURN_DAYS = int(os.getenv('FORWARD_RETURN_DAYS', 10))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', 15))
RIDGE_ALPHA = float(os.getenv('RIDGE_ALPHA', 1.0))

# ---------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------
TARGET_PORTFOLIO_VOL = float(os.getenv('TARGET_PORTFOLIO_VOL', 0.12))
MAX_SINGLE_POSITION = float(os.getenv('MAX_SINGLE_POSITION', 0.20))
SIGNAL_CAP = float(os.getenv('SIGNAL_CAP', 3.0))
VOL_REGIME_THRESHOLD = float(os.getenv('VOL_REGIME_THRESHOLD', 0.45))

# ---------------------------------------------------------------
# Risk-Free Rate (T-bill proxy for idle capital)
# Used by engine.py to earn return on undeployed capital
# ---------------------------------------------------------------
RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', 0.04))   # 4% annual default

# ---------------------------------------------------------------
# Backtest Settings
# ---------------------------------------------------------------
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2015-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2025-01-01')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1_000_000))

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ---------------------------------------------------------------
# Growing Season Months (Northern Hemisphere)
# Parses comma-separated list from .env e.g. "4,5,6,7"
# Falls back to June–October if not set
# ---------------------------------------------------------------
GROWING_SEASON_MONTHS = [
    int(m.strip())
    for m in os.getenv('GROWING_SEASON_MONTHS', '6,7,8,9,10').split(',')
]

# ---------------------------------------------------------------
# Commodities — driven by .env COMMODITIES key
# Duplicate hardcoded list removed; single source of truth
# ---------------------------------------------------------------
PHASE_1_COMMODITIES = [
    c.strip().lower()
    for c in os.getenv('COMMODITIES', 'corn,soybeans').split(',')
]

# ---------------------------------------------------------------
# Per-commodity helper functions
# ---------------------------------------------------------------
def get_vol_regime_threshold(commodity: str) -> float:
    """
    Get volatility regime threshold for a commodity.
    Priority: commodity-specific env var → global env var → hardcoded default.
    """
    # 1. Commodity-specific: e.g. CORN_VOL_REGIME_THRESHOLD
    commodity_raw = os.getenv(f"{commodity.upper()}_VOL_REGIME_THRESHOLD", None)
    if commodity_raw:
        return float(commodity_raw)

    # 2. Global override: VOL_REGIME_THRESHOLD
    global_raw = os.getenv('VOL_REGIME_THRESHOLD', None)
    if global_raw:
        return float(global_raw)

    # 3. Hardcoded fallback
    return 0.25


def get_trade_months(commodity: str) -> list:
    """
    Get active trading months for a commodity.
    Priority: commodity-specific env var → GROWING_SEASON_MONTHS fallback.
    Example .env: CORN_TRADE_MONTHS=4,5,6,7
    """
    raw = os.getenv(f"{commodity.upper()}_TRADE_MONTHS", None)
    if raw:
        return [int(m.strip()) for m in raw.split(',')]
    return GROWING_SEASON_MONTHS