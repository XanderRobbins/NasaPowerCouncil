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
MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', 0.5))
FIXED_POSITION_SIZE = float(os.getenv('FIXED_POSITION_SIZE', 0.10))
ONLY_TRADE_GROWING_SEASON = os.getenv('ONLY_TRADE_GROWING_SEASON', 'true').lower() == 'true'

# Model Parameters
TRAIN_WINDOW_YEARS = int(os.getenv('TRAIN_WINDOW_YEARS', 10))
FORWARD_RETURN_DAYS = int(os.getenv('FORWARD_RETURN_DAYS', 10))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', 15))
RIDGE_ALPHA = float(os.getenv('RIDGE_ALPHA', 1.0))

# Risk Management
TARGET_PORTFOLIO_VOL = float(os.getenv('TARGET_PORTFOLIO_VOL', 0.12))
MAX_SINGLE_POSITION = float(os.getenv('MAX_SINGLE_POSITION', 0.20))
SIGNAL_CAP = float(os.getenv('SIGNAL_CAP', 3.0))

# Backtest Settings
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2015-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2025-01-01')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1_000_000))

# Commodities
PHASE_1_COMMODITIES = ['corn', 'soybeans']

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Growing Season Months (Northern Hemisphere)
GROWING_SEASON_MONTHS = [4, 5, 6, 7, 8, 9]  # April-September