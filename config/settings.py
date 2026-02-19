"""
Global settings and configuration.
"""
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

# Create directories
for path in [RAW_DATA_PATH, PROCESSED_DATA_PATH, MODEL_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# API Settings
NASA_POWER_BASE_URL = os.getenv('NASA_POWER_BASE_URL', 
                                'https://power.larc.nasa.gov/api/temporal/daily/point')
MARKET_DATA_API_KEY = os.getenv('MARKET_DATA_API_KEY')
MARKET_DATA_PROVIDER = os.getenv('MARKET_DATA_PROVIDER', 'quandl')

# Database
DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DATA_STORAGE_PATH}/climate_futures.db')

# Risk Parameters
TARGET_PORTFOLIO_VOL = float(os.getenv('TARGET_PORTFOLIO_VOL', 0.12))
MAX_SINGLE_COMMODITY = float(os.getenv('MAX_SINGLE_COMMODITY', 0.20))
MAX_DRAWDOWN_THRESHOLD = float(os.getenv('MAX_DRAWDOWN_THRESHOLD', 0.10))
HARD_STOP_MULTIPLIER = float(os.getenv('HARD_STOP_MULTIPLIER', 2.5))

# Model Parameters
TRAIN_WINDOW_YEARS = int(os.getenv('TRAIN_WINDOW_YEARS', 10))
BASELINE_YEARS = int(os.getenv('BASELINE_YEARS', 20))
FORWARD_RETURN_DAYS = int(os.getenv('FORWARD_RETURN_DAYS', 20))
MAX_FEATURES_PER_COMMODITY = int(os.getenv('MAX_FEATURES_PER_COMMODITY', 15))

# Signal Parameters
SIGNAL_CAP = float(os.getenv('SIGNAL_CAP', 3.0))
SIGNAL_EMA_SPAN = int(os.getenv('SIGNAL_EMA_SPAN', 3))

# Backtest Settings
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2015-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2024-12-31')

# LLM Settings (for council)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
USE_LLM_COUNCIL = os.getenv('USE_LLM_COUNCIL', 'false').lower() == 'true'

# Phase 1 Commodities
PHASE_1_COMMODITIES = ['corn']

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = DATA_STORAGE_PATH / 'logs' / 'climate_futures.log'
LOG_FILE.parent.mkdir(exist_ok=True)