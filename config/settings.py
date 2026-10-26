"""Global settings.

Every value can be overridden with an environment variable (or a ``.env`` file in the
project root, see ``.env.example``). The defaults below are the values the strategy
was actually tested with, so running with no ``.env`` at all reproduces the documented
results.

Importing this module creates the ``data_storage`` folders as a side effect, because
the fetchers and ``main.py`` write to them.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_STORAGE_PATH = Path(os.getenv('DATA_STORAGE_PATH', PROJECT_ROOT / 'data_storage'))
RAW_DATA_PATH = DATA_STORAGE_PATH / 'raw'        # cached NASA POWER and Yahoo downloads
RESULTS_PATH = DATA_STORAGE_PATH / 'results'     # backtest CSV, chart, tearsheet
LOG_PATH = DATA_STORAGE_PATH / 'logs'

for path in [RAW_DATA_PATH, RESULTS_PATH, LOG_PATH]:
    path.mkdir(parents=True, exist_ok=True)

NASA_POWER_BASE_URL = 'https://power.larc.nasa.gov/api/temporal/daily/point'

# ----------------------------------------------------------------------------
# Strategy: how a signal becomes a position
# ----------------------------------------------------------------------------
# 'directional': hold +/-FIXED_POSITION_SIZE according to the sign of the signal.
# 'magnitude'  : scale the signal by TARGET_PORTFOLIO_VOL / realised volatility.
STRATEGY_MODE = os.getenv('STRATEGY_MODE', 'directional')
# In directional mode, |signal| below this means "no view" (flat).
MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', 0.01))
# Position size as a fraction of portfolio value (0.5 = 50% notional, long or short).
FIXED_POSITION_SIZE = float(os.getenv('FIXED_POSITION_SIZE', 0.5))
# True: only trade in each commodity's COMMODITY_TRADE_MONTHS and sit in cash otherwise.
# False: trade all year.
ONLY_TRADE_GROWING_SEASON = os.getenv('ONLY_TRADE_GROWING_SEASON', 'true').lower() == 'true'

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
TRAIN_WINDOW_YEARS = int(os.getenv('TRAIN_WINDOW_YEARS', 3))     # rolling training window
FORWARD_RETURN_DAYS = int(os.getenv('FORWARD_RETURN_DAYS', 10))  # prediction horizon
MAX_FEATURES = int(os.getenv('MAX_FEATURES', 15))                # keep the top-k correlated features
RIDGE_ALPHA = float(os.getenv('RIDGE_ALPHA', 1.0))               # L2 penalty of the Ridge model

# PCA is applied after scaling and before the regressor (linear model only).
# PCA_N_COMPONENTS is either an int (keep exactly that many components) or a float in
# (0, 1) (keep enough components to explain that share of the variance).
USE_PCA = os.getenv('USE_PCA', 'true').lower() == 'true'
_pca_raw = os.getenv('PCA_N_COMPONENTS', '0.95')
PCA_N_COMPONENTS = float(_pca_raw) if '.' in _pca_raw else int(_pca_raw)

# Regression model: 'ridge' (default) or 'xgboost' (needs the optional xgboost package).
MODEL_TYPE = os.getenv('MODEL_TYPE', 'ridge').lower()

# ----------------------------------------------------------------------------
# Risk management
# ----------------------------------------------------------------------------
TARGET_PORTFOLIO_VOL = float(os.getenv('TARGET_PORTFOLIO_VOL', 0.99))  # magnitude mode only
MAX_SINGLE_POSITION = float(os.getenv('MAX_SINGLE_POSITION', 1.0))     # hard cap per commodity
SIGNAL_CAP = float(os.getenv('SIGNAL_CAP', 3.0))                       # z-scored signal is clipped to +/- this
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', 0.05))                # close a position that loses this much
# The direction classifier vetoes a trade only when it disagrees with the regression
# AND is at least this confident.
VETO_CONFIDENCE = float(os.getenv('VETO_CONFIDENCE', 0.65))

# ----------------------------------------------------------------------------
# Execution realism
# ----------------------------------------------------------------------------
# NASA POWER publishes with a 1-2+ day delay, and a signal computed from today's close
# can only be traded at the next close. Set FEATURE_LAG_DAYS=0 and EXECUTE_NEXT_BAR=false
# to reproduce the optimistic same-day behaviour (results will look better than live).
FEATURE_LAG_DAYS = int(os.getenv('FEATURE_LAG_DAYS', 2))
EXECUTE_NEXT_BAR = os.getenv('EXECUTE_NEXT_BAR', 'true').lower() == 'true'

# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2010-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2023-01-01')
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 1_000_000))
TRANSACTION_COST_BPS = float(os.getenv('TRANSACTION_COST_BPS', 10.0))  # per unit of position change

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO') or 'INFO'

# ----------------------------------------------------------------------------
# Commodities
# ----------------------------------------------------------------------------
# Only corn is wired up for trading. Regions and thresholds exist for other crops
# (config/regions.py, config/thresholds.py) but they have no trade months or volatility
# thresholds here, so they cannot be run without adding entries below.
PHASE_1_COMMODITIES = ['corn']

# Months (1-12) in which each commodity may hold a position. Override with
# CORN_TRADE_MONTHS="4,5,6". The default is April-June (planting through early growth).
COMMODITY_TRADE_MONTHS = {}
for commodity in PHASE_1_COMMODITIES:
    months_str = os.getenv(f'{commodity.upper()}_TRADE_MONTHS', '4,5,6')
    try:
        COMMODITY_TRADE_MONTHS[commodity] = [int(m.strip()) for m in months_str.split(',')]
    except ValueError:
        COMMODITY_TRADE_MONTHS[commodity] = [4, 5, 6]

# Volatility-regime filter: stand aside when 20-day annualised volatility exceeds this.
# Override with CORN_VOL_REGIME_THRESHOLD.
COMMODITY_VOL_THRESHOLDS = {}
for commodity in PHASE_1_COMMODITIES:
    COMMODITY_VOL_THRESHOLDS[commodity] = float(os.getenv(f'{commodity.upper()}_VOL_REGIME_THRESHOLD', 0.3))
