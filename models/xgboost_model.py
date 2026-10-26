"""
Optional XGBoost regressor, a drop-in alternative to the Ridge model (``MODEL_TYPE=xgboost``).

It exposes the same surface as ``RollingRidgeModel`` (``compute_target``, ``prepare_features``,
``model``, ``feature_names``) so the engine does not need to know which one it has.
Needs the optional ``xgboost`` package.

Hyperparameters are deliberately conservative because training windows are small and noisy
(about 750 daily rows).
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "XGBoost not installed. Run: pip install xgboost "
        "(on macOS you may also need: brew install libomp)"
    ) from e

from config.settings import FORWARD_RETURN_DAYS, MAX_FEATURES, TRAIN_WINDOW_YEARS
from models.common import compute_forward_return, prepare_features


class XGBoostRegressionModel:
    """Gradient-boosted trees for the N-day forward return."""

    def __init__(self,
                 train_window_days: int = 252 * TRAIN_WINDOW_YEARS,
                 forward_horizon: int = FORWARD_RETURN_DAYS,
                 max_features: int = MAX_FEATURES,
                 n_estimators: int = 200,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0,
                 reg_alpha: float = 0.0,
                 random_state: int = 42):
        self.train_window = train_window_days
        self.forward_horizon = forward_horizon
        self.max_features = max_features
        self.feature_names: List[str] = []
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            random_state=random_state,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=1,  # the backtest is already sequential per date
            verbosity=0,
        )

    def compute_target(self, prices: pd.Series) -> pd.Series:
        """Forward return over ``forward_horizon`` days (NaN for the last few rows)."""
        return compute_forward_return(prices, self.forward_horizon)

    def prepare_features(self, df: pd.DataFrame, target: pd.Series = None) -> Tuple[np.ndarray, List[str]]:
        """Select the input columns (see ``models.common.prepare_features``); returns (X, names)."""
        return prepare_features(df, target, self.max_features, self.train_window)
