"""
Ridge regression wrapper used to forecast forward returns.

The backtest engine owns scaling, PCA and the rolling retraining schedule; this class only
supplies the target, the feature selection and the (unfitted) sklearn model.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config.settings import FORWARD_RETURN_DAYS, MAX_FEATURES, TRAIN_WINDOW_YEARS
from models.common import compute_forward_return, prepare_features


class RollingRidgeModel:
    """Ridge regression (L2-regularised linear model) for the N-day forward return."""

    def __init__(self,
                 train_window_days: int = 252 * TRAIN_WINDOW_YEARS,
                 forward_horizon: int = FORWARD_RETURN_DAYS,
                 alpha: float = 1.0,
                 max_features: int = MAX_FEATURES):
        self.train_window = train_window_days
        self.forward_horizon = forward_horizon
        self.max_features = max_features
        self.model = Ridge(alpha=alpha)   # fitted by the engine on scaled / PCA-reduced features
        self.feature_names: List[str] = []  # set by the engine after prepare_features

    def compute_target(self, prices: pd.Series) -> pd.Series:
        """Forward return over ``forward_horizon`` days (NaN for the last few rows)."""
        return compute_forward_return(prices, self.forward_horizon)

    def prepare_features(self, df: pd.DataFrame, target: pd.Series = None) -> Tuple[np.ndarray, List[str]]:
        """Select the input columns (see ``models.common.prepare_features``); returns (X, names)."""
        return prepare_features(df, target, self.max_features, self.train_window)
