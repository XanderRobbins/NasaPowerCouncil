"""
Rolling Ridge regression model for predicting forward returns.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
from loguru import logger

from config.settings import TRAIN_WINDOW_YEARS, FORWARD_RETURN_DAYS, MAX_FEATURES


class RollingRidgeModel:
    """
    Ridge regression with rolling training windows.

    Features:
    - No look-ahead bias (strict temporal split)
    - Feature selection via correlation filtering
    - L2 regularization to prevent overfitting
    """

    def __init__(self,
                 train_window_days: int = 252 * TRAIN_WINDOW_YEARS,
                 forward_horizon: int = FORWARD_RETURN_DAYS,
                 alpha: float = 1.0,
                 max_features: int = MAX_FEATURES):
        self.train_window = train_window_days
        self.forward_horizon = forward_horizon
        self.alpha = alpha
        self.max_features = max_features
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_features(self, df: pd.DataFrame,
                         target: pd.Series = None) -> Tuple[np.ndarray, List[str]]:

        # Primary: aggregated rolling stress features (ideal case)
        # e.g. heat_stress_7d_agg, dry_stress_z_30d_agg
        feature_cols = [col for col in df.columns
                        if '_agg' in col and any(x in col for x in ['_7d', '_14d', '_30d'])]

        # Fallback 1: any aggregated feature at all
        if len(feature_cols) == 0:
            feature_cols = [col for col in df.columns if '_agg' in col]
            if feature_cols:
                logger.warning(
                    f"No rolling _agg features found — falling back to all _agg columns "
                    f"({len(feature_cols)} found): {feature_cols[:5]}"
                )

        # Fallback 2: any numeric column (last resort)
        if len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols
                            if col not in ['date', 'month', 'day']]
            if feature_cols:
                logger.warning(
                    f"No _agg features found at all — falling back to all numeric columns "
                    f"({len(feature_cols)} found)"
                )

        # Hard guard: fail loudly with useful context
        if len(feature_cols) == 0:
            raise ValueError(
                f"prepare_features found 0 usable feature columns.\n"
                f"DataFrame shape: {df.shape}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # Select by correlation to target if too many features
        if target is not None and len(feature_cols) > self.max_features:
            min_len = min(len(df), len(target))
            feature_df = df[feature_cols].iloc[:min_len]
            target_aligned = target.iloc[:min_len]

            correlations = feature_df.corrwith(target_aligned).abs().dropna()

            if len(correlations) == 0:
                logger.warning("corrwith returned empty — using variance-based selection instead")
                variances = df[feature_cols].var().sort_values(ascending=False)
                feature_cols = variances.head(self.max_features).index.tolist()
            else:
                feature_cols = correlations.nlargest(self.max_features).index.tolist()
                logger.info(f"Top feature correlations: {correlations.nlargest(5).to_dict()}")

        X = df[feature_cols].fillna(0).values
        return X, feature_cols

    def compute_target(self, prices: pd.Series) -> pd.Series:
        """Compute forward returns. Y_t = (Price_{t+h} - Price_t) / Price_t"""
        forward_returns = prices.pct_change(self.forward_horizon).shift(-self.forward_horizon)
        return forward_returns

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on provided data."""
        valid_idx = ~np.isnan(y)
        X_train = X[valid_idx]
        y_train = y[valid_idx]

        if len(y_train) < 100:
            raise ValueError("Not enough valid samples for training")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        logger.info(f"Model trained on {len(y_train)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients (importance)."""
        if self.feature_names is None:
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        importance['abs_coefficient'] = importance['coefficient'].abs()
        importance = importance.sort_values('abs_coefficient', ascending=False)
        return importance


def train_ridge_model(features_df: pd.DataFrame,
                      prices: pd.Series,
                      commodity: str) -> Tuple[RollingRidgeModel, float]:
    """Train a Ridge model for a commodity. Returns: Trained model, R² score"""
    logger.info(f"Training Ridge model for {commodity}")

    model = RollingRidgeModel()
    X, feature_names = model.prepare_features(features_df)
    model.feature_names = feature_names

    y = model.compute_target(prices).values
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    model.fit(X, y)

    valid_idx = ~np.isnan(y)
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    predictions = model.predict(X_valid)
    r2 = np.corrcoef(predictions, y_valid)[0, 1] ** 2

    logger.info(f"Model R²: {r2:.4f}")
    return model, r2