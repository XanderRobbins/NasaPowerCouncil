"""
XGBoost regression model for predicting forward returns.

Mirrors the surface area of RollingRidgeModel so the backtest engine can
swap between the two without any structural changes:

    engine uses:
        m = RollingRidgeModel() | XGBoostRegressionModel()
        m.compute_target(prices)
        m.prepare_features(features_df, target=y_series)
        m.model.fit(X_train, y_train)   # inner sklearn-compatible model
        m.model.predict(X_test)
        m.feature_names = [...]

Hyperparameters are intentionally conservative for the small, noisy
training windows typical of weather-driven commodity data (~750 samples).
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from loguru import logger

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "XGBoost not installed. Run: pip install xgboost "
        "(on macOS you may also need: brew install libomp)"
    ) from e

from config.settings import TRAIN_WINDOW_YEARS, FORWARD_RETURN_DAYS, MAX_FEATURES


class XGBoostRegressionModel:
    """
    Gradient-boosted trees regressor for forward-return prediction.

    Drop-in replacement for RollingRidgeModel. Feature selection and
    target computation are identical to the Ridge path so we're comparing
    models, not pipelines.
    """

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
        self.feature_names = None
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
            n_jobs=1,  # 1-thread per fit; backtest is already sequential per date
            verbosity=0,
        )

    # ------------------------------------------------------------------
    # Shared pipeline surface (matches RollingRidgeModel 1:1)
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame,
                         target: pd.Series = None) -> Tuple[np.ndarray, List[str]]:
        # Primary: aggregated rolling stress features
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

        # Fallback 2: any numeric column
        if len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols
                            if col not in ['date', 'month', 'day']]
            if feature_cols:
                logger.warning(
                    f"No _agg features found at all — falling back to all numeric columns "
                    f"({len(feature_cols)} found)"
                )

        if len(feature_cols) == 0:
            raise ValueError(
                f"prepare_features found 0 usable feature columns.\n"
                f"DataFrame shape: {df.shape}\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # Correlation-based feature selection (same as Ridge path)
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
        """Y_t = (Price_{t+h} - Price_t) / Price_t"""
        forward_returns = prices.pct_change(self.forward_horizon).shift(-self.forward_horizon)
        return forward_returns

    def get_feature_importance(self) -> pd.DataFrame:
        """XGBoost feature importances (gain-based)."""
        if self.feature_names is None:
            return pd.DataFrame()

        importances = self.model.feature_importances_
        # If PCA was applied upstream, feature count won't match feature_names —
        # fall back to PC labels in that case.
        if len(importances) != len(self.feature_names):
            names = [f'PC{i+1}' for i in range(len(importances))]
        else:
            names = self.feature_names

        df = pd.DataFrame({
            'feature': names,
            'importance': importances,
        }).sort_values('importance', ascending=False)
        return df
