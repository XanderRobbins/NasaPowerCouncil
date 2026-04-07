"""
Directional classifier for predicting price movement direction.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, List
from loguru import logger

from config.settings import FORWARD_RETURN_DAYS, MAX_FEATURES


class DirectionalClassifier:
    """
    Logistic regression classifier for predicting direction (up/down).

    This optimizes directly for what matters in trading: getting the direction right.
    Unlike Ridge which minimizes squared error, this maximizes directional accuracy.
    """

    def __init__(self,
                 forward_horizon: int = FORWARD_RETURN_DAYS,
                 C: float = 1.0,
                 max_features: int = MAX_FEATURES):
        self.forward_horizon = forward_horizon
        self.C = C
        self.max_features = max_features
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None

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

        # Hard guard
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
        """Compute binary target: 1 if price goes up, 0 if down."""
        forward_returns = prices.pct_change(self.forward_horizon).shift(-self.forward_horizon)
        direction = (forward_returns > 0).astype(int)
        return direction

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        valid_idx = ~np.isnan(y)
        X_train = X[valid_idx]
        y_train = y[valid_idx]

        if len(y_train) < 100:
            raise ValueError("Not enough valid samples for training")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Training class distribution: {class_dist}")
        logger.info(f"Classifier trained on {len(y_train)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict direction (0 or 1)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of upward move."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature coefficients."""
        if self.feature_names is None:
            return pd.DataFrame()

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        })
        importance['abs_coefficient'] = importance['coefficient'].abs()
        importance = importance.sort_values('abs_coefficient', ascending=False)
        return importance


def train_classifier(features_df: pd.DataFrame,
                     prices: pd.Series,
                     commodity: str) -> Tuple[DirectionalClassifier, float]:
    """Train a directional classifier. Returns: Trained classifier, accuracy score"""
    logger.info(f"Training directional classifier for {commodity}")

    clf = DirectionalClassifier()
    X, feature_names = clf.prepare_features(features_df)
    clf.feature_names = feature_names

    y = clf.compute_target(prices).values
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    clf.fit(X, y)

    valid_idx = ~np.isnan(y)
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]

    predictions = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, predictions)

    logger.info(f"Classifier accuracy: {accuracy:.2%}")
    logger.info("\n" + classification_report(y_valid, predictions,
                                             target_names=['Down', 'Up']))

    return clf, accuracy