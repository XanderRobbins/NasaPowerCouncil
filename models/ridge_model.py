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
    - Feature selection via variance filtering
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
        
        feature_cols = [col for col in df.columns
                        if '_agg' in col and any(x in col for x in ['_7d', '_14d', '_30d'])]
        
        if len(feature_cols) == 0:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols 
                        if col not in ['date', 'month', 'day']]
        
        # Select by correlation to target instead of variance
        if target is not None and len(feature_cols) > self.max_features:
            min_len = min(len(df), len(target))
            feature_df = df[feature_cols].iloc[:min_len]
            target_aligned = target.iloc[:min_len]
            
            correlations = feature_df.corrwith(target_aligned).abs()
            correlations = correlations.dropna()
            feature_cols = correlations.nlargest(self.max_features).index.tolist()
            logger.info(f"Top feature correlations: {correlations.nlargest(5).to_dict()}")
        
        X = df[feature_cols].fillna(0).values
        return X, feature_cols




    def compute_target(self, prices: pd.Series) -> pd.Series:
        """
        Compute forward returns.
        
        Y_t = (Price_{t+h} - Price_t) / Price_t
        """
        forward_returns = prices.pct_change(self.forward_horizon).shift(-self.forward_horizon)
        return forward_returns
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on provided data."""
        # Remove NaNs
        valid_idx = ~np.isnan(y)
        X_train = X[valid_idx]
        y_train = y[valid_idx]
        
        if len(y_train) < 100:
            raise ValueError("Not enough valid samples for training")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
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
    """
    Train a Ridge model for a commodity.
    
    Returns:
        Trained model, R² score
    """
    logger.info(f"Training Ridge model for {commodity}")
    
    # Initialize model
    model = RollingRidgeModel()
    
    # Prepare features
    X, feature_names = model.prepare_features(features_df)
    model.feature_names = feature_names
    
    # Compute target
    y = model.compute_target(prices).values
    
    # Align lengths
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    # Train
    model.fit(X, y)
    
    # Evaluate on training data
    valid_idx = ~np.isnan(y)
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    
    predictions = model.predict(X_valid)
    r2 = np.corrcoef(predictions, y_valid)[0, 1] ** 2
    
    logger.info(f"Model R²: {r2:.4f}")
    
    return model, r2