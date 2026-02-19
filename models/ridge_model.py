"""
Rolling Ridge regression model.
Train on 10-year windows, predict forward returns.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from loguru import logger

from config.settings import TRAIN_WINDOW_YEARS, FORWARD_RETURN_DAYS, MAX_FEATURES_PER_COMMODITY


class RollingRidgeModel:
    """
    Ridge regression with rolling training windows.
    
    Key features:
    - No look-ahead bias (strict temporal split)
    - Rolling cross-validation for hyperparameter tuning
    - Feature selection via L2 penalty
    """
    
    def __init__(self, 
                 train_window_days: int = 252 * TRAIN_WINDOW_YEARS,
                 forward_horizon: int = FORWARD_RETURN_DAYS,
                 alpha: float = 1,
                 max_features: int = MAX_FEATURES_PER_COMMODITY):
        self.train_window = train_window_days
        self.forward_horizon = forward_horizon
        self.alpha = alpha
        self.max_features = max_features
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Select and prepare features for modeling.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Feature matrix, feature names
        """
        # Select cumulative stress features
        feature_cols = [col for col in df.columns 
                       if ('_agg' in col) and ('_7d' in col or '_14d' in col or '_30d' in col)]
        
        # Limit to max features (avoid overfitting)
        if len(feature_cols) > self.max_features:
            logger.warning(f"Too many features ({len(feature_cols)}), selecting top {self.max_features}")
            # Select features with highest variance (proxy for information content)
            variances = df[feature_cols].var().sort_values(ascending=False)
            feature_cols = variances.head(self.max_features).index.tolist()
        
        X = df[feature_cols].values
        
        return X, feature_cols
    
    def compute_target(self, prices: pd.Series) -> pd.Series:
        """
        Compute forward returns.
        
        Y_t = (Price_{t+h} - Price_t) / Price_t
        """
        forward_returns = prices.pct_change(self.forward_horizon).shift(-self.forward_horizon)
        return forward_returns
    
    def rolling_train_predict(self, 
                              X: np.ndarray, 
                              y: np.ndarray,
                              dates: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rolling window training and prediction.
        
        Args:
            X: Feature matrix
            y: Target (forward returns)
            dates: Date index
            
        Returns:
            Predictions, actual values (aligned, only for test periods)
        """
        predictions = []
        actuals = []
        test_dates = []
        
        # Ensure we have enough data
        if len(X) < self.train_window:
            logger.error(f"Not enough data: {len(X)} < {self.train_window}")
            return np.array([]), np.array([])
        
        # Rolling window
        for i in range(self.train_window, len(X) - self.forward_horizon):
            # Training window
            train_start = i - self.train_window
            train_end = i
            
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            
            # Remove NaN targets (from forward return calculation at end of series)
            valid_idx = ~np.isnan(y_train)
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
            
            if len(y_train) == 0:
                continue
            
            # Test point
            X_test = X[i:i+1]
            y_test = y[i]
            
            if np.isnan(y_test):
                continue
            
            # Scale
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train
            self.model.fit(X_train_scaled, y_train)
            
            # Predict
            pred = self.model.predict(X_test_scaled)[0]
            
            predictions.append(pred)
            actuals.append(y_test)
            test_dates.append(dates.iloc[i])
        
        return np.array(predictions), np.array(actuals), test_dates
    
    def cross_validate_alpha(self, 
                            X: np.ndarray, 
                            y: np.ndarray,
                            alphas: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]) -> float:
        """
        Use time-series cross-validation to select optimal alpha (L2 penalty).
        
        Returns:
            Best alpha value
        """
        tscv = TimeSeriesSplit(n_splits=5)
        best_alpha = alphas[0]
        best_score = -np.inf
        
        # Remove NaNs
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Fit and score
                model.fit(X_train_scaled, y_train)
                score = model.score(X_val_scaled, y_val)
                scores.append(score)
            
            avg_score = np.mean(scores)
            logger.debug(f"Alpha={alpha}: R²={avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_alpha = alpha
        
        logger.info(f"Best alpha: {best_alpha} (R²={best_score:.4f})")
        return best_alpha
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature coefficients (importance).
        """
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
                     commodity: str) -> Tuple[RollingRidgeModel, np.ndarray, np.ndarray]:
    """
    Convenience function to train a Ridge model for a commodity.
    
    Args:
        features_df: Feature DataFrame with date index
        prices: Price series (same date index)
        commodity: Commodity name
        
    Returns:
        Trained model, predictions, actuals
    """
    logger.info(f"Training Ridge model for {commodity}")
    
    # Initialize model
    model = RollingRidgeModel()
    
    # Prepare features
    X, feature_names = model.prepare_features(features_df)
    model.feature_names = feature_names
    
    # Compute target
    y = model.compute_target(prices).values
    
    # Cross-validate alpha
    best_alpha = model.cross_validate_alpha(X, y)
    model.alpha = best_alpha
    model.model = Ridge(alpha=best_alpha)
    
    # Rolling train and predict
    predictions, actuals, dates = model.rolling_train_predict(X, y, features_df['date'])
    
    logger.info(f"Model trained: {len(predictions)} predictions generated")
    
    return model, predictions, actuals, dates