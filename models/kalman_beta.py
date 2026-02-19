"""
Dynamic beta estimation using Kalman filtering.
Allows beta coefficients to evolve over time.
"""
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
from typing import Optional
from loguru import logger


class DynamicBetaEstimator:
    """
    Estimate time-varying betas using Kalman filter.
    
    β_t = β_{t-1} + ε_t
    
    This allows weather sensitivity to change over time (e.g., due to
    improved crop varieties, irrigation adoption, etc.)
    """
    
    def __init__(self, 
                 initial_beta: float = 0.0,
                 process_variance: float = 0.01,
                 observation_variance: float = 1.0):
        """
        Args:
            initial_beta: Initial state estimate
            process_variance: How much beta can change per period (Q)
            observation_variance: Noise in observations (R)
        """
        self.initial_beta = initial_beta
        self.process_variance = process_variance
        self.observation_variance = observation_variance
        self.kf = None
        
    def initialize_filter(self):
        """Initialize Kalman filter."""
        self.kf = KalmanFilter(
            initial_state_mean=self.initial_beta,
            initial_state_covariance=1.0,
            transition_matrices=[1],          # β_t = β_{t-1}
            observation_matrices=[1],          # y_t = β_t * x_t
            transition_covariance=self.process_variance,
            observation_covariance=self.observation_variance
        )
    
    def fit(self, observations: np.ndarray) -> np.ndarray:
        """
        Fit Kalman filter to observation sequence.
        
        Args:
            observations: Time series of observations
            
        Returns:
            Filtered state estimates (time-varying betas)
        """
        if self.kf is None:
            self.initialize_filter()
        
        # Remove NaNs
        valid_idx = ~np.isnan(observations)
        
        if valid_idx.sum() == 0:
            logger.warning("No valid observations for Kalman filter")
            return np.full_like(observations, self.initial_beta)
        
        # Filter
        state_means, state_covariances = self.kf.filter(observations[valid_idx])
        
        # Fill back to original length
        full_state_means = np.full_like(observations, np.nan)
        full_state_means[valid_idx] = state_means.flatten()
        
        # Forward fill NaNs
        full_state_means = pd.Series(full_state_means).fillna(method='ffill').fillna(self.initial_beta).values
        
        return full_state_means
    
    def smooth(self, observations: np.ndarray) -> np.ndarray:
        """
        Apply Kalman smoothing (uses future observations too).
        Only use for analysis, not for real-time prediction.
        
        Returns:
            Smoothed state estimates
        """
        if self.kf is None:
            self.initialize_filter()
        
        valid_idx = ~np.isnan(observations)
        
        if valid_idx.sum() == 0:
            return np.full_like(observations, self.initial_beta)
        
        state_means, state_covariances = self.kf.smooth(observations[valid_idx])
        
        full_state_means = np.full_like(observations, np.nan)
        full_state_means[valid_idx] = state_means.flatten()
        full_state_means = pd.Series(full_state_means).fillna(method='ffill').fillna(self.initial_beta).values
        
        return full_state_means
    
    def predict_next(self, current_state: float, current_covariance: float) -> tuple:
        """
        Predict next state (one-step ahead).
        
        Returns:
            Predicted state mean, predicted state covariance
        """
        # State prediction: β_{t+1|t} = β_t
        predicted_state = current_state
        predicted_covariance = current_covariance + self.process_variance
        
        return predicted_state, predicted_covariance


class MultiVariateKalmanBeta:
    """
    Multivariate Kalman filter for multiple features simultaneously.
    
    β_t = β_{t-1} + ε_t  (vector form)
    """
    
    def __init__(self, 
                 n_features: int,
                 process_variance: float = 0.01,
                 observation_variance: float = 1.0):
        self.n_features = n_features
        self.process_variance = process_variance
        self.observation_variance = observation_variance
        self.kf = None
        
    def initialize_filter(self):
        """Initialize multivariate Kalman filter."""
        self.kf = KalmanFilter(
            initial_state_mean=np.zeros(self.n_features),
            initial_state_covariance=np.eye(self.n_features),
            transition_matrices=np.eye(self.n_features),
            observation_matrices=np.eye(self.n_features),
            transition_covariance=np.eye(self.n_features) * self.process_variance,
            observation_covariance=np.eye(self.n_features) * self.observation_variance
        )
    
    def fit(self, observations: np.ndarray) -> np.ndarray:
        """
        Fit multivariate Kalman filter.
        
        Args:
            observations: Array of shape (n_timesteps, n_features)
            
        Returns:
            Filtered state estimates of shape (n_timesteps, n_features)
        """
        if self.kf is None:
            self.initialize_filter()
        
        # Handle NaNs
        mask = ~np.isnan(observations).any(axis=1)
        
        if mask.sum() == 0:
            return np.zeros_like(observations)
        
        state_means, state_covariances = self.kf.filter(observations[mask])
        
        # Fill back
        full_state_means = np.full_like(observations, np.nan)
        full_state_means[mask] = state_means
        
        # Forward fill
        full_state_means = pd.DataFrame(full_state_means).fillna(method='ffill').fillna(0).values
        
        return full_state_means


def estimate_dynamic_betas(features: np.ndarray, 
                          returns: np.ndarray,
                          feature_names: list) -> pd.DataFrame:
    """
    Estimate time-varying betas for each feature.
    
    Args:
        features: Feature matrix (n_timesteps, n_features)
        returns: Return series (n_timesteps,)
        feature_names: List of feature names
        
    Returns:
        DataFrame with time-varying betas
    """
    logger.info("Estimating dynamic betas using Kalman filter")
    
    n_features = features.shape[1]
    n_timesteps = features.shape[0]
    
    betas = np.zeros((n_timesteps, n_features))
    
    # Estimate beta for each feature independently
    for i, feature_name in enumerate(feature_names):
        # Compute "pseudo-beta" = return / feature
        # This is our observation of the underlying beta
        with np.errstate(divide='ignore', invalid='ignore'):
            observations = returns / (features[:, i] + 1e-8)
        
        # Remove extreme outliers
        observations = np.clip(observations, -10, 10)
        
        # Apply Kalman filter
        estimator = DynamicBetaEstimator(
            initial_beta=0.0,
            process_variance=0.01,
            observation_variance=1.0
        )
        
        betas[:, i] = estimator.fit(observations)
        
        logger.debug(f"Estimated dynamic beta for {feature_name}")
    
    # Convert to DataFrame
    beta_df = pd.DataFrame(betas, columns=[f'{name}_beta' for name in feature_names])
    
    return beta_df