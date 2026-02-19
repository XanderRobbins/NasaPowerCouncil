"""
Centralized model training orchestration.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from loguru import logger
import pickle

from models.ridge_model import RollingRidgeModel
from models.kalman_beta import DynamicBetaEstimator
from config.settings import MODEL_PATH


class ModelTrainer:
    """
    Orchestrate model training for all commodities.
    """
    
    def __init__(self, commodities: List[str]):
        self.commodities = commodities
        self.models: Dict[str, RollingRidgeModel] = {}
        self.beta_estimators: Dict[str, DynamicBetaEstimator] = {}
        
    def train_all_models(self,
                        features_data: Dict[str, pd.DataFrame],
                        price_data: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """
        Train models for all commodities.
        
        Returns:
            Dict with training results per commodity
        """
        results = {}
        
        for commodity in self.commodities:
            logger.info(f"Training model for {commodity}...")
            
            features = features_data.get(commodity)
            prices = price_data.get(commodity)
            
            if features is None or prices is None:
                logger.warning(f"Missing data for {commodity}, skipping")
                continue
            
            result = self.train_commodity_model(commodity, features, prices)
            results[commodity] = result
        
        return results
    
    def train_commodity_model(self,
                             commodity: str,
                             features: pd.DataFrame,
                             prices: pd.Series) -> Dict:
        """
        Train model for a single commodity.
        
        Returns:
            Training results
        """
        # Initialize model
        model = RollingRidgeModel()
        
        # Prepare features
        X, feature_names = model.prepare_features(features)
        model.feature_names = feature_names
        
        # Compute target
        y = model.compute_target(prices).values
        
        # Cross-validate alpha
        logger.info(f"Cross-validating alpha for {commodity}...")
        best_alpha = model.cross_val_alpha(X, y)
        model.alpha = best_alpha
        model.model.alpha = best_alpha
        
        # Train on full dataset (for initial deployment)
        # In production, use rolling window
        valid_idx = ~np.isnan(y)
        X_train = X[valid_idx]
        y_train = y[valid_idx]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model.scaler = scaler
        model.model.fit(X_train_scaled, y_train)
        
        # Compute training metrics
        y_pred = model.model.predict(X_train_scaled)
        
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        
        logger.info(f"{commodity} training complete: R²={r2:.4f}, MAE={mae:.4f}")
        
        # Store model
        self.models[commodity] = model
        
        # Also train dynamic beta estimator
        beta_estimator = DynamicBetaEstimator()
        betas = beta_estimator.fit(y_train)
        self.beta_estimators[commodity] = beta_estimator
        
        return {
            'commodity': commodity,
            'best_alpha': best_alpha,
            'r2': r2,
            'mae': mae,
            'n_features': len(feature_names),
            'n_samples': len(y_train)
        }
    
    def save_models(self):
        """Save trained models to disk."""
        MODEL_PATH.mkdir(exist_ok=True)
        
        for commodity, model in self.models.items():
            model_file = MODEL_PATH / f"{commodity}_model.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Saved model for {commodity} to {model_file}")
    
    def load_models(self) -> Dict[str, RollingRidgeModel]:
        """Load trained models from disk."""
        loaded_models = {}
        
        for commodity in self.commodities:
            model_file = MODEL_PATH / f"{commodity}_model.pkl"
            
            if not model_file.exists():
                logger.warning(f"Model file not found for {commodity}")
                continue
            
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                loaded_models[commodity] = model
                logger.info(f"Loaded model for {commodity}")
                
            except Exception as e:
                logger.error(f"Failed to load model for {commodity}: {e}")
        
        self.models = loaded_models
        return loaded_models
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all models."""
        summaries = []
        
        for commodity, model in self.models.items():
            summaries.append({
                'commodity': commodity,
                'alpha': model.alpha,
                'n_features': len(model.feature_names) if model.feature_names else 0,
                'model_type': 'RollingRidge'
            })
        
        return pd.DataFrame(summaries)