"""
Walk-forward backtesting engine with NO look-ahead bias.
Simplified: No council, no smoothing, cached models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from loguru import logger

from data.climate_fetcher import NASAPowerFetcher
from data.market_fetcher import MarketDataFetcher
from features.feature_pipeline import FeaturePipeline
from models.ridge_model import RollingRidgeModel
from config.settings import (
    GROWING_SEASON_MONTHS,
    ONLY_TRADE_GROWING_SEASON,
    STRATEGY_MODE, 
    FIXED_POSITION_SIZE, 
    TARGET_PORTFOLIO_VOL,
    MIN_SIGNAL_STRENGTH,
    MAX_SINGLE_POSITION,
    TRAIN_WINDOW_YEARS
)

class BacktestEngine:
    """
    Walk-forward backtesting engine.
    
    Critical: NO look-ahead bias. At each step, only use data available UP TO that date.
    """
    
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 commodities: List[str],
                 initial_capital: float = 1_000_000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.commodities = commodities
        self.initial_capital = initial_capital
        
        # Components
        self.climate_fetcher = NASAPowerFetcher()
        self.market_fetcher = MarketDataFetcher()
        
        # Model cache (retrain monthly, not daily)
        self.model_cache = {}
        
        # State tracking
        self.positions = {c: 0.0 for c in commodities}
        self.entry_prices = {c: 0.0 for c in commodities}
        self.portfolio_value = initial_capital
        
    def run(self) -> pd.DataFrame:
        """
        Run full backtest.
        
        Returns:
            DataFrame with daily results
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Commodities: {self.commodities}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Strategy Mode: {STRATEGY_MODE}")
        logger.info("=" * 80)
        
        # Step 1: Fetch all data
        logger.info("\nStep 1: Fetching climate data...")
        climate_data = self._fetch_all_climate_data()
        
        logger.info("\nStep 2: Fetching market data...")
        market_data = self._fetch_all_market_data()
        
        logger.info("\nStep 3: Generating features...")
        features_data = self._generate_all_features(climate_data)
        
        logger.info("\nStep 4: Running walk-forward simulation...")
        results = self._walk_forward_simulation(features_data, market_data)
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Total Return: {(self.portfolio_value / self.initial_capital - 1) * 100:.2f}%")
        logger.info("=" * 80)
        
        return results
    
    def _fetch_all_climate_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch climate data for all commodities and regions."""
        all_data = {}
        
        for commodity in self.commodities:
            logger.info(f"Fetching climate data for {commodity}...")
            commodity_data = self.climate_fetcher.fetch_commodity_regions(
                commodity,
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d')
            )
            all_data[commodity] = commodity_data
        
        return all_data
    
    def _fetch_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data (prices, volume, etc.) for all commodities."""
        market_data = {}
        
        for commodity in self.commodities:
            logger.info(f"Fetching market data for {commodity}...")
            
            df = self.market_fetcher.fetch_futures_data(
                commodity,
                'N/A',  # Contract not needed for Yahoo
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d')
            )
            
            market_data[commodity] = df
        
        return market_data
    
    def _generate_all_features(self, 
                              climate_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Generate features for all commodities."""
        features_data = {}
        
        for commodity, region_data in climate_data.items():
            logger.info(f"Generating features for {commodity}...")
            pipeline = FeaturePipeline(commodity)
            features = pipeline.run(region_data)
            features_data[commodity] = features
        
        return features_data
    


    def _walk_forward_simulation(self,
                                features_data: Dict[str, pd.DataFrame],
                                market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Walk-forward simulation (NO look-ahead, NO smoothing, FIXED P&L)."""
        results = []
        
        # Determine simulation dates
        sim_dates = market_data[self.commodities[0]]['date']
        
        # Start after sufficient history (10 years)
        from config.settings import TRAIN_WINDOW_YEARS
        min_history = 252 * TRAIN_WINDOW_YEARS
        start_idx = min(min_history, len(sim_dates) - 252)
        
        logger.info(f"Starting simulation at index {start_idx} / {len(sim_dates)}")
        
        # Track entry notional values (FIXED)
        
        for idx in range(start_idx, len(sim_dates)):
            current_date = sim_dates.iloc[idx]

            #testing growth Filter

            if ONLY_TRADE_GROWING_SEASON and current_date.month not in GROWING_SEASON_MONTHS:
                # Only trade during growing season (e.g., April to September)
                results.append({
                    'date': current_date,
                    'portfolio_value': self.portfolio_value,
                    'daily_pnl': 0.0,
                    'portfolio_return': 0.0,
                    **{f'{c}_signal': 0.0 for c in self.commodities},
                    **{f'{c}_position': self.positions.get(c, 0) for c in self.commodities}
                })
                continue
            
            # Log progress every 100 days
            if idx % 100 == 0:
                logger.info(f"Processing {current_date.date()} ({idx}/{len(sim_dates)}) | "
                        f"Portfolio: ${self.portfolio_value:,.0f}")
            
            # Generate signals for each commodity
            commodity_signals = {}
            commodity_vols = {}
            
            for commodity in self.commodities:
                # Get data available up to this date
                features_up_to_now = features_data[commodity][
                    features_data[commodity]['date'] <= current_date
                ]
                
                prices_up_to_now = market_data[commodity][
                    market_data[commodity]['date'] <= current_date
                ]
                
                # Align dates between features and prices
                common_dates = pd.merge(
                    features_up_to_now[['date']],
                    prices_up_to_now[['date']],
                    on='date',
                    how='inner'
                )['date']
                
                features_up_to_now = features_up_to_now[features_up_to_now['date'].isin(common_dates)]
                prices_up_to_now = prices_up_to_now[prices_up_to_now['date'].isin(common_dates)]['close']
                
                # Need at least 1 year of data to train
                if len(features_up_to_now) < 252 or len(prices_up_to_now) < 252:
                    continue
                
                # Generate signal (NO SMOOTHING)
                signal = self._generate_signal_for_date(
                    features_up_to_now,
                    prices_up_to_now,
                    commodity,
                    current_date
                )
                
                # Compute realized vol
                returns = prices_up_to_now.pct_change()
                vol = returns.iloc[-20:].std() * np.sqrt(252)
                
                commodity_signals[commodity] = signal
                commodity_vols[commodity] = vol
            
            if not commodity_signals:
                continue
            
            # Position sizing
            positions = {}
            for commodity, signal in commodity_signals.items():
                if STRATEGY_MODE == 'directional':
                    # Binary directional: fixed size, sign from signal
                    if abs(signal) >= MIN_SIGNAL_STRENGTH:
                        positions[commodity] = np.sign(signal) * FIXED_POSITION_SIZE
                    else:
                        positions[commodity] = 0.0
                else:
                    # Magnitude-based (vol-scaled)
                    positions[commodity] = signal * (TARGET_PORTFOLIO_VOL / commodity_vols[commodity])
            
            # Apply simple position caps
            for commodity in positions:
                positions[commodity] = np.clip(positions[commodity], -MAX_SINGLE_POSITION, MAX_SINGLE_POSITION)
            
            # Compute P&L and update positions (FIXED CALCULATION)
            daily_pnl = 0.0
            
            for commodity in self.commodities:
                if commodity not in positions:
                    continue
                
                price_row = market_data[commodity][market_data[commodity]['date'] == current_date]
                if price_row.empty:
                    continue
                
                current_price = price_row['close'].iloc[0]
                old_position = self.positions[commodity]
                new_position = positions[commodity]
                
                # Calculate P&L from existing position (FIXED)
                if abs(old_position) > 1e-6 and self.entry_prices[commodity] > 0:
                    daily_return = (current_price - self.entry_prices[commodity]) / self.entry_prices[commodity]
                    pnl = self.portfolio_value * old_position * daily_return
                    daily_pnl += pnl
                
                # Update position
                self.entry_prices[commodity] = current_price
                self.positions[commodity] = new_position

                # Update entry price when position changes
                #if abs(new_position - old_position) > 0.01:
                #    self.entry_prices[commodity] = current_price
            
            # Update portfolio value
            self.portfolio_value += daily_pnl
            portfolio_return = daily_pnl / (self.portfolio_value - daily_pnl) if (self.portfolio_value - daily_pnl) > 0 else 0
            
            # Record results
            result = {
                'date': current_date,
                'portfolio_value': self.portfolio_value,
                'daily_pnl': daily_pnl,
                'portfolio_return': portfolio_return,
                **{f'{c}_signal': commodity_signals.get(c, 0) for c in self.commodities},
                **{f'{c}_position': positions.get(c, 0) for c in self.commodities}
            }
            results.append(result)
        
        logger.info(f"Collected {len(results)} result records")
        return pd.DataFrame(results)
    
    def _generate_signal_for_date(self,
                                features: pd.DataFrame,
                                prices: pd.Series,
                                commodity: str,
                                current_date: pd.Timestamp) -> float:
        """
        Generate signal for a specific date (cached models, no smoothing).
        
        Returns:
            Vol-standardized signal, capped at ±3
        """
        # Cache key: retrain monthly instead of daily
        cache_key = f"{commodity}_{current_date.year}_{current_date.month}"
        
        # Check if model exists in cache
        if cache_key in self.model_cache:
            model, scaler = self.model_cache[cache_key]
            
            # Just predict using cached model
            X, _ = model.prepare_features(features)
            if len(X) == 0:
                return 0.0
            
            X_test = X[-1:]
            X_test_scaled = scaler.transform(X_test)
            prediction = model.model.predict(X_test_scaled)[0]
        
        else:
            # Train new model (first time seeing this month)
            model = RollingRidgeModel()
            
            # Prepare features
            X, feature_names = model.prepare_features(features)
            model.feature_names = feature_names
            
            # Compute target (forward returns)
            y = model.compute_target(prices).values
            
            # Align lengths
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Train/test split (last 10 years train, predict next point)
            from config.settings import TRAIN_WINDOW_YEARS
            train_window = min(252 * TRAIN_WINDOW_YEARS, len(X) - 1)
            if len(X) < train_window + 1:
                return 0.0
            
            X_train = X[-train_window-1:-1]
            y_train = y[-train_window-1:-1]
            X_test = X[-1:]
            
            # Clean NaNs
            valid_idx = ~np.isnan(y_train)
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
            
            if len(y_train) < 252:  # Need at least 1 year
                return 0.0
            
            # Scale
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model.model.fit(X_train_scaled, y_train)
            
            # Cache for future use this month
            self.model_cache[cache_key] = (model, scaler)
            
            # Predict
            prediction = model.model.predict(X_test_scaled)[0]
        
        # Standardize by realized vol
        returns = prices.pct_change().dropna()
        vol = returns.iloc[-20:].std() * np.sqrt(252)
        
        # Signal = prediction / vol, capped at ±3
        signal = prediction / (vol + 1e-8)
        signal = np.clip(signal, -3.0, 3.0)
        
        return signal


def run_backtest(start_date: str,
                end_date: str,
                commodities: List[str],
                initial_capital: float = 1_000_000) -> pd.DataFrame:
    """
    Convenience function to run backtest.
    """
    engine = BacktestEngine(start_date, end_date, commodities, initial_capital)
    results = engine.run()
    
    return results