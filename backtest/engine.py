"""
Walk-forward backtesting engine with NO look-ahead bias.
"""
from xml.parsers.expat import model

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from loguru import logger

from data.climate_fetcher import NASAPowerFetcher
from features.feature_pipeline import FeaturePipeline
from models.ridge_model import RollingRidgeModel
from signals.signal_constructor import SignalConstructor
from signals.signal_smoother import SignalSmoother
from council.council_orchestrator import CouncilOrchestrator
from portfolio.constructor import PortfolioConstructor
from portfolio.risk_manager import RiskManager


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
        
        # Initialize components
        self.climate_fetcher = NASAPowerFetcher()
        self.signal_constructor = SignalConstructor()
        self.signal_smoother = SignalSmoother()
        self.council = CouncilOrchestrator()
        self.portfolio_constructor = PortfolioConstructor()
        self.risk_manager = RiskManager()
        
        # State tracking
        self.results = []
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
        logger.info("=" * 80)
        
        # Step 1: Fetch all data (this is OK - we'll use it carefully with no look-ahead)
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
        """
        Fetch market data (prices, volume, etc.) for all commodities.
        """
        from data.market_fetcher import MarketDataFetcher
        
        market_data = {}
        fetcher = MarketDataFetcher(provider='yahoo')  # Use Yahoo Finance!
        
        for commodity in self.commodities:
            logger.info(f"Fetching market data for {commodity}...")
            
            df = fetcher.fetch_futures_data(
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
        """Walk-forward simulation (NO look-ahead)."""
        results = []
        
        # Determine simulation dates
        sim_dates = market_data[self.commodities[0]]['date']
        
        # Start after sufficient history
        min_history = 252 * 10
        start_idx = min(min_history, len(sim_dates) - 252)
        
        # Maintain signal history for smoothing
        signal_history = {c: [] for c in self.commodities}
        
        for idx in range(start_idx, len(sim_dates)):
            current_date = sim_dates.iloc[idx]
            
            if idx == start_idx:
                logger.info(f"Starting simulation at index {idx}, date {current_date}")
            
            if idx % 50 == 0:
                logger.info(f"Processing {current_date.date()} ({idx}/{len(sim_dates)})")
            
            # Get data available up to this date
            commodity_signals = {}
            commodity_vols = {}
            
            for commodity in self.commodities:
                # Features up to current date
                features_up_to_now = features_data[commodity][
                    features_data[commodity]['date'] <= current_date
                ]
                
                # Prices up to current date  
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
                
                # Filter both to common dates only
                features_up_to_now = features_up_to_now[features_up_to_now['date'].isin(common_dates)]
                prices_up_to_now = prices_up_to_now[prices_up_to_now['date'].isin(common_dates)]['close']
                
                # Need at least 1 year of data to train
                min_train = 252
                if len(features_up_to_now) < min_train or len(prices_up_to_now) < min_train:
                    continue
                
                # Generate raw signal
                raw_signal = self._generate_signal_for_date(
                    features_up_to_now,
                    prices_up_to_now,
                    commodity,
                    current_date
                )
                
                # Add to history
                signal_history[commodity].append(raw_signal)
                
                # Smooth using full history
                if len(signal_history[commodity]) > 1:
                    smoothed_signal = self.signal_smoother.smooth_ema(
                        np.array(signal_history[commodity])
                    )[-1]
                else:
                    smoothed_signal = raw_signal
                
                # Compute realized vol
                returns = prices_up_to_now.pct_change()
                vol = returns.iloc[-20:].std() * np.sqrt(252)
                
                commodity_signals[commodity] = smoothed_signal
                commodity_vols[commodity] = vol
                
                # Log signals every 50 days
                if idx % 50 == 0:
                    logger.info(f"{commodity}: raw={raw_signal:.4f}, smoothed={smoothed_signal:.4f}, vol={vol:.2%}")
            
            if not commodity_signals:
                continue
            
            # Council evaluation (optional - you can enable this)
            council_weights = commodity_signals
            
            # Construct portfolio


            from config.settings import STRATEGY_MODE, MIN_SIGNAL_STRENGTH, FIXED_POSITION_SIZE

            positions = {}

            for commodity, signal in commodity_signals.items():
                # Only trade if signal strength > threshold
                #if abs(signal) < MIN_SIGNAL_STRENGTH:
                 #   positions[commodity] = 0.0
                  #  continue
                
                # Binary directional: just trade direction, fixed size
                if STRATEGY_MODE == 'directional':
                    direction = np.sign(signal)  # +1 or -1
                    positions[commodity] = direction * FIXED_POSITION_SIZE
                else:
                    # Original magnitude-based (current broken approach)
                    positions[commodity] = signal * (TARGET_PORTFOLIO_VOL / commodity_vols[commodity])
            
            # Apply risk management
            portfolio_returns = pd.Series([r.get('portfolio_return', 0) for r in results])
            if len(portfolio_returns) > 0:
                current_prices = {c: market_data[c][market_data[c]['date'] == current_date]['close'].iloc[0] 
                                for c in self.commodities if c in positions}
                
                positions = self.risk_manager.apply_risk_limits(
                    positions,
                    commodity_signals,
                    self.entry_prices,
                    current_prices,
                    commodity_vols,
                    portfolio_returns
                )
            
            # Update positions and compute PnL
            daily_pnl = 0.0
            position_changes = []
            
            for commodity in self.commodities:
                if commodity not in positions:
                    continue
                
                current_price = market_data[commodity][
                    market_data[commodity]['date'] == current_date
                ]['close'].iloc[0]
                
                old_position = self.positions[commodity]
                new_position = positions[commodity]
                
                # Calculate PnL from existing position
                if abs(old_position) > 1e-6 and self.entry_prices[commodity] > 0:
                    price_return = (current_price - self.entry_prices[commodity]) / self.entry_prices[commodity]
                    portfolio_value_for_sizing = self.portfolio_value - daily_pnl
                    pnl = portfolio_value_for_sizing * old_position * price_return
                    daily_pnl += pnl
                    
                    # Debug log
                    if idx % 50 == 0 and abs(pnl) > 0.01:
                        logger.info(f"  {commodity}: pos={old_position:.3f}, return={price_return:.2%}, pnl=${pnl:.2f}")
                
                # Update position
                self.positions[commodity] = new_position
                
                # Update entry price only when position changes
                if abs(new_position - old_position) > 0.01:
                    position_changes.append(f"{commodity}: {old_position:.2f} → {new_position:.2f}")
                    self.entry_prices[commodity] = current_price
            
            # Log position changes
            if position_changes:
                logger.info(f"Position changes: {', '.join(position_changes)}")
            
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
        Generate RAW signal for a specific date (smoothing happens in simulation loop).
        """
        model = RollingRidgeModel()
        
        # Prepare features
        X, feature_names = model.prepare_features(features)
        model.feature_names = feature_names
        
        # Compute target
        y = model.compute_target(prices).values
        
        try:
            # FIX: Ensure X and y have same length by trimming to shortest
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            train_window = min(252 * 10, len(X) - 1)
            
            if len(X) < train_window + 1:
                return 0.0
            
            X_train = X[-train_window-1:-1]
            y_train = y[-train_window-1:-1]
            X_test = X[-1:]
            
            # Handle NaNs in X FIRST
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
            
            # THEN remove rows with NaN targets
            logger.info(f"[DEBUG] NaN rate in y_train: {np.isnan(y_train).sum() / len(y_train):.2%}")
            valid_idx = ~np.isnan(y_train)
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
            
            if len(y_train) < 100:
                return 0.0
            
            # Scale and fit
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.model.fit(X_train_scaled, y_train)
            prediction = model.model.predict(X_test_scaled)[0]
            

            logger.info(f"[DEBUG] Raw prediction: {prediction:.6f}")
            logger.info(f"[DEBUG] y_train stats: mean={y_train.mean():.6f}, std={y_train.std():.6f}, min={y_train.min():.6f}, max={y_train.max():.6f}")
            logger.info(f"[DEBUG] Model R²: {model.model.score(X_train_scaled, y_train):.4f}")


            # Construct signal (RAW, no smoothing here)
            returns = prices.pct_change().dropna()  # Calculate returns from prices
            signal = self.signal_constructor.construct_signal(
                np.array([prediction]),
                returns.iloc[-252:]  # Pass returns, not prices
            )[0]

            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating signal for {commodity}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0.0




def run_backtest(start_date: str,
                end_date: str,
                commodities: List[str]) -> pd.DataFrame:
    """
    Convenience function to run backtest.
    """
    engine = BacktestEngine(start_date, end_date, commodities)
    results = engine.run()
    
    return results