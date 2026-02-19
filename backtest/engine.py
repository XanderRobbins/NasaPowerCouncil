"""
Walk-forward backtesting engine with NO look-ahead bias.
"""
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
        """
        Walk-forward simulation (NO look-ahead).
        """
        results = []
        
        # Determine simulation dates
        sim_dates = market_data[self.commodities[0]]['date']
        
        # Start after sufficient history
        min_history = 252 * 10
        start_idx = min(min_history, len(sim_dates) - 252)
        
        # MAINTAIN SIGNAL HISTORY for smoothing
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
                
                # FIX: Align dates between features and prices
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
                    if idx == start_idx:
                        logger.warning(f"Skipping {commodity} on {current_date}: only {len(features_up_to_now)} days")
                    continue
                
                # Train model and generate RAW signal (NO LOOK-AHEAD)
                raw_signal = self._generate_signal_for_date(
                    features_up_to_now,
                    prices_up_to_now,
                    commodity,
                    current_date
                )
                
                # Add to history
                signal_history[commodity].append(raw_signal)
                
                # Smooth using FULL history (this is the fix!)
                if len(signal_history[commodity]) > 1:
                    smoothed_signal = self.signal_smoother.smooth_ema(
                        np.array(signal_history[commodity])
                    )[-1]  # Get last value
                else:
                    smoothed_signal = raw_signal
                
                # Compute realized vol
                returns = prices_up_to_now.pct_change()
                vol = returns.iloc[-20:].std() * np.sqrt(252)
                
                commodity_signals[commodity] = smoothed_signal
                commodity_vols[commodity] = vol
                
                # Log signal changes
                if idx % 50 == 0:
                    logger.info(f"{commodity}: raw={raw_signal:.4f}, smoothed={smoothed_signal:.4f}")
            
            if not commodity_signals:
                continue
            
            # Run council (you have it commented out - enable for governance)
            council_weights = {}
            for commodity, signal in commodity_signals.items():
                council_weights[commodity] = signal
            
            # Construct portfolio
            positions = self.portfolio_constructor.compute_position_sizes(
                council_weights,
                commodity_vols
            )
            
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
                
                # PnL from PRICE CHANGE on existing position
                if abs(old_position) > 1e-6 and self.entry_prices[commodity] > 0:
                    price_change = current_price - self.entry_prices[commodity]
                    pnl = old_position * price_change * 50  # Multiply by point value (adjust per commodity)
                    daily_pnl += pnl
                
                # Track position changes
                if abs(new_position - old_position) > 0.01:  # 1% threshold
                    position_changes.append(f"{commodity}: {old_position:.2f} -> {new_position:.2f}")
                    self.entry_prices[commodity] = current_price
                
                self.positions[commodity] = new_position
                self.entry_prices[commodity] = current_price
            
            # Log position changes
            if position_changes and idx % 50 == 0:
                logger.info(f"Position changes: {position_changes}")
            
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
        if len(results) > 0:
            logger.info(f"Sample result keys: {list(results[0].keys())}")
        else:
            logger.warning("No results collected - check data and model training")

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
            
            # Construct signal (RAW, no smoothing here)
            signal = self.signal_constructor.construct_signal(
                np.array([prediction]),
                prices.iloc[-252:]
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