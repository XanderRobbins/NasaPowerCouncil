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
        
        NOTE: This is a placeholder. In production, connect to your data provider.
        """
        market_data = {}
        
        for commodity in self.commodities:
            # Placeholder: Generate synthetic price data for demo
            dates = pd.date_range(self.start_date, self.end_date, freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
            
            market_data[commodity] = pd.DataFrame({
                'date': dates,
                'close': prices,
                'volume': np.random.randint(10000, 50000, len(dates))
            })
        
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
        
        For each date:
        1. Use only data available up to that date
        2. Train models on historical data
        3. Generate signals
        4. Run council
        5. Construct portfolio
        6. Execute trades
        7. Update PnL
        """
        results = []
        
        # Determine simulation dates (use market data dates as reference)
        sim_dates = market_data[self.commodities[0]]['date']
        
        # Start after sufficient history (e.g., 252 * 10 days for 10-year window)
        min_history = 252 * 10
        start_idx = min(min_history, len(sim_dates) - 252)
        
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
                ]['close']
                
                # Need at least 1 year of data to train
                min_train = 252
                if len(features_up_to_now) < min_train or len(prices_up_to_now) < min_train:
                    if idx == start_idx:
                        logger.warning(f"Skipping {commodity} on {current_date}: only {len(features_up_to_now)} days")
                    continue
                
                # Train model and generate signal (NO LOOK-AHEAD)
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
            
            # Run council for each commodity
            council_weights = {}
            for commodity, signal in commodity_signals.items():
                council_weights[commodity] = signal
                
                #council_context = {
                #   'raw_signal': signal,
                #    'commodity': commodity,
                #    'current_date': current_date,
                #    'features': features_data[commodity],
                #    'prices': market_data[commodity]['close'],
                #    'returns': market_data[commodity]['close'].pct_change()
                #}
                
                #council_result = self.council.evaluate(council_context)
                #council_weights[commodity] = council_result['final_weight']
            
            # Construct portfolio
            positions = self.portfolio_constructor.compute_position_sizes(
                council_weights,
                commodity_vols
            )
            
            # Apply risk management
            portfolio_returns = pd.Series([r['portfolio_return'] for r in results])
            if len(portfolio_returns) > 0:
                positions = self.risk_manager.apply_risk_limits(
                    positions,
                    commodity_signals,
                    self.entry_prices,
                    {c: market_data[c][market_data[c]['date'] == current_date]['close'].iloc[0] 
                     for c in self.commodities},
                    commodity_vols,
                    portfolio_returns
                )
            
            # Update positions and compute PnL
            daily_pnl = 0.0
            for commodity in self.commodities:
                if commodity not in positions:
                    continue
                
                current_price = market_data[commodity][
                    market_data[commodity]['date'] == current_date
                ]['close'].iloc[0]
                
                old_position = self.positions[commodity]
                new_position = positions[commodity]
                
                # PnL from existing position
                if abs(old_position) > 1e-6:
                    entry_price = self.entry_prices[commodity]
                    if old_position > 0:
                        pnl = old_position * (current_price - entry_price)
                    else:
                        pnl = -old_position * (entry_price - current_price)
                    daily_pnl += pnl
                
                # Update position
                if abs(new_position - old_position) > 1e-6:
                    self.entry_prices[commodity] = current_price
                
                self.positions[commodity] = new_position
            
            # Update portfolio value
            self.portfolio_value += daily_pnl
            portfolio_return = daily_pnl / self.portfolio_value if self.portfolio_value > 0 else 0
            
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
        Generate signal for a specific date using only data up to that date.
        """
        # Use RollingRidgeModel (already implements walk-forward)
        model = RollingRidgeModel()
        
        # Prepare features
        X, feature_names = model.prepare_features(features)
        model.feature_names = feature_names
        
        # Compute target
        y = model.compute_target(prices).values
        
        # Train and predict (only using data up to current date)
        try:
            # Simple approach: train on last 10 years, predict next day
            train_window = min(252 * 10, len(X) - 1)
            
            if len(X) < train_window + 1:
                return 0.0
            
            X_train = X[-train_window-1:-1]
            y_train = y[-train_window-1:-1]
            X_test = X[-1:]
            
            # Remove NaNs
            valid_idx = ~np.isnan(y_train)
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]
                        
            X_train = np.nan_to_num(X_train, nan=0.0)  # Replace NaN with 0
            X_test = np.nan_to_num(X_test, nan=0.0) 

            if len(y_train) < 100:
                return 0.0
            
            # Scale and fit
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.model.fit(X_train_scaled, y_train)
            prediction = model.model.predict(X_test_scaled)[0]
            
            # Construct signal
            signal = self.signal_constructor.construct_signal(
                np.array([prediction]),
                prices.iloc[-252:]
            )[0]
            
            # Smooth
            # (In production, maintain smoothing state across calls)
            signal = self.signal_smoother.smooth_ema(np.array([signal]))[0]
            
            return signal
        
        except Exception as e:
            logger.error(f"Error generating signal for {commodity}: {e}")
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