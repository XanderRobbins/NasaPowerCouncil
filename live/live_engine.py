"""
Live trading engine for real-time execution.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import time
from loguru import logger

from data.climate_fetcher import NASAPowerFetcher
from features.feature_pipeline import FeaturePipeline
from models.ridge_model import RollingRidgeModel
from signals.signal_constructor import SignalConstructor
from signals.signal_smoother import SignalSmoother
from council.council_orchestrator import CouncilOrchestrator
from portfolio.constructor import PortfolioConstructor
from portfolio.contract_selector import ContractSelector
from portfolio.risk_manager import RiskManager


class LiveTradingEngine:
    """
    Live trading engine.
    
    Runs on a schedule (e.g., daily after market close) to:
    1. Fetch latest climate data
    2. Update features
    3. Generate signals
    4. Run council
    5. Update positions
    6. Send orders
    """
    
    def __init__(self, 
                 commodities: list,
                 run_frequency: str = 'daily',  # 'daily', 'weekly'
                 paper_trading: bool = True):
        self.commodities = commodities
        self.run_frequency = run_frequency
        self.paper_trading = paper_trading
        
        # Initialize components
        self.climate_fetcher = NASAPowerFetcher()
        self.signal_constructor = SignalConstructor()
        self.signal_smoother = SignalSmoother()
        self.council = CouncilOrchestrator()
        self.portfolio_constructor = PortfolioConstructor()
        self.risk_manager = RiskManager()
        
        # State
        self.current_positions = {c: 0.0 for c in commodities}
        self.target_positions = {c: 0.0 for c in commodities}
        self.models = {}  # Store trained models per commodity
        self.feature_history = {}  # Store feature history
        
        logger.info(f"Live Trading Engine initialized")
        logger.info(f"Commodities: {commodities}")
        logger.info(f"Mode: {'PAPER TRADING' if paper_trading else 'LIVE'}")
    
    def run_daily_update(self):
        """
        Run daily update cycle.
        
        Call this once per day (e.g., via cron job or scheduler).
        """
        logger.info("=" * 80)
        logger.info(f"DAILY UPDATE - {datetime.now()}")
        logger.info("=" * 80)
        
        try:
            # Step 1: Fetch latest data
            logger.info("\n[1/6] Fetching latest climate data...")
            self._fetch_latest_climate_data()
            
            # Step 2: Update features
            logger.info("\n[2/6] Updating features...")
            self._update_features()
            
            # Step 3: Generate signals
            logger.info("\n[3/6] Generating signals...")
            signals = self._generate_signals()
            
            # Step 4: Run council
            logger.info("\n[4/6] Running council evaluation...")
            council_results = self._run_council(signals)
            
            # Step 5: Construct portfolio
            logger.info("\n[5/6] Constructing portfolio...")
            self.target_positions = self._construct_portfolio(council_results)
            
            # Step 6: Execute trades
            logger.info("\n[6/6] Executing trades...")
            self._execute_trades()
            
            logger.info("\n" + "=" * 80)
            logger.info("DAILY UPDATE COMPLETE")
            logger.info("=" * 80 + "\n")
            
        except Exception as e:
            logger.error(f"Error in daily update: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _fetch_latest_climate_data(self):
        """Fetch latest climate data for all regions."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Fetch last 7 days
        
        for commodity in self.commodities:
            logger.info(f"Fetching data for {commodity}...")
            
            region_data = self.climate_fetcher.fetch_commodity_regions(
                commodity,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            # Store or update
            if commodity not in self.feature_history:
                self.feature_history[commodity] = {}
            
            for region, df in region_data.items():
                if region in self.feature_history[commodity]:
                    # Append new data
                    existing = self.feature_history[commodity][region]
                    combined = pd.concat([existing, df]).drop_duplicates(subset=['date']).sort_values('date')
                    self.feature_history[commodity][region] = combined
                else:
                    self.feature_history[commodity][region] = df
    
    def _update_features(self):
        """Update features for all commodities."""
        for commodity in self.commodities:
            if commodity not in self.feature_history:
                continue
            
            logger.info(f"Updating features for {commodity}...")
            
            pipeline = FeaturePipeline(commodity)
            features = pipeline.run(self.feature_history[commodity])
            
            # Store features
            if commodity not in self.feature_history:
                self.feature_history[commodity] = {}
            
            self.feature_history[commodity]['aggregated_features'] = features
    
    def _generate_signals(self) -> Dict[str, float]:
        """Generate signals for all commodities."""
        signals = {}
        
        for commodity in self.commodities:
            if commodity not in self.feature_history:
                signals[commodity] = 0.0
                continue
            
            # Get features
            features = self.feature_history[commodity].get('aggregated_features')
            if features is None:
                signals[commodity] = 0.0
                continue
            
            # Get prices (placeholder - connect to your data provider)
            prices = self._fetch_prices(commodity)
            
            # Train/update model
            if commodity not in self.models:
                self.models[commodity] = RollingRidgeModel()
            
            model = self.models[commodity]
            
            # Prepare features
            X, feature_names = model.prepare_features(features)
            model.feature_names = feature_names
            
            # Compute target
            y = model.compute_target(prices).values
            
            # Train on all available data
            valid_idx = ~np.isnan(y)
            X_train = X[valid_idx][:-1]  # Exclude last (no target yet)
            y_train = y[valid_idx][:-1]
            X_latest = X[-1:]
            
            if len(y_train) < 100:
                signals[commodity] = 0.0
                continue
            
            # Scale and fit
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_latest_scaled = scaler.transform(X_latest)
            
            model.model.fit(X_train_scaled, y_train)
            prediction = model.model.predict(X_latest_scaled)[0]
            
            # Construct signal
            signal = self.signal_constructor.construct_signal(
                np.array([prediction]),
                prices.iloc[-252:]
            )[0]
            
            # Smooth (maintain state across calls)
            signal = self.signal_smoother.smooth_ema(np.array([signal]))[0]
            
            signals[commodity] = signal
            
            logger.info(f"{commodity}: signal = {signal:.4f}")
        
        return signals
    
    def _run_council(self, signals: Dict[str, float]) -> Dict[str, Dict]:
        """Run council for all commodities."""
        council_results = {}
        
        for commodity, signal in signals.items():
            features = self.feature_history[commodity].get('aggregated_features')
            prices = self._fetch_prices(commodity)
            
            context = {
                'raw_signal': signal,
                'commodity': commodity,
                'current_date': datetime.now(),
                'features': features,
                'prices': prices,
                'returns': prices.pct_change() if prices is not None else None
            }
            
            result = self.council.evaluate(context)
            council_results[commodity] = result
            
            logger.info(f"{commodity}: final_weight = {result['final_weight']:.4f} (decision: {result['decision']})")
        
        return council_results
    
    def _construct_portfolio(self, council_results: Dict[str, Dict]) -> Dict[str, float]:
        """Construct portfolio from council results."""
        signals = {c: r['final_weight'] for c, r in council_results.items()}
        
        # Get realized vols
        vols = {}
        for commodity in self.commodities:
            prices = self._fetch_prices(commodity)
            if prices is not None:
                returns = prices.pct_change()
                vol = returns.iloc[-20:].std() * np.sqrt(252)
                vols[commodity] = vol
            else:
                vols[commodity] = 0.15  # Default
        
        # Construct portfolio
        positions = self.portfolio_constructor.compute_position_sizes(signals, vols)
        
        return positions
    
    def _execute_trades(self):
        """Execute trades to reach target positions."""
        for commodity in self.commodities:
            current = self.current_positions[commodity]
            target = self.target_positions[commodity]
            
            trade_size = target - current
            
            if abs(trade_size) < 0.01:
                continue
            
            # Select contract
            selector = ContractSelector(commodity)
            contract = selector.select_contract('current_season', datetime.now())
            
            # Execute
            if self.paper_trading:
                logger.info(f"[PAPER] {commodity} ({contract}): {'BUY' if trade_size > 0 else 'SELL'} {abs(trade_size):.2f} contracts")
            else:
                # Connect to your broker API here
                self._send_order(commodity, contract, trade_size)
            
            # Update current position
            self.current_positions[commodity] = target
    
    def _fetch_prices(self, commodity: str) -> Optional[pd.Series]:
        """
        Fetch latest prices.
        
        PLACEHOLDER - Connect to your data provider (e.g., Quandl, Polygon, IB, etc.)
        """
        # For now, return synthetic data
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01)), index=dates)
        return prices
    
    def _send_order(self, commodity: str, contract: str, size: float):
        """
        Send order to broker.
        
        PLACEHOLDER - Implement with your broker's API (Interactive Brokers, TD Ameritrade, etc.)
        """
        logger.info(f"[LIVE] Sending order: {commodity} ({contract}) size={size:.2f}")
        # order = broker.create_order(contract, size, ...)
        pass
    
    def start_scheduler(self, run_time: str = "16:30"):
        """
        Start scheduler to run daily at specified time.
        
        Args:
            run_time: Time to run (HH:MM format, 24-hour)
        """
        import schedule
        
        schedule.every().day.at(run_time).do(self.run_daily_update)
        
        logger.info(f"Scheduler started. Will run daily at {run_time}")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def start_live_trading(commodities: list, 
                      paper_trading: bool = True,
                      run_time: str = "16:30"):
    """
    Start live trading engine.
    
    Args:
        commodities: List of commodities to trade
        paper_trading: If True, paper trade only
        run_time: Daily run time (after market close)
    """
    engine = LiveTradingEngine(commodities, paper_trading=paper_trading)
    engine.start_scheduler(run_time=run_time)