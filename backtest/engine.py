"""
Walk-forward backtesting engine with NO look-ahead bias.
Simplified: No council, no smoothing, cached models.
Per-commodity growing season windows supported.
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from sklearn.preprocessing import StandardScaler

from data.climate_fetcher import NASAPowerFetcher
from data.market_fetcher import MarketDataFetcher
from features.feature_pipeline import FeaturePipeline
from models.ridge_model import RollingRidgeModel
from models.classifier_model import DirectionalClassifier
from config.settings import (
    os,
    GROWING_SEASON_MONTHS,
    ONLY_TRADE_GROWING_SEASON,
    STRATEGY_MODE,
    FIXED_POSITION_SIZE,
    TARGET_PORTFOLIO_VOL,
    MIN_SIGNAL_STRENGTH,
    MAX_SINGLE_POSITION,
    TRAIN_WINDOW_YEARS,
    COMMODITY_TRADE_MONTHS,
    COMMODITY_VOL_THRESHOLDS
)


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Critical: NO look-ahead bias. At each step, only use data available UP TO that date.
    Per-commodity season windows: each commodity trades only during its relevant months.
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
        self.entry_price_at_open = {c: 0.0 for c in commodities}  # Track true entry price for stop loss
        self.portfolio_value = initial_capital

        # FIX: prediction_history moved here from lazy init in _generate_signal_for_date
        self.prediction_history: Dict[str, list] = {}

        # Per-commodity season tracking
        self.was_in_growing_season = {c: False for c in commodities}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

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
        """Fetch market data for all commodities."""
        market_data = {}

        for commodity in self.commodities:
            logger.info(f"Fetching market data for {commodity}...")
            df = self.market_fetcher.fetch_futures_data(
                commodity,
                'N/A',
                self.start_date.strftime('%Y-%m-%d'),
                self.end_date.strftime('%Y-%m-%d')
            )
            market_data[commodity] = df

        return market_data

    def _generate_all_features(self,
                                climate_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """Generate features for all commodities (parallel across commodities)."""
        def _run_pipeline(commodity, region_data):
            logger.info(f"Generating features for {commodity}...")
            pipeline = FeaturePipeline(commodity)
            return commodity, pipeline.run(region_data)

        features_data = {}
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_run_pipeline, c, d) for c, d in climate_data.items()]
            for future in futures:
                commodity, features = future.result()
                features_data[commodity] = features

        return features_data

    # ------------------------------------------------------------------
    # Walk-forward simulation
    # ------------------------------------------------------------------

    def _walk_forward_simulation(self,
                                  features_data: Dict[str, pd.DataFrame],
                                  market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Walk-forward simulation — no look-ahead, no smoothing, fixed P&L.
        Positions are explicitly closed per-commodity when their season ends.
        """
        results = []

        # Determine simulation dates from first commodity [1]
        sim_dates = market_data[self.commodities[0]]['date']

        # Start after sufficient history
        min_history = 252 * TRAIN_WINDOW_YEARS
        start_idx = min(min_history, len(sim_dates) - 252)

        logger.info(f"Starting simulation at index {start_idx} / {len(sim_dates)}")

        # Pre-index market data by date for O(1) lookups
        market_by_date = {
            c: df.set_index('date') for c, df in market_data.items()
        }

        # Pre-sort features DataFrames and store their date arrays for fast slicing
        features_dates = {c: df['date'].values for c, df in features_data.items()}
        market_dates   = {c: df['date'].values for c, df in market_data.items()}

        # Track whether last day was in growing season (for explicit position close)
        was_in_growing_season = False

        for idx in range(start_idx, len(sim_dates)):
            current_date = sim_dates.iloc[idx]

            # Log progress every 100 days
            if idx % 100 == 0:
                logger.info(
                    f"Processing {current_date.date()} ({idx}/{len(sim_dates)}) | "
                    f"Portfolio: ${self.portfolio_value:,.0f}"
                )

            # --- Per-commodity growing season filter ---
            active_commodities = []

            for commodity in self.commodities:
                trade_months = COMMODITY_TRADE_MONTHS.get(commodity, [])
                in_season = current_date.month in trade_months

                if ONLY_TRADE_GROWING_SEASON and not in_season:
                    # Close position if we just left the season
                    if self.was_in_growing_season[commodity]:
                        logger.info(
                            f"{commodity} season ended at {current_date.date()} "
                            f"— closing position."
                        )
                        self.positions[commodity] = 0.0
                        self.entry_prices[commodity] = 0.0
                        self.was_in_growing_season[commodity] = False
                else:
                    self.was_in_growing_season[commodity] = True
                    active_commodities.append(commodity)

            # Skip signal generation if no commodities are active
            if not active_commodities:
                results.append({
                    'date': current_date,
                    'portfolio_value': self.portfolio_value,
                    'daily_pnl': 0.0,
                    'portfolio_return': 0.0,
                    **{f'{c}_signal': 0.0 for c in self.commodities},
                    **{f'{c}_position': 0.0 for c in self.commodities}
                })
                continue

            # --- Generate signals for active commodities only ---
            commodity_signals = {}
            commodity_vols = {}

            for commodity in self.commodities:
                # Skip commodity if outside its trading months
                if current_date.month not in COMMODITY_TRADE_MONTHS[commodity]:
                    continue

                # Data available up to current date only (no look-ahead)
                # Use searchsorted for fast date slicing instead of boolean masks
                ts = np.datetime64(current_date)
                f_cut = int(np.searchsorted(features_dates[commodity], ts, side='right'))
                m_cut = int(np.searchsorted(market_dates[commodity],   ts, side='right'))
                features_up_to_now = features_data[commodity].iloc[:f_cut]
                prices_up_to_now   = market_data[commodity].iloc[:m_cut]

                # Align on common dates
                common_dates = pd.merge(
                    features_up_to_now[['date']],
                    prices_up_to_now[['date']],
                    on='date',
                    how='inner'
                )['date']

                features_up_to_now = features_up_to_now[
                    features_up_to_now['date'].isin(common_dates)
                ]
                prices_up_to_now = prices_up_to_now[
                    prices_up_to_now['date'].isin(common_dates)
                ]['close']

                # Need at least 1 year of data
                if len(features_up_to_now) < 252 or len(prices_up_to_now) < 252:
                    continue

                signal = self._generate_signal_for_date(
                    features_up_to_now,
                    prices_up_to_now,
                    commodity,
                    current_date
                )
                # Realized vol (20-day)
                returns = prices_up_to_now.pct_change()
                vol = returns.iloc[-20:].std() * np.sqrt(252)

                # Volatility regime filter: skip trade if vol exceeds threshold
                if vol > COMMODITY_VOL_THRESHOLDS[commodity]:
                    signal = 0.0

                commodity_signals[commodity] = signal
                commodity_vols[commodity] = vol

            if not commodity_signals:
                results.append({
                    'date': current_date,
                    'portfolio_value': self.portfolio_value,
                    'daily_pnl': 0.0,
                    'portfolio_return': 0.0,
                    **{f'{c}_signal': 0.0 for c in self.commodities},
                    **{f'{c}_position': self.positions.get(c, 0) for c in self.commodities}
                })
                continue

            # --- Position sizing ---
            positions = {}
            for commodity, signal in commodity_signals.items():
                if STRATEGY_MODE == 'directional':
                    # FIX: raised threshold to 1.0 — only trade high-conviction signals
                    if abs(signal) >= MIN_SIGNAL_STRENGTH:
                        positions[commodity] = np.sign(signal) * FIXED_POSITION_SIZE
                    else:
                        positions[commodity] = 0.0
                else:
                    # Magnitude-based (vol-scaled)
                    positions[commodity] = signal * (
                        TARGET_PORTFOLIO_VOL / (commodity_vols[commodity] + 1e-8)
                    )

                # Apply position caps
                positions[commodity] = np.clip(
                    positions[commodity], -MAX_SINGLE_POSITION, MAX_SINGLE_POSITION
                )

            # --- P&L calculation ---
            daily_pnl = 0.0

            for commodity in self.commodities:
                if commodity not in positions:
                    continue

                if current_date not in market_by_date[commodity].index:
                    continue

                current_price = market_by_date[commodity].at[current_date, 'close']
                old_position = self.positions[commodity]
                new_position = positions[commodity]
                exit_price = current_price  # Default: exit at market

                # --- Stop loss check (5% per-position) ---
                if abs(old_position) > 1e-6 and self.entry_price_at_open[commodity] > 0:
                    unrealized_loss = (current_price - self.entry_price_at_open[commodity]) / self.entry_price_at_open[commodity]
                    if unrealized_loss < -0.05:  # Loss exceeds 5%
                        new_position = 0.0  # Force close the position

                if abs(old_position) > 1e-6 and self.entry_prices[commodity] > 0:
                    # Daily mark-to-market return on existing position
                    daily_return = (
                        (current_price - self.entry_prices[commodity])
                        / self.entry_prices[commodity]
                    )
                    pnl = self.portfolio_value * old_position * daily_return
                    daily_pnl += pnl

                # Record entry price when position opens (transitions from ~0 to non-0)
                if abs(old_position) < 1e-6 and abs(new_position) > 1e-6:
                    self.entry_price_at_open[commodity] = current_price

                # Reset entry price when position closes
                if abs(new_position) < 1e-6:
                    self.entry_price_at_open[commodity] = 0.0

                # Update to today's price and new position
                self.entry_prices[commodity] = current_price
                self.positions[commodity] = new_position

            # Update portfolio value
            prev_value = (
                self.portfolio_value - daily_pnl
                if (self.portfolio_value - daily_pnl) > 0
                else self.portfolio_value
            )
            self.portfolio_value += daily_pnl
            portfolio_return = daily_pnl / prev_value if prev_value > 0 else 0.0

            results.append({
                'date': current_date,
                'portfolio_value': self.portfolio_value,
                'daily_pnl': daily_pnl,
                'portfolio_return': portfolio_return,
                **{f'{c}_signal': commodity_signals.get(c, 0) for c in self.commodities},
                **{f'{c}_position': positions.get(c, 0) for c in self.commodities}
            })

        logger.info(f"Collected {len(results)} result records")
        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal_for_date(self,
                                   features: pd.DataFrame,
                                   prices: pd.Series,
                                   commodity: str,
                                   current_date: pd.Timestamp) -> float:
        """
        Generate signal for a specific date.

        FIX: Classifier used as direction confirmation filter.
        FIX: prediction_history initialized in __init__, not here.

        Returns:
            Z-scored signal, capped at ±3.0
        """
        # Cache key: retrain monthly
        cache_key = f"{commodity}_{current_date.year}_{current_date.month}"

        if cache_key in self.model_cache:
            ridge_model, clf_model, scaler, selected_features = self.model_cache[cache_key]

            X = features[selected_features].fillna(0).values
            if len(X) == 0:
                return 0.0

            X_test = X[-1:]
            X_test_scaled = scaler.transform(X_test)

            ridge_prediction = ridge_model.model.predict(X_test_scaled)[0]
            clf_direction = clf_model.model.predict(X_test_scaled)[0]

        else:
            # --- Train Ridge model ---
            ridge_model = RollingRidgeModel()
            y = ridge_model.compute_target(prices).values
            y_series = pd.Series(y, index=range(len(y)))

            X, feature_names = ridge_model.prepare_features(features, target=y_series)
            ridge_model.feature_names = feature_names

            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]

            train_window = min(252 * TRAIN_WINDOW_YEARS, len(X) - 1)
            if len(X) < train_window + 1:
                return 0.0

            X_train = X[-train_window - 1:-1]
            y_train = y[-train_window - 1:-1]
            X_test = X[-1:]

            valid_idx = ~np.isnan(y_train)
            X_train = X_train[valid_idx]
            y_train = y_train[valid_idx]

            if len(y_train) < 252:
                return 0.0

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            ridge_model.model.fit(X_train_scaled, y_train)
            ridge_prediction = ridge_model.model.predict(X_test_scaled)[0]

            # --- Train Classifier on same window ---
            clf_model = DirectionalClassifier()
            y_binary = (y_train > 0).astype(int)

            # Guard against degenerate class distributions
            if len(np.unique(y_binary)) < 2:
                clf_direction = 1 if ridge_prediction > 0 else 0
            else:
                clf_model.model.fit(X_train_scaled, y_binary)
                clf_direction = clf_model.model.predict(X_test_scaled)[0]

            # Cache ridge, classifier, scaler, and feature names
            self.model_cache[cache_key] = (ridge_model, clf_model, scaler, feature_names)

        # --- Classifier confirmation filter ---
        # Only trade when ridge magnitude and classifier direction agree
        ridge_is_long = ridge_prediction > 0
        clf_is_long = clf_direction == 1

        if ridge_is_long != clf_is_long:
            clf_proba = clf_model.model.predict_proba(X_test_scaled)[0]
            clf_confidence = max(clf_proba)

            # Only block if classifier is CONFIDENTLY disagreeing
            if clf_confidence > 0.65:
                return 0.0

        # --- Z-score the prediction using rolling history ---
        hist_key = f"{commodity}_preds"
        if hist_key not in self.prediction_history:
            self.prediction_history[hist_key] = []

        self.prediction_history[hist_key].append(ridge_prediction)
        pred_history = self.prediction_history[hist_key][-60:]  # Last 60 months

        if len(pred_history) >= 5:
            pred_std = np.std(pred_history) + 1e-8
            pred_mean = np.mean(pred_history)
            signal = (ridge_prediction - pred_mean) / pred_std
        else:
            # Fallback for early periods
            returns = prices.pct_change().dropna()
            vol = returns.iloc[-20:].std() * np.sqrt(252)
            signal = ridge_prediction / (vol + 1e-8)

        signal = np.clip(signal, -3.0, 3.0)
        return signal


# ------------------------------------------------------------------
# Convenience function
# ------------------------------------------------------------------

def run_backtest(start_date: str,
                 end_date: str,
                 commodities: List[str],
                 initial_capital: float = 1_000_000) -> pd.DataFrame:
    """Convenience function to run backtest."""
    engine = BacktestEngine(start_date, end_date, commodities, initial_capital)
    return engine.run()