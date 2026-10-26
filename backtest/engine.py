"""
Walk-forward backtest engine.

For every simulated day and every commodity that is in season, the engine

1. trains (or reuses) a model using only information that was available at that time,
2. turns the model's forecast into a signal, and the signal into a target position,
3. trades to that position, charging costs and applying a stop-loss, and
4. marks the portfolio to market.

No look-ahead: each day's model sees only rows up to that day, and a training row's label
(the return N days later) is only used once those N days have passed.

Timing assumptions (both on by default, see ``config/settings.py``):

* ``FEATURE_LAG_DAYS``: weather is published with a delay, so day d's features are only
  usable from day d + lag.
* ``EXECUTE_NEXT_BAR``: a decision made at today's close is traded at tomorrow's close.

Output: one row per simulated day with portfolio value, P&L, costs, return and each
commodity's signal and position. Column names are the contract with ``backtest/metrics.py``
and ``backtest/plots.py``.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import (
    COMMODITY_TRADE_MONTHS,
    COMMODITY_VOL_THRESHOLDS,
    EXECUTE_NEXT_BAR,
    FEATURE_LAG_DAYS,
    FIXED_POSITION_SIZE,
    MAX_SINGLE_POSITION,
    MIN_SIGNAL_STRENGTH,
    MODEL_TYPE,
    ONLY_TRADE_GROWING_SEASON,
    PCA_N_COMPONENTS,
    RIDGE_ALPHA,
    SIGNAL_CAP,
    STOP_LOSS_PCT,
    STRATEGY_MODE,
    TARGET_PORTFOLIO_VOL,
    TRAIN_WINDOW_YEARS,
    TRANSACTION_COST_BPS,
    USE_PCA,
    VETO_CONFIDENCE,
)
from data.climate_fetcher import NASAPowerFetcher
from data.market_fetcher import MarketDataFetcher
from features.feature_pipeline import FeaturePipeline
from models.classifier_model import DirectionalClassifier
from models.ridge_model import RollingRidgeModel

TRADING_DAYS_PER_YEAR = 252
MIN_HISTORY_ROWS = TRADING_DAYS_PER_YEAR   # no signal until a year of aligned data exists
FLAT = 1e-6                                # |position| below this counts as "no position"
SIGNAL_HISTORY_LEN = 60                    # forecasts used to standardise a new forecast
MIN_SIGNAL_HISTORY = 5                     # below this many forecasts, fall back to forecast / vol


def _build_regression_model():
    """A fresh, unfitted regression model for ``MODEL_TYPE`` ('ridge' or 'xgboost')."""
    if MODEL_TYPE == 'xgboost':
        from models.xgboost_model import XGBoostRegressionModel  # optional dependency
        return XGBoostRegressionModel()
    if MODEL_TYPE == 'ridge':
        return RollingRidgeModel(alpha=RIDGE_ALPHA)
    raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE!r}. Expected 'ridge' or 'xgboost'.")


class BacktestEngine:
    """Walk-forward backtest over one or more commodities (see module docstring)."""

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 commodities: List[str],
                 initial_capital: float = 1_000_000):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.commodities = commodities
        self.initial_capital = initial_capital

        self.climate_fetcher = NASAPowerFetcher()
        self.market_fetcher = MarketDataFetcher()

        # Fitted models are cached per (commodity, year, month): retrained monthly, not daily.
        self.model_cache = {}
        # Recent raw forecasts per commodity, used to standardise the newest forecast.
        self.prediction_history: Dict[str, list] = {}

        # Per-commodity state.
        self.portfolio_value = initial_capital
        self.positions = {c: 0.0 for c in commodities}            # signed fraction of portfolio held now
        self.entry_prices = {c: 0.0 for c in commodities}         # last close; base for tomorrow's mark-to-market
        self.entry_price_at_open = {c: 0.0 for c in commodities}  # price when the position was opened; stop reference
        self.pending_target = {c: 0.0 for c in commodities}       # decided yesterday, traded today (EXECUTE_NEXT_BAR)
        self.was_in_growing_season = {c: False for c in commodities}

    def _in_season(self, commodity: str, date: pd.Timestamp) -> bool:
        """True when ``commodity`` may hold a position on ``date`` (always, if the season filter is off)."""
        if not ONLY_TRADE_GROWING_SEASON:
            return True
        return date.month in COMMODITY_TRADE_MONTHS.get(commodity, [])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Fetch data, build features and simulate. Returns the daily results table."""
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Commodities: {self.commodities}")
        logger.info(f"Initial Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Strategy Mode: {STRATEGY_MODE}")
        logger.info("=" * 80)

        logger.info("Step 1: Fetching climate data...")
        climate_data = self._fetch_all_climate_data()

        logger.info("Step 2: Fetching market data...")
        market_data = self._fetch_all_market_data()

        logger.info("Step 3: Generating features...")
        features_data = self._generate_all_features(climate_data)

        logger.info("Step 4: Running walk-forward simulation...")
        results = self._walk_forward_simulation(features_data, market_data)

        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Total Return: {(self.portfolio_value / self.initial_capital - 1) * 100:.2f}%")
        logger.info("=" * 80)
        return results

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_all_climate_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """commodity -> region -> raw daily weather."""
        start, end = self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')
        all_data = {}
        for commodity in self.commodities:
            logger.info(f"Fetching climate data for {commodity}...")
            all_data[commodity] = self.climate_fetcher.fetch_commodity_regions(commodity, start, end)
        return all_data

    def _fetch_all_market_data(self) -> Dict[str, pd.DataFrame]:
        """commodity -> daily futures prices."""
        start, end = self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')
        market_data = {}
        for commodity in self.commodities:
            logger.info(f"Fetching market data for {commodity}...")
            market_data[commodity] = self.market_fetcher.fetch_futures_data(commodity, start, end)
        return market_data

    def _generate_all_features(self,
                               climate_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """commodity -> feature table, with the publication lag applied."""
        def _run_pipeline(commodity, region_data):
            logger.info(f"Generating features for {commodity}...")
            return commodity, FeaturePipeline(commodity).run(region_data)

        features_data = {}
        with ThreadPoolExecutor() as executor:  # commodities are independent
            futures = [executor.submit(_run_pipeline, c, d) for c, d in climate_data.items()]
            for future in futures:
                commodity, features = future.result()
                if FEATURE_LAG_DAYS > 0:
                    # Weather for day d is only published ~FEATURE_LAG_DAYS later, so the
                    # row computed from data through d becomes usable on d + lag.
                    features = features.copy()
                    features['date'] = features['date'] + pd.Timedelta(days=FEATURE_LAG_DAYS)
                features_data[commodity] = features
        return features_data

    # ------------------------------------------------------------------
    # Walk-forward simulation
    # ------------------------------------------------------------------

    def _walk_forward_simulation(self,
                                 features_data: Dict[str, pd.DataFrame],
                                 market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Loop over trading days; see the module docstring for what happens each day."""
        results = []

        # Simulation calendar: the first commodity's trading days, starting once a full
        # training window of history exists (or one year before the end, if data is short).
        sim_dates = market_data[self.commodities[0]]['date']
        start_idx = min(TRADING_DAYS_PER_YEAR * TRAIN_WINDOW_YEARS, len(sim_dates) - TRADING_DAYS_PER_YEAR)
        logger.info(f"Starting simulation at index {start_idx} / {len(sim_dates)}")

        market_by_date = {c: df.set_index('date') for c, df in market_data.items()}
        # Sorted date arrays let each day slice "everything up to today" with a binary search.
        features_dates = {c: df['date'].values for c, df in features_data.items()}
        market_dates = {c: df['date'].values for c, df in market_data.items()}

        for idx in range(start_idx, len(sim_dates)):
            current_date = sim_dates.iloc[idx]
            if idx % 100 == 0:
                logger.info(f"Processing {current_date.date()} ({idx}/{len(sim_dates)}) | "
                            f"Portfolio: ${self.portfolio_value:,.0f}")

            # 1) Season bookkeeping: flatten anything whose season just ended.
            active, season_end_costs = self._update_seasons(current_date)

            # 2) Nothing in season today: only the closing cost (if any) hits the account.
            if not active:
                results.append(self._idle_row(current_date, season_end_costs, hold_positions=False))
                continue

            # 3) Signals for every in-season commodity.
            signals, vols = self._compute_signals(current_date, features_data, market_data,
                                                  features_dates, market_dates)
            if not signals:  # e.g. not enough aligned history yet
                results.append(self._idle_row(current_date, season_end_costs, hold_positions=True))
                continue

            # 4) Signals -> target positions -> trades and P&L.
            targets = self._target_positions(signals, vols)
            results.append(self._trade_and_mark(current_date, signals, targets,
                                                market_by_date, season_end_costs))

        logger.info(f"Collected {len(results)} result records")
        return pd.DataFrame(results)

    def _update_seasons(self, current_date: pd.Timestamp) -> Tuple[List[str], float]:
        """
        Close positions in commodities whose season has just ended.

        Returns (commodities in season today, transaction cost of the season-end closes).
        The close happens at the previous in-season close, so the first out-of-season day's
        price move is (correctly) not earned; only the closing cost is charged, and it is
        charged even when no commodity is in season today.
        """
        active = []
        closed_positions = {}
        for commodity in self.commodities:
            if self._in_season(commodity, current_date):
                self.was_in_growing_season[commodity] = True
                active.append(commodity)
            elif self.was_in_growing_season[commodity]:
                logger.info(f"{commodity} season ended at {current_date.date()} — closing position.")
                closed_positions[commodity] = self.positions[commodity]
                self.positions[commodity] = 0.0
                self.entry_prices[commodity] = 0.0
                self.entry_price_at_open[commodity] = 0.0
                self.pending_target[commodity] = 0.0
                self.was_in_growing_season[commodity] = False

        season_end_costs = sum(
            self.portfolio_value * abs(pos) * (TRANSACTION_COST_BPS / 10000.0)
            for pos in closed_positions.values() if abs(pos) > FLAT
        )
        return active, season_end_costs

    def _idle_row(self, current_date: pd.Timestamp, costs: float, hold_positions: bool) -> dict:
        """
        A day with no trading. Only season-end closing costs (possibly 0) are charged.

        ``hold_positions`` reports the positions currently held (in season but no signal
        yet) instead of zeros (out of season, everything is flat).
        """
        start_value = self.portfolio_value
        self.portfolio_value -= costs
        return {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': 0.0,
            'transaction_costs': costs,
            'portfolio_return': -costs / start_value if start_value > 0 else 0.0,
            **{f'{c}_signal': 0.0 for c in self.commodities},
            **{f'{c}_position': (self.positions.get(c, 0) if hold_positions else 0.0) for c in self.commodities},
        }

    def _compute_signals(self, current_date, features_data, market_data,
                         features_dates, market_dates) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Signal and realised volatility for each in-season commodity that has enough history.

        Only rows dated on or before ``current_date`` are visible, and features and prices are
        aligned on their common dates.
        """
        signals, vols = {}, {}
        for commodity in self.commodities:
            if not self._in_season(commodity, current_date):
                continue

            # Everything up to and including today (binary search on the sorted dates).
            ts = np.datetime64(current_date)
            f_cut = int(np.searchsorted(features_dates[commodity], ts, side='right'))
            m_cut = int(np.searchsorted(market_dates[commodity], ts, side='right'))
            features_so_far = features_data[commodity].iloc[:f_cut]
            prices_so_far = market_data[commodity].iloc[:m_cut]

            # Keep only dates present in both tables so row i of features is row i of prices.
            common_dates = pd.merge(features_so_far[['date']], prices_so_far[['date']],
                                    on='date', how='inner')['date']
            features_so_far = features_so_far[features_so_far['date'].isin(common_dates)]
            prices_so_far = prices_so_far[prices_so_far['date'].isin(common_dates)]['close']

            if len(features_so_far) < MIN_HISTORY_ROWS or len(prices_so_far) < MIN_HISTORY_ROWS:
                continue

            signal = self._generate_signal_for_date(features_so_far, prices_so_far, commodity, current_date)

            # 20-day annualised realised volatility.
            vol = prices_so_far.pct_change().iloc[-20:].std() * np.sqrt(TRADING_DAYS_PER_YEAR)

            # Volatility-regime filter: stand aside when the market is too turbulent.
            if vol > COMMODITY_VOL_THRESHOLDS[commodity]:
                signal = 0.0

            signals[commodity] = signal
            vols[commodity] = vol
        return signals, vols

    def _target_positions(self, signals: Dict[str, float], vols: Dict[str, float]) -> Dict[str, float]:
        """
        Signal -> position size (signed fraction of portfolio value).

        * directional: +/- FIXED_POSITION_SIZE by the signal's sign, flat if |signal| is tiny;
        * magnitude:   signal scaled by TARGET_PORTFOLIO_VOL / realised volatility.
        Positions are capped at +/- MAX_SINGLE_POSITION. With EXECUTE_NEXT_BAR the position
        returned is the one decided on the previous day; today's decision is stored for tomorrow.
        """
        targets = {}
        for commodity, signal in signals.items():
            if STRATEGY_MODE == 'directional':
                target = np.sign(signal) * FIXED_POSITION_SIZE if abs(signal) >= MIN_SIGNAL_STRENGTH else 0.0
            else:
                target = signal * (TARGET_PORTFOLIO_VOL / (vols[commodity] + 1e-8))
            target = np.clip(target, -MAX_SINGLE_POSITION, MAX_SINGLE_POSITION)

            if EXECUTE_NEXT_BAR:
                decided_today = target
                target = self.pending_target[commodity]
                self.pending_target[commodity] = decided_today
            targets[commodity] = target
        return targets

    def _trade_and_mark(self, current_date, signals, targets, market_by_date, season_end_costs) -> dict:
        """
        Mark existing positions to today's close, apply the stop-loss, trade to the targets
        and settle the day's P&L and costs against the start-of-day portfolio value.
        """
        daily_pnl = 0.0
        start_value = self.portfolio_value
        transaction_costs = season_end_costs

        for commodity in self.commodities:
            if commodity not in targets:
                continue
            if current_date not in market_by_date[commodity].index:
                continue  # no price today for this commodity

            current_price = market_by_date[commodity].at[current_date, 'close']
            old_position = self.positions[commodity]
            new_position = targets[commodity]

            # Stop-loss: measured from the price at which this position was opened, signed by
            # direction (for a short, a rising price is the loss). Forces the position flat.
            if abs(old_position) > FLAT and self.entry_price_at_open[commodity] > 0:
                move = (current_price - self.entry_price_at_open[commodity]) / self.entry_price_at_open[commodity]
                if (move if old_position > 0 else -move) < -STOP_LOSS_PCT:
                    new_position = 0.0

            # Daily mark-to-market: yesterday's close -> today's close on yesterday's position.
            if abs(old_position) > FLAT and self.entry_prices[commodity] > 0:
                daily_return = (current_price - self.entry_prices[commodity]) / self.entry_prices[commodity]
                daily_pnl += start_value * old_position * daily_return

            # Cost of changing the position.
            position_change = abs(new_position - old_position)
            if position_change > FLAT:
                transaction_costs += start_value * position_change * (TRANSACTION_COST_BPS / 10000.0)

            # Stop reference: set when a position opens or flips side, cleared when it closes.
            flipped = (abs(old_position) > FLAT and abs(new_position) > FLAT
                       and (old_position > 0) != (new_position > 0))
            if (abs(old_position) < FLAT and abs(new_position) > FLAT) or flipped:
                self.entry_price_at_open[commodity] = current_price
            if abs(new_position) < FLAT:
                self.entry_price_at_open[commodity] = 0.0

            self.entry_prices[commodity] = current_price
            self.positions[commodity] = new_position

        self.portfolio_value += daily_pnl - transaction_costs
        portfolio_return = (daily_pnl - transaction_costs) / start_value if start_value > 0 else 0.0

        return {
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'daily_pnl': daily_pnl,
            'transaction_costs': transaction_costs,
            'portfolio_return': portfolio_return,
            **{f'{c}_signal': signals.get(c, 0) for c in self.commodities},
            **{f'{c}_position': targets.get(c, 0) for c in self.commodities},
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _generate_signal_for_date(self, features: pd.DataFrame, prices: pd.Series,
                                  commodity: str, current_date: pd.Timestamp) -> float:
        """
        Today's signal for one commodity, from the model for the current month.

        1. Get today's forecast and the direction classifier's view (training the models the
           first time a month is seen, otherwise reusing them).
        2. Standardise the forecast against the last SIGNAL_HISTORY_LEN forecasts. Note the
           signal is therefore centred on the recent *average* forecast, not on zero.
        3. If the classifier confidently (> VETO_CONFIDENCE) disagrees with the regression's
           direction, return 0 (no trade). Otherwise clip to +/- SIGNAL_CAP.
        """
        cache_key = f"{commodity}_{current_date.year}_{current_date.month}"
        fitted = self.model_cache.get(cache_key)
        if fitted is None:
            fitted = self._train_models(features, prices, commodity, current_date)
            if fitted is None:
                return 0.0  # not enough usable training data yet
            self.model_cache[cache_key] = fitted
        ridge_model, clf_model, scaler, pca, selected_features = fitted

        # Today's feature row, scaled (and PCA-rotated) exactly like the training rows.
        X_today = features[selected_features].fillna(0).values[-1:]
        X_today_scaled = scaler.transform(X_today)
        if pca is not None:
            X_today_scaled = pca.transform(X_today_scaled)

        ridge_prediction = ridge_model.model.predict(X_today_scaled)[0]
        if clf_model is None:
            # Training labels were all one sign, so no classifier could be fitted: it simply agrees.
            clf_is_long = ridge_prediction > 0
        else:
            clf_is_long = clf_model.model.predict(X_today_scaled)[0] == 1

        # Veto only when the two models disagree AND the classifier is confident.
        vetoed = False
        if (ridge_prediction > 0) != clf_is_long:
            vetoed = max(clf_model.model.predict_proba(X_today_scaled)[0]) > VETO_CONFIDENCE

        # Record the forecast (before the veto, so vetoed days do not leave gaps) and standardise.
        history = self.prediction_history.setdefault(f"{commodity}_preds", [])
        history.append(ridge_prediction)
        recent = history[-SIGNAL_HISTORY_LEN:]

        if len(recent) >= MIN_SIGNAL_HISTORY:
            signal = (ridge_prediction - np.mean(recent)) / (np.std(recent) + 1e-8)
        else:
            # Very early on: scale the raw forecast by realised volatility instead.
            vol = prices.pct_change().dropna().iloc[-20:].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            signal = ridge_prediction / (vol + 1e-8)

        if vetoed:
            return 0.0
        return np.clip(signal, -SIGNAL_CAP, SIGNAL_CAP)

    def _train_models(self, features: pd.DataFrame, prices: pd.Series, commodity: str,
                      current_date: pd.Timestamp):
        """
        Fit the regression (and direction classifier) on the most recent training window.

        Returns (regression model, classifier or None, scaler, pca or None, feature names),
        or None if there is not enough data.
        """
        ridge_model = _build_regression_model()
        y = ridge_model.compute_target(prices).values  # forward returns; NaN for the last few rows

        X, feature_names = ridge_model.prepare_features(features, target=pd.Series(y))
        ridge_model.feature_names = feature_names

        n = min(len(X), len(y))
        X, y = X[:n], y[:n]

        # Training window = the rows before today. Rows whose label is still unknown
        # (the last forward-horizon rows) have NaN labels and are dropped below.
        window = min(TRADING_DAYS_PER_YEAR * TRAIN_WINDOW_YEARS, len(X) - 1)
        if len(X) < window + 1:
            return None
        X_train, y_train = X[-window - 1:-1], y[-window - 1:-1]

        known = ~np.isnan(y_train)
        X_train, y_train = X_train[known], y_train[known]
        if len(y_train) < MIN_HISTORY_ROWS:
            return None

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # PCA decorrelates the overlapping 7/14/30-day features. Trees do not benefit
        # (rotating the axes hurts axis-aligned splits), so it is skipped for xgboost.
        pca = None
        if USE_PCA and MODEL_TYPE != 'xgboost':
            if isinstance(PCA_N_COMPONENTS, int):  # fixed count, clamped to what is feasible
                n_components = min(PCA_N_COMPONENTS, X_train_scaled.shape[1], X_train_scaled.shape[0])
            else:                                  # float in (0, 1): share of variance to keep
                n_components = PCA_N_COMPONENTS
            pca = PCA(n_components=n_components)
            X_train_scaled = pca.fit_transform(X_train_scaled)
            logger.debug(f"PCA ({commodity} {current_date.date()}): {pca.n_components_} components, "
                         f"explained={pca.explained_variance_ratio_.sum():.3f}")

        ridge_model.model.fit(X_train_scaled, y_train)

        # Direction classifier on the same window. If every label has the same sign there is
        # nothing to classify, so skip it (the veto then never fires).
        y_up = (y_train > 0).astype(int)
        clf_model = None
        if len(np.unique(y_up)) == 2:
            clf_model = DirectionalClassifier()
            clf_model.model.fit(X_train_scaled, y_up)

        return ridge_model, clf_model, scaler, pca, feature_names


def run_backtest(start_date: str,
                 end_date: str,
                 commodities: List[str],
                 initial_capital: float = 1_000_000) -> pd.DataFrame:
    """Convenience wrapper: build an engine, run it, return the daily results."""
    return BacktestEngine(start_date, end_date, commodities, initial_capital).run()
