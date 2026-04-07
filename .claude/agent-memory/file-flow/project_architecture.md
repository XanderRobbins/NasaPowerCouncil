---
name: NasaPowerCouncil Architecture
description: End-to-end data flow, module dependency map, and key design decisions for the weather-based commodity trading backtest system
type: project
---

## System Summary
Weather-based commodity futures trading strategy. Uses NASA POWER satellite climate data to generate stress signals for agricultural commodities, runs Ridge + Logistic Regression models in a walk-forward backtest, and produces metrics/visualizations.

## Entry Point
`main.py` — calls `run_backtest()`, then compact → metrics → plot → QuantStats report.

## Data Flow (end-to-end)
1. `NASAPowerFetcher.fetch_commodity_regions()` — pulls 7 daily weather vars per lat/lon from NASA POWER API, caches as .pkl in `data_storage/raw/climate/`
2. `MarketDataFetcher.fetch_futures_data()` — pulls OHLCV from Yahoo Finance (continuous futures tickers), caches via `CacheManager` in `data_storage/raw/market/`
3. `FeaturePipeline.run()` — per region: seasonal Z-scores (vectorized expanding groupby, no look-ahead) → stress indicators (binary + Z + 7/14/30d rolling) → production-weighted regional aggregation
4. `BacktestEngine._walk_forward_simulation()` — per calendar day: growing season filter → volatility regime filter → monthly model retrain (Ridge + LogReg trained on same scaled window) → classifier confirmation filter → Z-score signal → position sizing → mark-to-market P&L
5. `compact_results()` — strips flat/idle calendar days, re-anchors equity curve on active days only
6. `compute_metrics()` / `compute_commodity_attribution()` — Sharpe, Sortino, Calmar, win rate, per-commodity attribution
7. `BacktestVisualizer.create_full_report()` — 6-panel dark-theme chart saved to `data_storage/results/`

## Module Dependency Map
```
main.py
├── backtest/engine.py          [central hub — imports all layers]
│   ├── data/climate_fetcher.py
│   │   └── config/regions.py
│   ├── data/market_fetcher.py
│   │   └── data/cache_manager.py
│   ├── features/feature_pipeline.py
│   │   ├── features/seasonal_deviation.py
│   │   ├── features/stress_calculator.py  → config/thresholds.py
│   │   └── features/regional_aggregator.py → config/regions.py
│   ├── models/ridge_model.py
│   ├── models/classifier_model.py
│   └── config/settings.py
├── backtest/compactor.py
├── backtest/metrics.py
├── backtest/visualizer.py
└── config/settings.py
```

## Key Design Decisions
- **No look-ahead bias**: seasonal baseline uses `shift(1).expanding()` per calendar-day group; walk-forward simulation slices all data to `<= current_date` before training
- **Monthly model cache**: `model_cache[f"{commodity}_{year}_{month}"]` avoids retraining every day — retrained once per month
- **Dual-model signal**: Ridge predicts magnitude, LogReg predicts direction. If they disagree AND classifier confidence > 0.65, signal = 0. Otherwise Ridge wins.
- **Z-scored signal**: Ridge prediction normalized against rolling 60-month prediction history → clipped to ±3.0
- **Two position sizing modes** (env-controlled): `directional` (fixed 10% if |signal| >= threshold) vs `magnitude` (vol-scaled)
- **Per-commodity season windows**: each commodity has its own `TRADE_MONTHS` from env; positions closed explicitly when season ends
- **Volatility regime filter**: per-commodity threshold (env `{COMMODITY}_VOL_REGIME_THRESHOLD`); skips signal if 20-day realized vol exceeds it
- **Compacted vs full results**: engine outputs full calendar results; `compact_results()` strips idle days and re-chains equity curve for clean metrics/plots

## Config Hierarchy
All settings: `config/settings.py` — reads `.env` first, falls back to hardcoded defaults.
- `config/regions.py` — lat/lon + production weights per commodity region
- `config/thresholds.py` — per-commodity crop stress temperature/precip thresholds

## Cache Architecture
Two-layer cache:
- `NASAPowerFetcher`: simple filename cache `{commodity}_{region}_{start}_{end}.pkl` in `data_storage/raw/climate/`
- `CacheManager` (singleton `cache_manager`): MD5-hash-keyed TTL cache used by `MarketDataFetcher` in `data_storage/raw/market/`

## Feature Column Naming Convention
Raw weather → Z-scored (`temp_max_z`) → stress binary (`heat_stress`) → stress Z (`heat_stress_z`) → rolling windows (`heat_stress_7d`, `_14d`, `_30d`) → production-weighted aggregate (`heat_stress_7d_agg`). Models use only `*_agg` columns with rolling suffixes as features.

## Outputs
- `data_storage/results/backtest_results_full.csv`
- `data_storage/results/backtest_results_compacted.csv`
- `data_storage/results/backtest_plot.png`
- `data_storage/results/quantstats_report.html`
- `data_storage/logs/backtest.log`
