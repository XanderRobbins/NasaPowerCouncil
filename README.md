# NasaPowerCouncil

> *A quantitative trading system for agricultural commodity futures, driven by satellite-derived climate intelligence and machine learning.*

---

## Overview

**NasaPowerCouncil** is a systematic trading strategy that leverages NASA POWER climate data to forecast price movements in agricultural commodity futures. The system employs Ridge regression with rolling training windows to uncover relationships between weather stress indicators and forward returns — enabling disciplined, data-driven position taking in corn and soybean markets.

The core thesis is elegant in its simplicity: extreme weather events — droughts, heat waves, cold snaps — materially impact crop yields, and these impacts are often underpriced in futures markets during the growing season. By quantifying weather stress across key production regions and modeling its historical relationship to forward returns, the system attempts to capture this edge systematically and repeatably.

---

## Motivation

Traditional commodity trading relies heavily on fundamental analysis — crop reports, export data, inventory levels — which are inherently backward-looking and subject to revision. Weather derivatives and insurance products exist but are often illiquid and expensive. NasaPowerCouncil offers a third path: using freely available, high-quality satellite data to generate forward-looking signals in liquid futures markets.

NASA's **POWER** (Prediction Of Worldwide Energy Resources) project provides validated, gap-filled climate data at daily resolution for any point on Earth — including temperature extremes, precipitation, solar radiation, and humidity, all critical inputs for crop stress modeling. By aggregating this data across production-weighted regions and training statistical models on historical price behavior, the system systematically identifies periods when weather stress creates tradeable dislocations.

---

## Key Features

### Data Infrastructure

| Module | Description |
|---|---|
| `climate_fetcher.py` | Automated fetching of daily NASA POWER climate data for major production regions |
| `market_fetcher.py` | Continuous front-month futures pricing via Yahoo Finance — no API costs |
| `cot_fetcher.py` | CFTC Commitments of Traders data for positioning and sentiment context |
| `cache_manager.py` | Intelligent local persistence layer to minimize redundant API calls |

### Feature Engineering

- **Seasonal Deviation Analysis** — Z-score normalization against historical baselines
- **Weather Stress Indicators** — Commodity-specific heat, cold, and drought thresholds (`thresholds.py`)
- **Multi-Timeframe Aggregation** — Rolling 7-day, 14-day, and 30-day cumulative stress windows
- **Production Weighting** — Regional aggregation using actual production volumes (`regions.py`)
- **Growing Season Calendars** — Stage-specific sensitivity weights for pollination, grain fill, and other critical phenological periods (`calendars.py`)

### Modeling Framework

- **Rolling Ridge Regression** — 10-year training windows with L2 regularization to control overfitting (`ridge_model.py`)
- **Time Series Cross-Validation** — Proper temporal splits that strictly prevent look-ahead bias
- **Dynamic Beta Estimation** — Kalman filtering allows weather sensitivity to evolve with market regimes (`kalman_beta.py`)
- **Feature Selection** — Variance-based filtering limits each model to its most informative signals

### Backtesting Engine

- **Walk-Forward Simulation** — Strict point-in-time data access with monthly model retraining
- **Transaction Cost Modeling** — Configurable slippage and commission assumptions
- **Risk Management** — Position sizing via volatility normalization and drawdown limits
- **Performance Attribution** — Sharpe, Sortino, Calmar, hit rate, and profit factor reporting

---

## Installation

### Prerequisites

- Python 3.9 or higher
- A virtual environment (strongly recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/NasaPowerCouncil.git
cd NasaPowerCouncil

# Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` to configure strategy parameters:

```bash
# Strategy Settings
STRATEGY_MODE=directional          # 'directional' or 'magnitude'
MIN_SIGNAL_STRENGTH=0.3            # Minimum signal threshold
FIXED_POSITION_SIZE=0.05           # Position size as fraction of portfolio
ONLY_TRADE_GROWING_SEASON=true    # Restrict trading to April–September

# Model Parameters
TRAIN_WINDOW_YEARS=5               # Rolling training window
FORWARD_RETURN_DAYS=10             # Prediction horizon
MAX_FEATURES=15                    # Feature selection limit

# Risk Management
TARGET_PORTFOLIO_VOL=0.12          # Target annualized volatility
MAX_SINGLE_POSITION=0.20           # Maximum position size
```

---

## Usage

### Running a Backtest

```python
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, print_metrics

results = run_backtest(
    start_date='2015-01-01',
    end_date='2025-01-01',
    commodities=['corn', 'soybeans'],
    initial_capital=1_000_000
)

metrics = compute_metrics(results)
print_metrics(metrics)

results.to_csv('backtest_results.csv', index=False)
```

### Training Models Independently

```python
from models.ridge_model import train_ridge_model
from features.feature_pipeline import run_feature_pipeline
from data.climate_fetcher import fetch_commodity_regions
from data.market_fetcher import fetch_market_data
from sklearn.metrics import r2_score

climate_data = fetch_commodity_regions('corn', '2010-01-01', '2025-01-01')
market_data  = fetch_market_data('corn', 'ZC=F', '2010-01-01', '2025-01-01')

features = run_feature_pipeline('corn', climate_data)

model, predictions, actuals, dates = train_ridge_model(
    features,
    market_data['close'],
    'corn'
)

print(f"Model R²: {r2_score(actuals, predictions):.4f}")
```

### Command Line

```bash
python main.py
```

Results are saved to `data_storage/results/`, including CSV output and performance visualizations.

---

## Methodology

### Weather Stress Quantification

The system quantifies three distinct forms of crop stress:

**Heat Stress** — Days where maximum temperature exceeds critical thresholds (32°C for corn, 35°C for soybeans). Cumulative heat stress is particularly destructive during pollination, when pollen viability collapses above these levels.

**Cold Stress** — Days where minimum temperature falls below emergence thresholds (10°C for corn, 15°C for soybeans). Early-season cold can delay stand establishment and reduce final yield potential.

**Drought Stress** — Weekly cumulative precipitation below critical thresholds (25mm for corn, 30mm for soybeans). Water deficit during vegetative growth and grain fill directly diminishes yield.

These binary indicators are aggregated into rolling cumulative measures across 7-day, 14-day, and 30-day windows, then production-weighted across major growing regions (`regions.py`).

### Prediction Model

A Ridge regression model predicts forward returns as a linear function of current weather stress:

```
Y_t = β₀ + β₁·HeatStress₇d + β₂·ColdStress₁₄d + ... + ε_t
```

Where `Y_t` is the N-day forward return, features are production-weighted stress indicators, and coefficients are estimated via L2-regularized regression on a rolling 5–10 year training window. Models are retrained monthly to adapt to gradual regime shifts.

The model is intentionally parsimonious. Complex architectures — neural networks, gradient boosting — tend to overfit severely given the limited sample size of 15–20 growing seasons. Ridge regression offers a favorable bias-variance tradeoff and produces interpretable coefficients that can be examined and challenged.

### Walk-Forward Backtesting

The backtesting engine enforces strict temporal integrity at every step:

1. At each date `t`, use only data available through `t-1`
2. Train the model on the most recent N years of history
3. Generate a prediction for date `t`
4. Size positions based on signal strength and realized volatility
5. Mark-to-market all positions at date `t` prices
6. Update portfolio value and advance to `t+1`

This walk-forward discipline eliminates look-ahead bias — a common and often invisible failure mode in commodity strategy research.

---

## Performance Characteristics

> Based on testing from 2015–2025 with realistic assumptions (10 bps/trade transaction costs, 5% position sizing, monthly retraining, growing season only).

| Metric | Range |
|---|---|
| Annualized Return | 15–20% (after costs) |
| Sharpe Ratio | 1.2–1.8 |
| Maximum Drawdown | -30% to -40% |
| Win Rate | 54–58% |
| Profit Factor | 1.4–1.8 |

The strategy exhibits elevated drawdowns characteristic of weather-driven dislocation strategies. Severe weather events — droughts especially — tend to persist, creating extended adversarial periods before mean reversion reasserts itself. This profile makes the system unsuitable for low-volatility mandates but potentially compelling for absolute return allocators with appropriate risk tolerance.

---

## Project Structure

```
NasaPowerCouncil/
├── config/                  # Configuration and settings
│   ├── settings.py          # Global parameters
│   ├── regions.py           # Production region definitions
│   ├── thresholds.py        # Weather stress thresholds
│   └── calendars.py         # Growing season calendars
├── data/                    # Data acquisition and caching
│   ├── climate_fetcher.py   # NASA POWER integration
│   ├── market_fetcher.py    # Yahoo Finance integration
│   ├── cot_fetcher.py       # CFTC positioning data
│   └── cache_manager.py     # Local caching layer
├── features/                # Feature engineering pipeline
│   ├── feature_pipeline.py      # Pipeline orchestration
│   ├── seasonal_deviation.py    # Z-score normalization
│   ├── stress_calculator.py     # Weather stress indicators
│   └── regional_aggregator.py  # Production weighting
├── models/                  # Prediction models
│   ├── ridge_model.py       # Rolling Ridge regression
│   └── kalman_beta.py       # Dynamic beta estimation
├── backtest/                # Backtesting framework
│   ├── engine.py            # Walk-forward simulation
│   ├── metrics.py           # Performance calculation
│   └── visualizer.py        # Results plotting
├── signals/                 # Signal generation
│   └── signal_generator.py  # Volatility normalization
├── utils/                   # Utilities
│   └── data_validator.py    # Data quality checks
└── main.py                  # Entry point
```

---

## Data Sources

**NASA POWER** — The POWER project provides satellite-derived climate data at 0.5° × 0.5° global resolution, available from 1981 to near real-time (typically 1–2 day lag). Funded by NASA's Earth Science program and maintained at Langley Research Center. API access is free but rate-limited (~60 requests/minute); the caching layer respects these limits intelligently.

**Yahoo Finance** — Commodity futures data via the `yfinance` Python library. Continuous front-month contracts (e.g., `ZC=F` for corn) are used throughout, avoiding the complexity of manual roll adjustments.

**CFTC Commitments of Traders** — Weekly positioning data providing sentiment context for the modeling and signal generation layers.

---

## Limitations & Risks

**Model Risk** — The strategy assumes weather-price relationships are stationary. Climate change, advances in agricultural technology, and evolving trade patterns may gradually erode these relationships. Monthly retraining helps, but structural breaks can cause sustained underperformance before they are detected.

**Execution Risk** — Backtests assume frictionless execution at closing prices. Real-world slippage can be material, particularly during the volatile conditions when weather-driven signals tend to be strongest. The 10 bps cost assumption may prove optimistic at scale.

**Data Risk** — NASA POWER data occasionally contains gaps or anomalies, particularly for recent dates. Validation checks are in place but cannot eliminate all data quality issues.

**Regime Risk** — When markets are dominated by factors orthogonal to weather — currency moves, policy shocks, demand disruptions — weather signals may fail systematically. The strategy has no explicit regime filter and will continue trading through unfavorable environments.

---

## Roadmap

| Phase | Focus |
|---|---|
| **Phase 2** | Extend to coffee, natural gas, wheat, and cotton with commodity-specific calendars and thresholds |
| **Phase 3** | Explore gradient boosting and ensemble methods with modern regularization techniques |
| **Phase 4** | Live trading via Interactive Brokers API with real-time data pipelines and position monitoring |
| **Phase 5** | Options strategies — trade implied volatility using weather forecast regimes |

---

## Contributing

Contributions are welcome. Please follow these conventions:

1. Fork the repository and create a feature branch (`git checkout -b feature/your-feature`)
2. Write tests for new functionality
3. Ensure all tests pass (`pytest`)
4. Submit a pull request with a clear description of motivation and approach

---

## License

Distributed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

This project draws on data from the **NASA POWER** project (Langley Research Center), the **CFTC** for Commitments of Traders reports, and **Yahoo Finance** for market data. The methodology is informed by academic research in weather derivatives and agricultural commodity pricing.

---

> **Disclaimer:** This system is provided for research and educational purposes only. Past performance does not guarantee future results. Trading commodity futures involves substantial risk of loss. Consult a qualified financial advisor before deploying capital.