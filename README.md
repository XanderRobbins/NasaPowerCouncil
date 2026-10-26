# NasaPowerCouncil

A research backtest for trading agricultural futures on weather. It turns NASA POWER satellite
weather into crop-stress features, forecasts the next-N-day return with a Ridge regression, and
simulates trading it walk-forward. Only **corn** is wired up for trading.

> **Status: research prototype, not a proven strategy.** See [Results](#results-what-to-expect)
> for what the tests actually showed.

## The idea

Heat, cold and dry spells during the growing season hurt crop yields, and the thesis is that the
market under-prices that early. For each production region the code measures how unusual the
weather is for the date, counts stress days, accumulates them over 7/14/30 days, and averages the
regions by production weight. A model learns, on a rolling window, how those stress features
relate to the following 10-day corn return.

## How it works

```
NASA POWER weather ─► seasonal z-scores ─► stress flags + 7/14/30-day sums ─► production-weighted
(per region)           "unusual for this    heat / cold / dry, per region      average over regions
                        calendar day?"                                                │
Yahoo futures prices ───────────────────────────────────────────────────────────────┤
                                                                                      ▼
   each in-season day: (monthly refit) top-k features → scale → PCA → Ridge  (+ logistic classifier)
        forecast → z-score vs last 60 forecasts → classifier veto → cap ±3 → volatility filter
                  → target position → (trade at next close) → costs, stop-loss, mark-to-market
```

Key rules:

* **No look-ahead.** Seasonal baselines use only earlier years; a model trained on day *t* only
  uses rows whose N-day outcome is already known; weather features are delayed by
  `FEATURE_LAG_DAYS` (NASA POWER publishes 1–2+ days late) and decisions are traded at the next
  close (`EXECUTE_NEXT_BAR`).
* **Seasonal.** Corn is traded April–June (`CORN_TRADE_MONTHS`) and the account is in cash otherwise.
* **Monthly refits** on a rolling 3-year window.
* **Risk controls.** 5% stop-loss per position, a volatility filter (flat when 20-day vol > 30%),
  10 bps cost per unit of position change.

## Results: what to expect

Corn, 10 bps costs, 0.5× position, April–June, with the 2-day weather lag and next-bar execution.
"In-season Sharpe" is mean/std of daily returns on April–June days × √252.

| Period | Price series | In-season Sharpe | Return per season |
|---|---|---|---|
| 2013–2022 | `ZC=F` (raw) | ≈ 1.4 | ≈ 3.6% |
| 2013–2022 | `ZC=F` (roll-neutralised) | ≈ 1.1–1.2 | ≈ 2.9% |
| 2013–2022 | Teucrium CORN ETF (no roll gaps) | ≈ 1.4 | ≈ 3.3% |
| 2003–2022 | `ZC=F` (raw) | ≈ 0.9 | ≈ 2.3% |

Read these carefully:

* **The uncertainty is large.** With 10–19 seasons, 95% intervals on the Sharpe run from about
  zero to about 2.5.
* **The 2013–2022 window is partly in-sample**, and ~a dozen variants were tried on it.
  The 19-season figure is the more honest one.
* **Results move a lot with small changes** (timing assumptions, roll handling, which bug fixes
  are on): total return swung by 10–20 points. Do not trust a single number.
* **The edge looks like timing, not just being long corn**: buy-and-hold over the same days scored
  about 0.5–0.6 (futures) and about −0.1 (ETF), and the strategy is roughly half long, half short.
* The Sharpe printed by `main.py` (in-season CAGR ÷ volatility of active days) is a different,
  slightly lower, formula; see `backtest/metrics.py`.
* Earlier versions of this project reported much better numbers. Those came from same-day
  execution, bugs that flattered results (label-misaligned feature selection, an inverted short
  stop-loss, a future-data leak in the z-scores) and an optimistic short window; they are fixed.

## Limitations

* **Roll gaps.** Yahoo's `ZC=F` is not back-adjusted. On contract-roll days the price jumps by the
  calendar spread, and a backtest holding through it books that as profit. Treat raw-futures results
  as optimistic; prefer a back-adjusted series.
* **Costs.** 10 bps per unit of position change is probably light for real corn execution; shorting an
  ETF would add borrow costs.
* **Sample size.** Weather-to-price relationships are estimated from ~10–20 seasons.
* **Stationarity.** Climate, technology and trade patterns change; monthly refits only partly help.
* **Single market.** Only corn is configured for trading.

## Quick start

```bash
pip install -r requirements.txt     # add quantstats for the HTML tearsheet, xgboost for MODEL_TYPE=xgboost
cp .env.example .env                # optional: every setting has a default
python main.py                      # fetch -> features -> backtest -> metrics, chart, CSV, tearsheet
```

Outputs go to `data_storage/results/` (`backtest_results.csv`, `backtest_plot.png`, and
`quantstats_report.html` if quantstats is installed). Downloads are cached in `data_storage/raw/`.

Settings are environment variables (or a `.env` file); `.env.example` lists all of them with their
defaults. `python sweep.py` re-runs the backtest while varying one setting at a time, to show how
sensitive the result is (not to pick parameters: it is in-sample).

To reproduce the older, optimistic behaviour for comparison:
`FEATURE_LAG_DAYS=0 EXECUTE_NEXT_BAR=false python main.py`.

## Project structure

```
main.py                        entry point: run, print metrics, write outputs
sweep.py                       one-at-a-time parameter sensitivity (in-sample!)
config/
  settings.py                  all settings and defaults (env-overridable)
  regions.py                   production regions: lat/lon and weights per commodity
  thresholds.py                heat / cold / dry thresholds per commodity
data/
  climate_fetcher.py           NASA POWER client, cached; rejects bad rainfall units
  market_fetcher.py            Yahoo Finance futures prices, cached for a day
  cache_manager.py             pickle cache with a time-to-live
features/
  seasonal_deviation.py        z-score of the weather vs the same date in earlier years
  stress_calculator.py         heat/cold/dry flags and 7/14/30-day sums
  regional_aggregator.py       production-weighted average over regions
  feature_pipeline.py          chains the three steps
models/
  common.py                    forward-return target and feature selection (shared)
  ridge_model.py               Ridge wrapper (default)
  xgboost_model.py             optional XGBoost wrapper
  classifier_model.py          logistic regression used as a veto
backtest/
  engine.py                    the walk-forward simulation
  metrics.py                   in-season performance metrics
  plots.py                     six-panel chart
  quantstats_report.py         optional HTML tearsheet
```

## Data sources

* **NASA POWER** daily point weather (free, rate limited; published with a 1–2+ day delay).
* **Yahoo Finance** continuous front-month futures via `yfinance` (unadjusted for rolls, see above).

## Disclaimer

For research and education only. Past performance, and especially backtested performance, does not
guarantee future results. Trading futures involves substantial risk of loss.
