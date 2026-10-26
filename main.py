"""
Entry point: run the backtest, print metrics, and write the CSV, chart and tearsheet.

    python main.py

Settings come from environment variables / a ``.env`` file (see ``config/settings.py``).
Outputs go to ``data_storage/results/``.
"""
import sys

from loguru import logger

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics_by_commodity, print_metrics_by_commodity
from backtest.plots import plot_backtest_results
from config.settings import (
    BACKTEST_END_DATE,
    BACKTEST_START_DATE,
    COMMODITY_TRADE_MONTHS,
    EXECUTE_NEXT_BAR,
    FEATURE_LAG_DAYS,
    INITIAL_CAPITAL,
    LOG_LEVEL,
    LOG_PATH,
    MODEL_TYPE,
    PCA_N_COMPONENTS,
    PHASE_1_COMMODITIES,
    RESULTS_PATH,
    USE_PCA,
)


def setup_logging():
    """Log to the terminal at LOG_LEVEL and to ``data_storage/logs/backtest.log`` at DEBUG."""
    logger.remove()
    logger.add(sys.stdout,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
               level=LOG_LEVEL)
    log_file = LOG_PATH / "backtest.log"
    logger.add(log_file,
               format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
               level="DEBUG",
               rotation="10 MB")
    logger.info(f"Logging to {log_file}")


def main():
    setup_logging()

    logger.info("=" * 80)
    logger.info("NASAPOWERCOUNCIL - Weather-Based Commodity Trading System")
    logger.info("=" * 80)
    logger.info(f"Backtest Period: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    logger.info(f"Commodities: {PHASE_1_COMMODITIES}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    logger.info(f"Model: {MODEL_TYPE}  |  PCA: {USE_PCA} (n_components={PCA_N_COMPONENTS})")
    logger.info(f"Weather lag: {FEATURE_LAG_DAYS} day(s)  |  Execute next bar: {EXECUTE_NEXT_BAR}")
    logger.info("=" * 80)

    try:
        results = run_backtest(
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE,
            commodities=PHASE_1_COMMODITIES,
            initial_capital=INITIAL_CAPITAL,
        )
        if results.empty:
            logger.error("Backtest returned no results")
            return

        # Per-commodity metrics, each over that commodity's trade months.
        print_metrics_by_commodity(compute_metrics_by_commodity(results, COMMODITY_TRADE_MONTHS))

        results_file = RESULTS_PATH / 'backtest_results.csv'
        results.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        # Chart of the first commodity's trade months.
        plot_file = RESULTS_PATH / 'backtest_plot.png'
        trade_months = COMMODITY_TRADE_MONTHS[PHASE_1_COMMODITIES[0]] if PHASE_1_COMMODITIES else None
        plot_backtest_results(results, PHASE_1_COMMODITIES, save_path=str(plot_file), trade_months=trade_months)
        logger.info(f"Plot saved to {plot_file}")

        # HTML tearsheet needs the optional quantstats package.
        try:
            from backtest.quantstats_report import generate_quantstats_report
            generate_quantstats_report(results, output_path=str(RESULTS_PATH / 'quantstats_report.html'),
                                       commodities=PHASE_1_COMMODITIES)
        except ImportError:
            logger.warning("quantstats is not installed; skipping the HTML tearsheet (pip install quantstats)")

        logger.info("BACKTEST COMPLETE")

    except KeyboardInterrupt:
        logger.warning("Backtest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Backtest failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
