"""
Main entry point for NasaPowerCouncil backtest.
"""
from loguru import logger
import sys
import pandas as pd
from pathlib import Path
from config.settings import RISK_FREE_RATE  # add this import at top of function or file

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, print_metrics
from backtest.visualizer import plot_backtest_results
from config.settings import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    PHASE_1_COMMODITIES,
    INITIAL_CAPITAL,
    LOG_PATH,
    RESULTS_PATH
)


def setup_logging():
    """Configure logging."""
    logger.remove()

    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )

    log_file = LOG_PATH / "backtest.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB"
    )
    logger.info(f"Logging to {log_file}")


def generate_quantstats_report(results: pd.DataFrame):
    """
    Generate quantstats HTML report with SPY benchmark.
    Saves to RESULTS_PATH/quantstats_report.html
    quantstats is imported here (not at module level) so a missing
    install never prevents the rest of the pipeline from running.
    """
    try:
        import quantstats as qs
    except ImportError:
        logger.warning("quantstats not installed — skipping HTML report. Run: pip install quantstats")
        return

    try:
        logger.info("\n📊 Generating quantstats HTML report...")

        returns_series = (
            results
            .set_index('date')['portfolio_return']
            .dropna()
        )
        returns_series.index = pd.to_datetime(returns_series.index)


        report_path = str(RESULTS_PATH / 'quantstats_report.html')
        qs.reports.html(
            returns_series,
            benchmark='SPY',
            output=report_path,
            rf=RISK_FREE_RATE, 
            title='NasaPowerCouncil — Weather-Based Commodity Strategy'
        )
        logger.info(f"✓ Quantstats report saved to {report_path}")

    except Exception as e:
        logger.error(f"Quantstats report generation failed: {e}")
        logger.warning("Continuing without quantstats report")


def main():
    """Run full backtest pipeline."""

    setup_logging()

    logger.info("=" * 80)
    logger.info("NASAPOWERCOUNCIL - Weather-Based Commodity Trading System")
    logger.info("=" * 80)
    logger.info(f"Backtest Period: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    logger.info(f"Commodities:     {PHASE_1_COMMODITIES}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    logger.info("=" * 80)

    try:
        # --- Step 1: Run backtest ---
        logger.info("\n🚀 Starting backtest...")
        results = run_backtest(
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE,
            commodities=PHASE_1_COMMODITIES,
            initial_capital=INITIAL_CAPITAL
        )

        if results.empty:
            logger.error("❌ Backtest returned no results")
            return

        logger.info("✓ Backtest complete")

        # --- Step 2: Compute and print metrics ---
        logger.info("\n📊 Computing performance metrics...")
        metrics = compute_metrics(results)

        if not metrics:
            logger.error("❌ Metrics computation returned empty — check logs above")
            return

        print_metrics(metrics)

        # --- Diagnostic: flag extreme return days ---
        extreme_days = results[results['portfolio_return'].abs() > 0.05]
        logger.info(f"Days with >5% single-day return: {len(extreme_days)}")
        if not extreme_days.empty:
            logger.info(f"\n{extreme_days[['date', 'portfolio_return', 'daily_pnl']].to_string()}")

        # --- Step 3: Save raw results CSV ---
        results_file = RESULTS_PATH / 'backtest_results.csv'
        results.to_csv(results_file, index=False)
        logger.info(f"✓ Results saved to {results_file}")

        # --- Step 4: Quantstats HTML report ---
        generate_quantstats_report(results)

        # --- Step 5: Custom visualizations ---
        logger.info("\n📈 Creating visualizations...")
        plot_file = RESULTS_PATH / 'backtest_plot.png'
        plot_backtest_results(results, PHASE_1_COMMODITIES, save_path=str(plot_file))
        logger.info(f"✓ Plot saved to {plot_file}")

        # --- Completion summary ---
        logger.info("\n" + "=" * 80)
        logger.info("✅ BACKTEST COMPLETE")
        logger.info(f"   Results CSV:       {results_file}")
        logger.info(f"   Plot:              {plot_file}")
        logger.info(f"   Quantstats Report: {RESULTS_PATH / 'quantstats_report.html'}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Backtest interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception(f"❌ Backtest failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()