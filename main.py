"""
Main entry point for NasaPowerCouncil backtest.
"""
from loguru import logger
import sys
from pathlib import Path
from backtest.compactor import compact_results
from backtest.engine import run_backtest
<<<<<<< HEAD
from backtest.metrics import compute_metrics_by_commodity, print_metrics_by_commodity
=======
from backtest.metrics import compute_metrics, print_metrics, compute_commodity_attribution, print_attribution
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
from backtest.visualizer import plot_backtest_results
from backtest.quantstats_report import generate_quantstats_report
from config.settings import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
    PHASE_1_COMMODITIES,
    INITIAL_CAPITAL,
    LOG_PATH,
    RESULTS_PATH,
    COMMODITY_TRADE_MONTHS
)

def setup_logging():
    """Configure logging."""
    logger.remove()
    logger.add(sys.stdout,
              format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
              level="INFO")
    log_file = LOG_PATH / "backtest.log"
    logger.add(log_file,
              format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
              level="DEBUG",
              rotation="10 MB")
    logger.info(f"Logging to {log_file}")


def main():
    """Run full backtest pipeline."""
    setup_logging()

    logger.info("=" * 80)
    logger.info("NASAPOWERCOUNCIL - Weather-Based Commodity Trading System")
    logger.info("=" * 80)
    logger.info(f"Backtest Period: {BACKTEST_START_DATE} to {BACKTEST_END_DATE}")
    logger.info(f"Commodities: {PHASE_1_COMMODITIES}")
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")
    logger.info("=" * 80)

    try:
        # Run backtest
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
<<<<<<< HEAD
        
        # Compute per-commodity metrics
        logger.info("\n📊 Computing performance metrics by commodity...")
        metrics_by_commodity = compute_metrics_by_commodity(results, COMMODITY_TRADE_MONTHS)
        print_metrics_by_commodity(metrics_by_commodity)

        # Diagnostic: check drawdown data
        logger.info("\n🔍 Drawdown diagnostic:")
        active_mask = results['portfolio_return'] != 0
        active_values = results.loc[active_mask, 'portfolio_value'].values

        if len(active_values) > 0:
            logger.info(f"  Portfolio values (active days):")
            logger.info(f"    Min: ${active_values.min():,.0f}")
            logger.info(f"    Max: ${active_values.max():,.0f}")
            logger.info(f"    Initial: ${active_values[0]:,.0f}")
            logger.info(f"  Biggest single-day loss:")
            biggest_loss_idx = results['daily_pnl'].idxmin()
            if biggest_loss_idx > 0:
                window = results.loc[biggest_loss_idx - 1:biggest_loss_idx + 1,
                                      ['date', 'portfolio_value', 'daily_pnl', 'portfolio_return']]
                for _, row in window.iterrows():
                    logger.info(f"    {row['date'].date()} | PV: ${row['portfolio_value']:,.0f} | PnL: ${row['daily_pnl']:,.0f} | Return: {row['portfolio_return']:.2%}")
        
        # Save results
        results_file = RESULTS_PATH / 'backtest_results.csv'
        results.to_csv(results_file, index=False)
        logger.info(f"✓ Results saved to {results_file}")
        
        # Create visualizations (filtered to trade months)
        logger.info("\n📈 Creating visualizations...")
        plot_file = RESULTS_PATH / 'backtest_plot.png'
        trade_months = COMMODITY_TRADE_MONTHS[PHASE_1_COMMODITIES[0]] if PHASE_1_COMMODITIES else None
        # Get first commodity metrics for display
        display_metrics = metrics_by_commodity.get(PHASE_1_COMMODITIES[0], {}) if PHASE_1_COMMODITIES else {}
        plot_backtest_results(results, PHASE_1_COMMODITIES, save_path=str(plot_file),
                            trade_months=trade_months, metrics=display_metrics)
=======

        # Compact to active trading days only
        logger.info("\n🗜️  Compacting results to active trading days...")
        compacted = compact_results(results, PHASE_1_COMMODITIES)
        logger.info(f"  Full results: {len(results)} calendar days → {len(compacted)} active trading days")

        # Metrics on compacted frame
        logger.info("\n📊 Computing performance metrics...")
        metrics = compute_metrics(compacted)
        print_metrics(metrics)
        attribution = compute_commodity_attribution(compacted, PHASE_1_COMMODITIES)
        print_attribution(attribution)

        # Save both full and compacted
        results.to_csv(RESULTS_PATH / 'backtest_results_full.csv', index=False)
        compacted.to_csv(RESULTS_PATH / 'backtest_results_compacted.csv', index=False)
        logger.info(f"✓ Results saved to {RESULTS_PATH}")

        # Visualize compacted only
        logger.info("\n📈 Creating visualizations...")
        plot_file = RESULTS_PATH / 'backtest_plot.png'
        plot_backtest_results(compacted, PHASE_1_COMMODITIES, save_path=str(plot_file))
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
        logger.info(f"✓ Plot saved to {plot_file}")

        # QuantStats tearsheet — uses full results (real calendar dates)
        logger.info("\n📊 Generating QuantStats tearsheet...")
        qs_report_file = RESULTS_PATH / 'quantstats_report.html'
        generate_quantstats_report(
            results=results,
            output_path=qs_report_file,
            benchmark_ticker='DJP',
            commodities=PHASE_1_COMMODITIES
        )
        logger.info(f"✓ QuantStats report saved to {qs_report_file}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ BACKTEST COMPLETE")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Backtest interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.exception(f"❌ Backtest failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()