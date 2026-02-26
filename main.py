"""
Main entry point for NasaPowerCouncil backtest.
"""
from loguru import logger
import sys
from pathlib import Path

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
    logger.remove()  # Remove default handler
    
    # Console handler (INFO level)
    logger.add(sys.stdout, 
              format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
              level="INFO")
    
    # File handler (DEBUG level)
    log_file = LOG_PATH / "backtest.log"
    logger.add(log_file,
              format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
              level="DEBUG",
              rotation="10 MB")
    
    logger.info(f"Logging to {log_file}")

def main():
    """Run full backtest pipeline."""
    
    # Setup
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
        
        # Compute metrics
        logger.info("\n📊 Computing performance metrics...")
        metrics = compute_metrics(results)
        print_metrics(metrics)
        
        # Save results
        results_file = RESULTS_PATH / 'backtest_results.csv'
        results.to_csv(results_file, index=False)
        logger.info(f"✓ Results saved to {results_file}")
        
        # Create visualizations
        logger.info("\n📈 Creating visualizations...")
        plot_file = RESULTS_PATH / 'backtest_plot.png'
        plot_backtest_results(results, PHASE_1_COMMODITIES, save_path=str(plot_file))
        logger.info(f"✓ Plot saved to {plot_file}")
        
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