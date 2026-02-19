"""
Script to run a full backtest.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.engine import run_backtest
from backtest.metrics import PerformanceMetrics
from backtest.visualizer import visualize_backtest
from config.settings import BACKTEST_START_DATE, BACKTEST_END_DATE, PHASE_1_COMMODITIES
from loguru import logger


def main():
    """Run backtest and generate report."""
    
    # Configuration
    commodities = PHASE_1_COMMODITIES
    start_date = BACKTEST_START_DATE
    end_date = BACKTEST_END_DATE
    
    logger.info("Starting backtest...")
    logger.info(f"Commodities: {commodities}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Run backtest
    results = run_backtest(start_date, end_date, commodities)
    
    # Compute metrics
    logger.info("\nComputing performance metrics...")
    if results.empty or 'portfolio_return' not in results.columns:
        logger.error("Backtest returned no results or incomplete data")
        logger.info(f"Results shape: {results.shape}")
        logger.info(f"Results columns: {list(results.columns)}")
        
        # Save what we have
        results_file = project_root / 'data_storage' / 'results' / 'backtest_results.pkl'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_pickle(results_file)
        logger.info(f"Saved incomplete results to {results_file}")
        
        import sys
        sys.exit(1)

    metrics_calc = PerformanceMetrics(results)
    metrics_calc.print_summary()
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    visualize_backtest(results, commodities)
    
    # Save results
    results_file = project_root / 'data_storage' / 'results' / 'backtest_results.parquet'
    results.to_parquet(results_file)
    logger.info(f"\n✓ Results saved to {results_file}")


if __name__ == '__main__':
    main()