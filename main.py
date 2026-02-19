"""
Main entry point for Climate Futures Trading System.
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config.settings import LOG_FILE, LOG_LEVEL

# Configure logging
logger.add(LOG_FILE, rotation="1 day", retention="30 days", level=LOG_LEVEL)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Climate Futures Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest
  python main.py backtest --start 2020-01-01 --end 2023-12-31

  # Deploy live (paper trading)
  python main.py live --paper --run-time 16:30

  # Deploy live (real trading)
  python main.py live --run-time 16:30

  # Fetch data only
  python main.py fetch-data --commodity corn --start 2020-01-01 --end 2023-12-31
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    backtest_parser.add_argument('--commodities', nargs='+', help='Commodities to trade')
    
    # Live command
    live_parser = subparsers.add_parser('live', help='Deploy live trading')
    live_parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    live_parser.add_argument('--run-time', default='16:30', help='Daily run time (HH:MM)')
    live_parser.add_argument('--run-now', action='store_true', help='Run immediately')
    
    # Data fetch command
    fetch_parser = subparsers.add_parser('fetch-data', help='Fetch climate data')
    fetch_parser.add_argument('--commodity', required=True, help='Commodity name')
    fetch_parser.add_argument('--start', required=True, help='Start date')
    fetch_parser.add_argument('--end', required=True, help='End date')
    
    args = parser.parse_args()
    
    if args.command == 'backtest':
        from scripts.run_backtest import main as run_backtest_main
        run_backtest_main()
    
    elif args.command == 'live':
        from scripts.deploy_live import main as deploy_live_main
        deploy_live_main()
    
    elif args.command == 'fetch-data':
        from data.climate_fetcher import NASAPowerFetcher
        fetcher = NASAPowerFetcher()
        data = fetcher.fetch_commodity_regions(args.commodity, args.start, args.end)
        logger.info(f"Fetched data for {args.commodity}: {len(data)} regions")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()