"""
Script to fetch and cache historical data for all commodities.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.climate_fetcher import NASAPowerFetcher
from data.market_fetcher import MarketDataFetcher
from data.cot_fetcher import COTFetcher
from data.data_validator import DataValidator
from config.settings import PHASE_1_COMMODITIES, RAW_DATA_PATH
from loguru import logger
import argparse


def fetch_climate_data(commodities: list, start_date: str, end_date: str):
    """Fetch climate data for all commodities."""
    logger.info("=" * 80)
    logger.info("FETCHING CLIMATE DATA")
    logger.info("=" * 80)
    
    fetcher = NASAPowerFetcher()
    validator = DataValidator()
    
    for commodity in commodities:
        logger.info(f"\nFetching climate data for {commodity}...")
        
        try:
            region_data = fetcher.fetch_commodity_regions(
                commodity,
                start_date,
                end_date
            )
            
            # Validate each region
            for region, df in region_data.items():
                is_valid, issues = validator.validate_climate_data(df, region)
                
                if not is_valid:
                    logger.warning(f"Validation issues for {commodity} - {region}:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
                else:
                    logger.info(f"✓ {commodity} - {region}: {len(df)} records, valid")
            
        except Exception as e:
            logger.error(f"Failed to fetch climate data for {commodity}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("CLIMATE DATA FETCH COMPLETE")
    logger.info("=" * 80)


def fetch_market_data(commodities: list, start_date: str, end_date: str):
    """Fetch market data for all commodities."""
    logger.info("=" * 80)
    logger.info("FETCHING MARKET DATA")
    logger.info("=" * 80)
    
    fetcher = MarketDataFetcher()
    validator = DataValidator()
    
    for commodity in commodities:
        logger.info(f"\nFetching market data for {commodity}...")
        
        try:
            # Fetch front contract (placeholder - use actual contract symbols in production)
            df = fetcher.fetch_futures_data(
                commodity,
                f"{commodity}_front",
                start_date,
                end_date
            )
            
            # Validate
            is_valid, issues = validator.validate_price_data(df, commodity)
            
            if not is_valid:
                logger.warning(f"Validation issues for {commodity}:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.info(f"✓ {commodity}: {len(df)} records, valid")
            
        except Exception as e:
            logger.error(f"Failed to fetch market data for {commodity}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("MARKET DATA FETCH COMPLETE")
    logger.info("=" * 80)


def fetch_cot_data(commodities: list, start_date: str, end_date: str):
    """Fetch COT positioning data."""
    logger.info("=" * 80)
    logger.info("FETCHING COT DATA")
    logger.info("=" * 80)
    
    fetcher = COTFetcher()
    
    for commodity in commodities:
        logger.info(f"\nFetching COT data for {commodity}...")
        
        try:
            df = fetcher.fetch_cot(commodity, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No COT data available for {commodity}")
            else:
                logger.info(f"✓ {commodity}: {len(df)} COT reports")
            
        except Exception as e:
            logger.error(f"Failed to fetch COT data for {commodity}: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("COT DATA FETCH COMPLETE")
    logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fetch historical data for Climate Futures Trading System')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--commodities', nargs='+', default=PHASE_1_COMMODITIES, help='Commodities to fetch')
    parser.add_argument('--data-types', nargs='+', default=['climate', 'market', 'cot'], 
                       help='Data types to fetch (climate, market, cot)')
    
    args = parser.parse_args()
    
    logger.info("Starting historical data fetch...")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Commodities: {args.commodities}")
    logger.info(f"Data types: {args.data_types}")
    
    # Fetch each data type
    if 'climate' in args.data_types:
        fetch_climate_data(args.commodities, args.start_date, args.end_date)
    
    if 'market' in args.data_types:
        fetch_market_data(args.commodities, args.start_date, args.end_date)
    
    if 'cot' in args.data_types:
        fetch_cot_data(args.commodities, args.start_date, args.end_date)
    
    logger.info("\n✓ All data fetching complete!")


if __name__ == '__main__':
    main()