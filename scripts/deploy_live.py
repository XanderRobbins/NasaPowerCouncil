"""
Script to deploy live trading engine.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from live.live_engine import start_live_trading
from config.settings import PHASE_1_COMMODITIES
from loguru import logger
import argparse


def main():
    """Deploy live trading engine."""
    parser = argparse.ArgumentParser(description='Deploy Climate Futures Trading System')
    parser.add_argument('--paper', action='store_true', help='Run in paper trading mode')
    parser.add_argument('--run-time', default='16:30', help='Daily run time (HH:MM)')
    parser.add_argument('--run-now', action='store_true', help='Run immediately instead of scheduling')
    
    args = parser.parse_args()
    
    commodities = PHASE_1_COMMODITIES
    
    logger.info("=" * 80)
    logger.info("DEPLOYING CLIMATE FUTURES TRADING SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Mode: {'PAPER TRADING' if args.paper else 'LIVE TRADING'}")
    logger.info(f"Commodities: {commodities}")
    logger.info(f"Schedule: Daily at {args.run_time}")
    logger.info("=" * 80)
    
    if not args.paper:
        confirm = input("\n⚠️  You are about to deploy LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            logger.info("Deployment cancelled.")
            return
    
    if args.run_now:
        # Run once immediately
        from live.live_engine import LiveTradingEngine
        engine = LiveTradingEngine(commodities, paper_trading=args.paper)
        engine.run_daily_update()
    else:
        # Start scheduler
        start_live_trading(commodities, paper_trading=args.paper, run_time=args.run_time)


if __name__ == '__main__':
    main()