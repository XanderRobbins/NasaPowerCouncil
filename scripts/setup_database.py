"""
Script to set up the database schema.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from storage.database import DatabaseManager
from storage.schemas import create_all_tables
from config.settings import DATABASE_URL
from loguru import logger


def main():
    """Set up database."""
    logger.info("=" * 80)
    logger.info("DATABASE SETUP")
    logger.info("=" * 80)
    logger.info(f"Database URL: {DATABASE_URL}")
    
    # Initialize database manager
    db = DatabaseManager(DATABASE_URL)
    
    # Create all tables
    logger.info("\nCreating database tables...")
    create_all_tables(db.engine)
    
    logger.info("\n✓ Database setup complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()