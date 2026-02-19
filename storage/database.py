"""
Database interface for time-series data storage.
"""
import pandas as pd
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Optional, List
from loguru import logger

from config.settings import DATABASE_URL


class DatabaseManager:
    """
    Manage database connections and operations.
    
    Uses SQLAlchemy for ORM and raw SQL support.
    """
    
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.metadata = MetaData()
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info(f"Database initialized: {database_url}")
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """
        Save DataFrame to database table.
        
        Args:
            df: DataFrame to save
            table_name: Name of table
            if_exists: What to do if table exists ('fail', 'replace', 'append')
        """
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Saved {len(df)} records to {table_name}")
        except Exception as e:
            logger.error(f"Failed to save to {table_name}: {e}")
            raise
    
    def load_dataframe(self, 
                      table_name: str,
                      columns: Optional[List[str]] = None,
                      where: Optional[str] = None) -> pd.DataFrame:
        """
        Load DataFrame from database table.
        
        Args:
            table_name: Name of table
            columns: List of columns to select (None = all)
            where: WHERE clause (e.g., "date > '2020-01-01'")
            
        Returns:
            DataFrame
        """
        try:
            query = f"SELECT "
            
            if columns:
                query += ", ".join(columns)
            else:
                query += "*"
            
            query += f" FROM {table_name}"
            
            if where:
                query += f" WHERE {where}"
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} records from {table_name}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from {table_name}: {e}")
            raise
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute raw SQL query and return results."""
        try:
            df = pd.read_sql(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def get_table_info(self, table_name: str) -> dict:
        """Get information about a table."""
        from sqlalchemy import inspect
        
        if not self.table_exists(table_name):
            return {}
        
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name)
        
        return {
            'columns': [col['name'] for col in columns],
            'column_types': {col['name']: str(col['type']) for col in columns}
        }
    
    def delete_records(self, table_name: str, where: str):
        """
        Delete records from table.
        
        Args:
            table_name: Table name
            where: WHERE clause (e.g., "date < '2020-01-01'")
        """
        query = f"DELETE FROM {table_name} WHERE {where}"
        
        with self.get_session() as session:
            result = session.execute(query)
            deleted_count = result.rowcount
            logger.info(f"Deleted {deleted_count} records from {table_name}")


# Global database instance
db_manager = DatabaseManager()