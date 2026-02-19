"""
Database schema definitions.
"""
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Table, MetaData
from datetime import datetime

Base = declarative_base()


class ClimateData(Base):
    """Climate data table."""
    __tablename__ = 'climate_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    commodity = Column(String(50), nullable=False)
    region = Column(String(100), nullable=False)
    date = Column(DateTime, nullable=False)
    temp_avg = Column(Float)
    temp_max = Column(Float)
    temp_min = Column(Float)
    precipitation = Column(Float)
    solar_radiation = Column(Float)
    wind_speed = Column(Float)
    relative_humidity = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_climate_commodity_date', 'commodity', 'date'),
        Index('idx_climate_region_date', 'region', 'date'),
    )


class MarketData(Base):
    """Market price data table."""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    commodity = Column(String(50), nullable=False)
    contract = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    open_interest = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_market_commodity_date', 'commodity', 'date'),
        Index('idx_market_contract_date', 'contract', 'date'),
    )


class SignalHistory(Base):
    """Signal history table."""
    __tablename__ = 'signal_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    commodity = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    raw_signal = Column(Float, nullable=False)
    smoothed_signal = Column(Float)
    final_weight = Column(Float)
    council_decision = Column(String(20))
    data_integrity_score = Column(Float)
    feature_drift_score = Column(Float)
    model_stability_score = Column(Float)
    regime_score = Column(Float)
    red_team_penalty = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_signal_commodity_date', 'commodity', 'date'),
    )


class PositionHistory(Base):
    """Position history table."""
    __tablename__ = 'position_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    commodity = Column(String(50), nullable=False)
    contract = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False)
    position = Column(Float, nullable=False)
    entry_price = Column(Float)
    current_price = Column(Float)
    unrealized_pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_position_commodity_date', 'commodity', 'date'),
    )


class OrderHistory(Base):
    """Order history table."""
    __tablename__ = 'order_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(50), unique=True, nullable=False)
    commodity = Column(String(50), nullable=False)
    contract = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)  # BUY or SELL
    quantity = Column(Float, nullable=False)
    order_type = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    limit_price = Column(Float)
    stop_price = Column(Float)
    filled_price = Column(Float)
    filled_quantity = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    filled_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_order_commodity', 'commodity'),
        Index('idx_order_status', 'status'),
    )


class PerformanceMetrics(Base):
    """Daily performance metrics table."""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, unique=True)
    portfolio_value = Column(Float, nullable=False)
    daily_pnl = Column(Float)
    daily_return = Column(Float)
    sharpe_ratio_60d = Column(Float)
    max_drawdown = Column(Float)
    volatility_20d = Column(Float)
    n_positions = Column(Integer)
    gross_exposure = Column(Float)
    net_exposure = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    __table_args__ = (
        Index('idx_performance_date', 'date'),
    )


def create_all_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)
    logger.info("All database tables created successfully")


def drop_all_tables(engine):
    """Drop all tables (use with caution!)."""
    Base.metadata.drop_all(engine)
    logger.warning("All database tables dropped")