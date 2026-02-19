"""
Yahoo Finance ticker mappings for commodities.
"""

# Yahoo Finance uses specific ticker symbols for futures
# Format: Symbol=F for continuous front month
YAHOO_TICKERS = {
    'corn': 'ZC=F',           # Corn Futures
    'soybeans': 'ZS=F',       # Soybean Futures
    'wheat': 'ZW=F',          # Wheat Futures
    'coffee': 'KC=F',         # Coffee Futures
    'sugar': 'SB=F',          # Sugar Futures
    'cotton': 'CT=F',         # Cotton Futures
    'cocoa': 'CC=F',          # Cocoa Futures
    'natural_gas': 'NG=F',    # Natural Gas Futures
    'crude_oil': 'CL=F',      # Crude Oil Futures
}

# Alternative: specific contract months
# Format: ZCH24.CBT (Corn March 2024)
# For now, we'll use continuous front month (=F)

def get_yahoo_ticker(commodity: str) -> str:
    """Get Yahoo Finance ticker for commodity."""
    ticker = YAHOO_TICKERS.get(commodity.lower())
    if ticker is None:
        raise ValueError(f"No Yahoo Finance ticker defined for {commodity}")
    return ticker