# test_alignment.py
import pandas as pd
from data.market_fetcher import MarketDataFetcher
from data.climate_fetcher import NASAPowerFetcher
from features.feature_pipeline import FeaturePipeline

# Fetch data
climate = NASAPowerFetcher()
market = MarketDataFetcher(provider='yahoo')

regions = climate.fetch_commodity_regions('corn', '2020-01-01', '2023-12-31')
prices = market.fetch_futures_data('corn', 'N/A', '2020-01-01', '2023-12-31')

print(f"Climate regions: {len(regions)}")
for r, df in regions.items():
    print(f"  {r}: {len(df)} days, {df['date'].min()} to {df['date'].max()}")

print(f"\nPrices: {len(prices)} days, {prices['date'].min()} to {prices['date'].max()}")

# Generate features
pipeline = FeaturePipeline('corn')
features = pipeline.run(regions)

print(f"\nFeatures: {len(features)} days, {features['date'].min()} to {features['date'].max()}")
print(f"Feature columns: {len(features.columns)}")

# Check alignment
common_dates = pd.merge(features[['date']], prices[['date']], on='date', how='inner')
print(f"\nCommon dates: {len(common_dates)}")
print(f"Missing in features: {len(prices) - len(common_dates)}")
print(f"Missing in prices: {len(features) - len(common_dates)}")