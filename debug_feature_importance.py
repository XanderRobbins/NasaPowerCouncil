"""
Check which features actually matter.
"""
import pandas as pd
import numpy as np
from data.market_fetcher import MarketDataFetcher
from features.feature_pipeline import FeaturePipeline
from data.climate_fetcher import NASAPowerFetcher
from models.ridge_model import RollingRidgeModel

# Fetch data
climate_fetcher = NASAPowerFetcher()
market_fetcher = MarketDataFetcher(provider='yahoo')

commodity = 'corn'
print(f"Analyzing feature importance for {commodity}...")

# Get data
climate_data = climate_fetcher.fetch_commodity_regions(commodity, '2015-01-01', '2024-12-31')
prices_df = market_fetcher.fetch_futures_data(commodity, 'N/A', '2015-01-01', '2024-12-31')

# Generate features
pipeline = FeaturePipeline(commodity)
features = pipeline.run(climate_data)

# ALIGN dates between features and prices
common_dates = pd.merge(
    features[['date']],
    prices_df[['date']],
    on='date',
    how='inner'
)['date']

features_aligned = features[features['date'].isin(common_dates)].reset_index(drop=True)
prices_aligned = prices_df[prices_df['date'].isin(common_dates)]['close'].reset_index(drop=True)

print(f"Features shape: {features_aligned.shape}")
print(f"Prices length: {len(prices_aligned)}")

# Train model
model = RollingRidgeModel()
X, feature_names = model.prepare_features(features_aligned)
y = model.compute_target(prices_aligned).values

print(f"X shape after feature prep: {X.shape}")
print(f"y length: {len(y)}")

# Trim to shortest
min_len = min(len(X), len(y))
X = X[:min_len]
y = y[:min_len]

print(f"After alignment - X: {X.shape}, y: {len(y)}")

# HANDLE NaNs IN X FIRST (before filtering by y)
print(f"NaNs in X before cleaning: {np.isnan(X).sum()}")
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f"NaNs in X after cleaning: {np.isnan(X).sum()}")

# Filter valid data (non-NaN targets)
valid_idx = ~np.isnan(y)
print(f"Valid samples: {valid_idx.sum()} out of {len(valid_idx)}")

X_valid = X[valid_idx]
y_valid = y[valid_idx]

# Use first 2000 valid samples (or all if less)
n_samples = min(2000, len(X_valid))
X_train = X_valid[:n_samples]
y_train = y_valid[:n_samples]

print(f"Training on {len(X_train)} samples")
print(f"Final check - NaNs in X_train: {np.isnan(X_train).sum()}")
print(f"Final check - NaNs in y_train: {np.isnan(y_train).sum()}")

# Train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

print(f"After scaling - NaNs in X_scaled: {np.isnan(X_scaled).sum()}")

model.model.fit(X_scaled, y_train)

# Model performance
train_score = model.model.score(X_scaled, y_train)
print(f"\nModel R² on training data: {train_score:.4f}")

# Get feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.model.coef_,
    'abs_coef': np.abs(model.model.coef_)
}).sort_values('abs_coef', ascending=False)

print("\n" + "=" * 80)
print("TOP 10 MOST IMPORTANT FEATURES")
print("=" * 80)
print(importance.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("BOTTOM 10 LEAST IMPORTANT FEATURES")
print("=" * 80)
print(importance.tail(10).to_string(index=False))

# Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
top_features = importance.head(15)
plt.barh(range(len(top_features)), top_features['abs_coef'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Absolute Coefficient', fontsize=12)
plt.title('Top 15 Feature Importance (Ridge Regression)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('data_storage/results/feature_importance.png', dpi=150)
print("\n✓ Saved plot to feature_importance.png")

# Check correlation of top features with returns
print("\n" + "=" * 80)
print("TOP FEATURE CORRELATIONS WITH FORWARD RETURNS")
print("=" * 80)

from scipy.stats import spearmanr, pearsonr

for i, feat in enumerate(importance.head(10)['feature']):
    feat_idx = feature_names.index(feat)
    feat_values = X_train[:, feat_idx]
    
    # Both Spearman (rank) and Pearson (linear)
    spear_corr, spear_p = spearmanr(feat_values, y_train)
    pear_corr, pear_p = pearsonr(feat_values, y_train)
    
    sig_spear = "***" if spear_p < 0.001 else "**" if spear_p < 0.01 else "*" if spear_p < 0.05 else ""
    sig_pear = "***" if pear_p < 0.001 else "**" if pear_p < 0.01 else "*" if pear_p < 0.05 else ""
    
    print(f"{i+1:2}. {feat:45} Spear: {spear_corr:+.4f}{sig_spear:3}  Pearson: {pear_corr:+.4f}{sig_pear:3}")

print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05")

# Overall prediction quality check
print("\n" + "=" * 80)
print("MODEL PREDICTION QUALITY")
print("=" * 80)

predictions = model.model.predict(X_scaled)
pred_corr, pred_p = spearmanr(predictions, y_train)

print(f"Prediction-Actual Correlation: {pred_corr:.4f} (p={pred_p:.4f})")
print(f"Model R²: {train_score:.4f}")
print(f"RMSE: {np.sqrt(np.mean((predictions - y_train)**2)):.6f}")

# Directional accuracy
correct_direction = np.sign(predictions) == np.sign(y_train)
direction_accuracy = correct_direction.mean()
print(f"Directional Accuracy: {direction_accuracy:.2%}")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

if train_score < 0.05:
    print("❌ CRITICAL: Model has almost NO predictive power (R² < 5%)")
    print("   → Weather features do not predict 20-day returns")
    print("   → Strategy thesis may be invalid")
    print("\n   RECOMMENDED ACTIONS:")
    print("   1. Try shorter horizon (5-day instead of 20-day)")
    print("   2. Only trade during growing season (April-September)")
    print("   3. Focus on extreme events only (>2 std deviations)")
    print("   4. Consider predicting volatility instead of returns")
elif train_score < 0.15:
    print("⚠️  WARNING: Model has weak predictive power (R² < 15%)")
    print("   → Edge exists but is very small")
    print("   → Transaction costs likely eat all profits")
    print("\n   RECOMMENDED ACTIONS:")
    print("   1. Enable the Council to filter bad trades")
    print("   2. Reduce position sizes (lower vol target)")
    print("   3. Trade only highest-conviction signals")
elif train_score < 0.30:
    print("✓ MODERATE: Model has some predictive power")
    print("   → Strategy may work with proper risk management")
    print("   → Enable Council and optimize parameters")
else:
    print("✓✓ STRONG: Model has good predictive power")
    print("   → Strategy should be profitable")

if direction_accuracy < 0.52:
    print(f"\n❌ Directional accuracy {direction_accuracy:.1%} → No edge on direction")
elif direction_accuracy < 0.55:
    print(f"\n⚠️  Directional accuracy {direction_accuracy:.1%} → Weak edge")
else:
    print(f"\n✓ Directional accuracy {direction_accuracy:.1%} → Real edge!")

print("=" * 80)