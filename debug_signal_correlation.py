"""
Check if signals actually predict future returns.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# Load backtest results
try:
    results = pd.read_parquet('data_storage/results/backtest_results.parquet')
except FileNotFoundError:
    print("ERROR: Run backtest first to generate results!")
    exit(1)

print("=" * 80)
print("SIGNAL-RETURN CORRELATION ANALYSIS")
print("=" * 80)
print(f"Total records: {len(results)}")
print(f"Date range: {results['date'].min()} to {results['date'].max()}")

# Analyze each commodity
for commodity in ['corn', 'soybeans']:
    signal_col = f'{commodity}_signal'
    position_col = f'{commodity}_position'
    
    if signal_col not in results.columns:
        print(f"\n⚠️  {commodity} not found in results")
        continue
    
    print("\n" + "=" * 80)
    print(f"{commodity.upper()} ANALYSIS")
    print("=" * 80)
    
    # Get signal and forward returns
    signals = results[signal_col].values
    
    # Calculate forward returns at different horizons
    for horizon in [1, 5, 10, 20]:
        forward_returns = results['portfolio_return'].shift(-horizon).values
        
        # Remove NaNs
        mask = ~(np.isnan(signals) | np.isnan(forward_returns))
        signals_clean = signals[mask]
        returns_clean = forward_returns[mask]
        
        if len(signals_clean) < 50:
            continue
        
        # Compute correlations
        spear_corr, spear_p = spearmanr(signals_clean, returns_clean)
        pear_corr, pear_p = pearsonr(signals_clean, returns_clean)
        
        sig = "✓✓✓" if spear_p < 0.001 else "✓✓" if spear_p < 0.01 else "✓" if spear_p < 0.05 else "✗"
        
        print(f"\n{horizon}-Day Forward Returns:")
        print(f"  Spearman: {spear_corr:+.4f} (p={spear_p:.4f}) {sig}")
        print(f"  Pearson:  {pear_corr:+.4f} (p={pear_p:.4f})")
        print(f"  Samples:  {len(signals_clean)}")
    
    # Quintile analysis (20-day horizon)
    horizon = 20
    forward_returns = results['portfolio_return'].shift(-horizon).values
    mask = ~(np.isnan(signals) | np.isnan(forward_returns))
    signals_clean = signals[mask]
    returns_clean = forward_returns[mask]
    
    if len(signals_clean) >= 50:
        print(f"\n20-Day Returns by Signal Quintile:")
        
        # Split into quintiles
        try:
            quintiles = pd.qcut(signals_clean, q=5, labels=['Q1 (Short)', 'Q2', 'Q3', 'Q4', 'Q5 (Long)'], duplicates='drop')
            quintile_df = pd.DataFrame({
                'signal': signals_clean,
                'return': returns_clean,
                'quintile': quintiles
            })
            
            quintile_stats = quintile_df.groupby('quintile')['return'].agg(['mean', 'count', 'std'])
            print(quintile_stats)
            
            # Check monotonicity (should increase from Q1 to Q5)
            means = quintile_stats['mean'].values
            is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
            print(f"\n  Monotonic (Q1→Q5)? {'YES ✓' if is_monotonic else 'NO ✗'}")
            
        except ValueError as e:
            print(f"  (Could not create quintiles: {e})")
    
    # Plot scatter
    if len(signals_clean) >= 50:
        plt.figure(figsize=(14, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(signals_clean, returns_clean, alpha=0.3, s=20, c='steelblue')
        plt.xlabel(f'{commodity.upper()} Signal', fontsize=12)
        plt.ylabel('20-Day Forward Return', fontsize=12)
        plt.title(f'{commodity.upper()}: Signal vs Return\n(Spearman r={spear_corr:.3f})', 
                 fontsize=13, fontweight='bold')
        plt.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.grid(alpha=0.3)
        
        # Quintile bar chart
        plt.subplot(1, 2, 2)
        if 'quintile_stats' in locals():
            quintile_stats['mean'].plot(kind='bar', color='steelblue', edgecolor='black')
            plt.title(f'{commodity.upper()}: Avg Return by Signal Strength', fontsize=13, fontweight='bold')
            plt.ylabel('Avg 20-Day Return', fontsize=12)
            plt.xlabel('Signal Quintile', fontsize=12)
            plt.axhline(0, color='red', linestyle='--', linewidth=1)
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'data_storage/results/signal_correlation_{commodity}.png', dpi=150)
        print(f"\n✓ Saved plot to signal_correlation_{commodity}.png")

print("\n" + "=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)

# Load again for portfolio-level analysis
signals_corn = results['corn_signal'].values if 'corn_signal' in results.columns else np.zeros(len(results))
signals_soy = results['soybeans_signal'].values if 'soybeans_signal' in results.columns else np.zeros(len(results))

# Average signal
avg_signal = (signals_corn + signals_soy) / 2
forward_returns = results['portfolio_return'].shift(-20).values

mask = ~(np.isnan(avg_signal) | np.isnan(forward_returns))
corr, p_val = spearmanr(avg_signal[mask], forward_returns[mask])

print(f"\nPortfolio-Level Signal → Return Correlation: {corr:+.4f} (p={p_val:.4f})")

if abs(corr) < 0.05:
    print("\n❌ CRITICAL: No correlation detected!")
    print("   → Strategy has NO predictive power")
    print("   → Losses are due to lack of edge, not execution")
elif abs(corr) < 0.15:
    print("\n⚠️  WARNING: Very weak correlation")
    print("   → Edge exists but is tiny")
    print("   → Transaction costs likely dominate")
elif abs(corr) < 0.30:
    print("\n✓ MODERATE: Detectable correlation")
    print("   → Strategy may work with optimization")
else:
    print("\n✓✓ STRONG: Good correlation!")
    print("   → Strategy should be profitable")

print("=" * 80)