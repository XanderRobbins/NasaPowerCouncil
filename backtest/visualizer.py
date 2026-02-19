"""
Visualization tools for backtest analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from pathlib import Path

from config.settings import RESULTS_PATH


class BacktestVisualizer:
    """
    Visualize backtest results.
    """
    
    def __init__(self, results: pd.DataFrame):
        self.results = results
        self.results['date'] = pd.to_datetime(self.results['date'])
        
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (14, 8)
        
    def plot_equity_curve(self, save_path: Optional[Path] = None):
        """Plot portfolio equity curve."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(self.results['date'], 
                self.results['portfolio_value'], 
                linewidth=2, 
                color='darkblue',
                label='Portfolio Value')
        
        # Add buy & hold benchmark (if available)
        initial_value = self.results['portfolio_value'].iloc[0]
        ax.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved equity curve to {save_path}")
        
        plt.show()
    
    def plot_drawdown(self, save_path: Optional[Path] = None):
        """Plot drawdown chart."""
        cumulative = (1 + self.results['portfolio_return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative / running_max - 1) * 100  # Convert to percentage
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.fill_between(self.results['date'], 
                        drawdown, 
                        0, 
                        color='red', 
                        alpha=0.3)
        ax.plot(self.results['date'], 
                drawdown, 
                color='darkred', 
                linewidth=1.5,
                label='Drawdown')
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_date = self.results['date'].iloc[max_dd_idx]
        max_dd_value = drawdown.iloc[max_dd_idx]
        
        ax.scatter([max_dd_date], [max_dd_value], 
                  color='darkred', s=100, zorder=5,
                  label=f'Max DD: {max_dd_value:.2f}%')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved drawdown chart to {save_path}")
        
        plt.show()
    
    def plot_returns_distribution(self, save_path: Optional[Path] = None):
        """Plot distribution of returns."""
        returns_pct = self.results['portfolio_return'] * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(returns_pct, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
        axes[0].set_xlabel('Daily Return (%)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns_pct.dropna(), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved returns distribution to {save_path}")
        
        plt.show()
    
    def plot_rolling_metrics(self, window: int = 60, save_path: Optional[Path] = None):
        """Plot rolling Sharpe and volatility."""
        returns = self.results['portfolio_return']
        
        # Rolling metrics
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Rolling Sharpe
        axes[0].plot(self.results['date'], rolling_sharpe, color='green', linewidth=2)
        axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_ylabel(f'Rolling Sharpe ({window}d)', fontsize=12)
        axes[0].set_title('Rolling Sharpe Ratio', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rolling Vol
        axes[1].plot(self.results['date'], rolling_vol, color='orange', linewidth=2)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].set_ylabel(f'Rolling Vol ({window}d, Ann. %)', fontsize=12)
        axes[1].set_title('Rolling Volatility', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved rolling metrics to {save_path}")
        
        plt.show()
    
    def plot_signal_analysis(self, commodities: list, save_path: Optional[Path] = None):
        """Plot signals and positions for each commodity."""
        n_commodities = len(commodities)
        fig, axes = plt.subplots(n_commodities, 1, figsize=(14, 4*n_commodities))
        
        if n_commodities == 1:
            axes = [axes]
        
        for i, commodity in enumerate(commodities):
            signal_col = f'{commodity}_signal'
            position_col = f'{commodity}_position'
            
            if signal_col not in self.results.columns:
                continue
            
            ax = axes[i]
            ax2 = ax.twinx()
            
            # Plot signal
            ax.plot(self.results['date'], 
                   self.results[signal_col], 
                   color='blue', 
                   linewidth=1.5, 
                   alpha=0.7,
                   label='Signal')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Plot position
            ax2.plot(self.results['date'], 
                    self.results[position_col], 
                    color='red', 
                    linewidth=1.5, 
                    alpha=0.7,
                    label='Position')
            
            ax.set_ylabel('Signal', fontsize=10, color='blue')
            ax2.set_ylabel('Position', fontsize=10, color='red')
            ax.set_title(f'{commodity.upper()} - Signal vs Position', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved signal analysis to {save_path}")
        
        plt.show()
    
    def plot_monthly_returns_heatmap(self, save_path: Optional[Path] = None):
        """Plot monthly returns heatmap."""
        # Compute monthly returns
        self.results['year'] = self.results['date'].dt.year
        self.results['month'] = self.results['date'].dt.month
        
        monthly_returns = self.results.groupby(['year', 'month'])['portfolio_return'].sum() * 100
        monthly_returns = monthly_returns.reset_index()
        
        # Pivot for heatmap
        pivot = monthly_returns.pivot(index='year', columns='month', values='portfolio_return')
        
        # Month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = month_names
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='RdYlGn', 
                   center=0,
                   cbar_kws={'label': 'Return (%)'},
                   linewidths=0.5,
                   ax=ax)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved monthly heatmap to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, commodities: list):
        """Generate all visualizations and save to results folder."""
        report_dir = RESULTS_PATH / 'backtest_report'
        report_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating full backtest report in {report_dir}...")
        
        self.plot_equity_curve(save_path=report_dir / 'equity_curve.png')
        self.plot_drawdown(save_path=report_dir / 'drawdown.png')
        self.plot_returns_distribution(save_path=report_dir / 'returns_distribution.png')
        self.plot_rolling_metrics(save_path=report_dir / 'rolling_metrics.png')
        self.plot_signal_analysis(commodities, save_path=report_dir / 'signal_analysis.png')
        self.plot_monthly_returns_heatmap(save_path=report_dir / 'monthly_heatmap.png')
        
        print(f"\n✓ Full report generated in {report_dir}")


def visualize_backtest(results: pd.DataFrame, commodities: list):
    """Convenience function to generate all visualizations."""
    visualizer = BacktestVisualizer(results)
    visualizer.generate_full_report(commodities)