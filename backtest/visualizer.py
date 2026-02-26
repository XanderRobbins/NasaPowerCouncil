"""
Backtest visualization with dark theme.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger

# Set dark theme globally
plt.style.use('dark_background')
sns.set_palette("bright")

class BacktestVisualizer:
    """
    Create visualizations for backtest results.
    Dark theme for easy viewing.
    """
    
    def __init__(self, figsize: tuple = (14, 10)):
        self.figsize = figsize
        
        # Custom color scheme (bright colors on dark background)
        self.colors = {
            'equity': '#00ff41',      # Bright green
            'drawdown': '#ff073a',    # Bright red
            'signal': '#00d9ff',      # Cyan
            'position': '#ffb700',    # Orange
            'benchmark': '#888888',   # Gray
        }
    
    def plot_equity_curve(self, results: pd.DataFrame, ax: Optional[plt.Axes] = None):
        """Plot portfolio equity curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(results['date'], results['portfolio_value'], 
               color=self.colors['equity'], linewidth=2, label='Portfolio Value')
        
        # Add starting value reference line
        initial_value = results['portfolio_value'].iloc[0]
        ax.axhline(initial_value, color=self.colors['benchmark'], 
                  linestyle='--', alpha=0.5, label='Initial Capital')
        
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        return ax
    
    def plot_drawdown(self, results: pd.DataFrame, ax: Optional[plt.Axes] = None):
        """Plot drawdown chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        
        # Compute drawdown
        cumulative = results['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(results['date'], drawdown * 100, 0, 
                       color=self.colors['drawdown'], alpha=0.6)
        ax.plot(results['date'], drawdown * 100, 
               color=self.colors['drawdown'], linewidth=1)
        
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Drawdown (%)', fontsize=12, color='white')
        ax.set_title('Underwater Plot (Drawdown)', fontsize=14, fontweight='bold', color='white')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        return ax
    
    def plot_returns_distribution(self, results: pd.DataFrame, ax: Optional[plt.Axes] = None):
        """Plot distribution of daily returns."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        
        returns = results['portfolio_return'].dropna()
        
        # Histogram
        ax.hist(returns * 100, bins=50, color=self.colors['equity'], 
               alpha=0.7, edgecolor='white')
        
        # Add mean line
        mean_return = returns.mean() * 100
        ax.axvline(mean_return, color='yellow', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_return:.3f}%')
        
        ax.set_xlabel('Daily Return (%)', fontsize=12, color='white')
        ax.set_ylabel('Frequency', fontsize=12, color='white')
        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        ax.tick_params(colors='white')
        
        return ax
    
    def plot_signals(self, results: pd.DataFrame, commodities: list, 
                    ax: Optional[plt.Axes] = None):
        """Plot signals over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        
        for commodity in commodities:
            signal_col = f'{commodity}_signal'
            if signal_col in results.columns:
                ax.plot(results['date'], results[signal_col], 
                       label=commodity.capitalize(), alpha=0.8, linewidth=1.5)
        
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Signal Strength', fontsize=12, color='white')
        ax.set_title('Trading Signals', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        return ax
    
    def plot_positions(self, results: pd.DataFrame, commodities: list,
                      ax: Optional[plt.Axes] = None):
        """Plot position sizes over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        
        for commodity in commodities:
            position_col = f'{commodity}_position'
            if position_col in results.columns:
                ax.plot(results['date'], results[position_col], 
                       label=commodity.capitalize(), alpha=0.8, linewidth=1.5)
        
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Position Size (% of Portfolio)', fontsize=12, color='white')
        ax.set_title('Portfolio Positions', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        return ax
    
    def plot_rolling_sharpe(self, results: pd.DataFrame, window: int = 60,
                           ax: Optional[plt.Axes] = None):
        """Plot rolling Sharpe ratio."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        
        returns = results['portfolio_return']
        rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252)
        
        ax.plot(results['date'], rolling_sharpe, 
               color=self.colors['signal'], linewidth=2)
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax.axhline(1, color='yellow', linestyle='--', alpha=0.5, label='Sharpe = 1')
        
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12, color='white')
        ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        
        return ax
    
    def create_full_report(self, results: pd.DataFrame, commodities: list,
                          save_path: Optional[Path] = None):
        """Create comprehensive backtest report with all plots."""
        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_facecolor('#0a0a0a')  # Very dark background
        
        # Create grid
        gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3)
        
        # Plot 1: Equity curve (top, spans both columns)
        ax1 = fig.add_subplot(gs[0:2, :])
        self.plot_equity_curve(results, ax=ax1)
        
        # Plot 2: Drawdown
        ax2 = fig.add_subplot(gs[2, :])
        self.plot_drawdown(results, ax=ax2)
        
        # Plot 3: Returns distribution
        ax3 = fig.add_subplot(gs[3, 0])
        self.plot_returns_distribution(results, ax=ax3)
        
        # Plot 4: Rolling Sharpe
        ax4 = fig.add_subplot(gs[3, 1])
        self.plot_rolling_sharpe(results, ax=ax4)
        
        # Plot 5: Signals
        ax5 = fig.add_subplot(gs[4, :])
        self.plot_signals(results, commodities, ax=ax5)
        
        # Plot 6: Positions
        ax6 = fig.add_subplot(gs[5, :])
        self.plot_positions(results, commodities, ax=ax6)
        
        # Overall title
        fig.suptitle('Backtest Performance Report', 
                    fontsize=16, fontweight='bold', color='white', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0a0a0a', 
                       edgecolor='none', bbox_inches='tight')
            logger.info(f"Saved backtest visualization to {save_path}")
        
        return fig


def plot_backtest_results(results: pd.DataFrame, 
                         commodities: list,
                         save_path: Optional[str] = None):
    """
    Convenience function to create full backtest visualization.
    
    Args:
        results: Backtest results DataFrame
        commodities: List of commodities
        save_path: Path to save figure
    """
    visualizer = BacktestVisualizer()
    fig = visualizer.create_full_report(results, commodities, save_path)
    plt.show()
    
    return fig