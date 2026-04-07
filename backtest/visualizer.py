"""
Interactive backtest visualization using Plotly.
Premium styling with full interactivity.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional
from loguru import logger

<<<<<<< HEAD
# Theme colors (from preferred UI)
COLORS = {
    'bg_primary': '#282838',
    'bg_deep': '#202028',
    'accent': '#D97757',
    'text_primary': '#F0F0F0',
    'text_secondary': '#9090A8',
    'border': '#38384A',
    'profit': '#4CAF50',
    'loss': '#FF6B6B',
    'neutral': '#64B5F6',
    'warning': '#FFC107',
}

class BacktestVisualizer:
    """Interactive backtest visualization with Plotly."""

    def __init__(self):
        self.colors = COLORS

    def _get_layout_template(self, title: str = ""):
        """Get standard layout template."""
        return dict(
            title=dict(text=title, font=dict(size=16, color=COLORS['text_primary'], family='Arial Black')),
            template='plotly_dark',
            plot_bgcolor=COLORS['bg_deep'],
            paper_bgcolor=COLORS['bg_primary'],
            hovermode='x unified',
            margin=dict(l=60, r=40, t=50, b=50),
            font=dict(family='Arial, sans-serif', size=11, color=COLORS['text_primary']),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS['border'],
                showline=True,
                linewidth=1,
                linecolor=COLORS['border'],
                color=COLORS['text_secondary'],
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor=COLORS['border'],
                showline=True,
                linewidth=1,
                linecolor=COLORS['border'],
                color=COLORS['text_secondary'],
            ),
        )

    def plot_equity_curve(self, results: pd.DataFrame) -> go.Figure:
        """Interactive equity curve."""
        fig = go.Figure()

        x = results['date']
        y = results['portfolio_value']
        initial = y.iloc[0]

        # Main line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name='Portfolio Value',
            mode='lines',
            line=dict(color=COLORS['accent'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba(217, 119, 87, 0.1)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Value: $%{y:,.0f}<extra></extra>',
        ))

        # Initial capital line
        fig.add_hline(
            y=initial,
            line_dash='dash',
            line_color=COLORS['text_secondary'],
            line_width=1,
            annotation_text='Initial Capital',
            annotation_position='right',
        )

        layout = self._get_layout_template('Equity Curve')
        layout.update(
            height=400,
            yaxis=dict(
                **layout['yaxis'],
                tickformat='$,.0f',
            )
        )

        fig.update_layout(**layout)
        return fig

    def plot_drawdown(self, results: pd.DataFrame) -> go.Figure:
        """Interactive drawdown with color coding."""
        cumulative = results['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        fig = go.Figure()

        # Create color array based on severity
        colors = []
        for dd in drawdown:
            if dd < -20:
                colors.append(COLORS['loss'])
            elif dd < -10:
                colors.append(COLORS['warning'])
            else:
                colors.append(COLORS['neutral'])

        fig.add_trace(go.Bar(
            x=results['date'],
            y=drawdown,
            name='Drawdown',
            marker=dict(color=colors),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>',
        ))

        fig.add_hline(y=0, line_color=COLORS['text_secondary'], line_width=1)

        layout = self._get_layout_template('Underwater Plot (Drawdown)')
        layout.update(height=300, showlegend=False)
        fig.update_layout(**layout)
        return fig

    def plot_returns_distribution(self, results: pd.DataFrame) -> go.Figure:
        """Interactive returns histogram."""
        returns = results['portfolio_return'].dropna() * 100

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns,
            name='Daily Returns',
            nbinsx=50,
            marker=dict(color=COLORS['accent'], line=dict(color=COLORS['border'], width=0.5)),
            hovertemplate='<b>Return Range: %{x:.2f}%</b><br>Frequency: %{y}<extra></extra>',
        ))

        mean_ret = returns.mean()
        median_ret = returns.median()

        fig.add_vline(x=mean_ret, line_dash='dash', line_color=COLORS['profit'],
                     annotation_text=f'Mean: {mean_ret:.2f}%', annotation_position='top right')
        fig.add_vline(x=median_ret, line_dash='dash', line_color=COLORS['neutral'],
                     annotation_text=f'Median: {median_ret:.2f}%', annotation_position='top left')

        layout = self._get_layout_template('Returns Distribution')
        layout.update(height=350, showlegend=False)
        fig.update_layout(**layout)
        return fig

    def plot_signals(self, results: pd.DataFrame, commodities: list) -> go.Figure:
        """Interactive signals chart."""
        fig = go.Figure()
=======
plt.style.use('dark_background')
sns.set_palette("bright")

class BacktestVisualizer:
    """
    Create visualizations for backtest results.
    Dark theme for easy viewing.
    """

    def __init__(self, figsize: tuple = (14, 10)):
        self.figsize = figsize

        self.colors = {
            'equity': '#00ff41',
            'drawdown': '#ff073a',
            'signal': '#00d9ff',
            'position': '#ffb700',
            'benchmark': '#888888',
        }

    def _format_xaxis_as_trading_days(self, ax, results: pd.DataFrame):
        """
        Replace synthetic date x-axis with 'Trading Day N' labels.
        Annotates real date range in the corner if 'real_date' is present.
        """
        n = len(results)
        if n == 0:
            return

        tick_indices = [int(i * (n - 1) / 5) for i in range(6)]
        tick_positions = results['date'].iloc[tick_indices]
        tick_labels = [f"Day {i}" for i in tick_indices]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=15, fontsize=9, color='white')

        if 'real_date' in results.columns:
            start = results['real_date'].iloc[0].strftime('%Y-%m-%d')
            end = results['real_date'].iloc[-1].strftime('%Y-%m-%d')
            ax.annotate(
                f"Active periods: {start} → {end}",
                xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=8, color='#888888'
            )

    def plot_equity_curve(self, results: pd.DataFrame, ax: Optional[plt.Axes] = None):
        """Plot portfolio equity curve."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(results['date'], results['portfolio_value'],
               color=self.colors['equity'], linewidth=2, label='Portfolio Value')

        initial_value = results['portfolio_value'].iloc[0]
        ax.axhline(initial_value, color=self.colors['benchmark'],
                  linestyle='--', alpha=0.5, label='Initial Capital')

        ax.set_xlabel('Trading Day', fontsize=12, color='white')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, color='white')
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        self._format_xaxis_as_trading_days(ax, results)

        return ax

    def plot_drawdown(self, results: pd.DataFrame, ax: Optional[plt.Axes] = None):
        """Plot drawdown chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        cumulative = results['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        ax.fill_between(results['date'], drawdown * 100, 0,
                       color=self.colors['drawdown'], alpha=0.6)
        ax.plot(results['date'], drawdown * 100,
               color=self.colors['drawdown'], linewidth=1)

        ax.set_xlabel('Trading Day', fontsize=12, color='white')
        ax.set_ylabel('Drawdown (%)', fontsize=12, color='white')
        ax.set_title('Underwater Plot (Drawdown)', fontsize=14, fontweight='bold', color='white')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        self._format_xaxis_as_trading_days(ax, results)

        return ax

    def plot_returns_distribution(self, results: pd.DataFrame, ax: Optional[plt.Axes] = None):
        """Plot distribution of daily returns."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        returns = results['portfolio_return'].dropna()

        ax.hist(returns * 100, bins=50, color=self.colors['equity'],
               alpha=0.7, edgecolor='white')

        mean_return = returns.mean() * 100
        ax.axvline(mean_return, color='yellow', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_return:.3f}%')

        ax.set_xlabel('Daily Return (%)', fontsize=12, color='white')
        ax.set_ylabel('Frequency', fontsize=12, color='white')
        ax.set_title('Returns Distribution', fontsize=14, fontweight='bold', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')
        ax.tick_params(colors='white')
        # Note: no _format_xaxis here — x-axis is return magnitude, not time

        return ax

    def plot_signals(self, results: pd.DataFrame, commodities: list,
                    ax: Optional[plt.Axes] = None):
        """Plot signals over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b

        for commodity in commodities:
            signal_col = f'{commodity}_signal'
            if signal_col in results.columns:
<<<<<<< HEAD
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results[signal_col],
                    name=commodity.capitalize(),
                    mode='lines',
                    line=dict(width=2),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + commodity + ': %{y:.3f}<extra></extra>',
                ))

        fig.add_hline(y=0, line_color=COLORS['text_secondary'], line_width=1)
        fig.add_hrect(y0=0, y1=3, fillcolor=COLORS['profit'], opacity=0.05, layer='below', line_width=0)
        fig.add_hrect(y0=-3, y1=0, fillcolor=COLORS['loss'], opacity=0.05, layer='below', line_width=0)

        layout = self._get_layout_template('Trading Signals')
        layout.update(height=300)
        fig.update_layout(**layout)
        return fig

    def plot_positions(self, results: pd.DataFrame, commodities: list) -> go.Figure:
        """Interactive positions chart."""
        fig = go.Figure()
=======
                ax.plot(results['date'], results[signal_col],
                       label=commodity.capitalize(), alpha=0.8, linewidth=1.5)

        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax.set_xlabel('Trading Day', fontsize=12, color='white')
        ax.set_ylabel('Signal Strength', fontsize=12, color='white')
        ax.set_title('Trading Signals', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        self._format_xaxis_as_trading_days(ax, results)

        return ax

    def plot_positions(self, results: pd.DataFrame, commodities: list,
                      ax: Optional[plt.Axes] = None):
        """Plot position sizes over time."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b

        for commodity in commodities:
            position_col = f'{commodity}_position'
            if position_col in results.columns:
<<<<<<< HEAD
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results[position_col],
                    name=commodity.capitalize(),
                    mode='lines',
                    line=dict(width=2),
                    fill='tozeroy',
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + commodity + ': %{y:.3f}<extra></extra>',
                ))

        fig.add_hline(y=0, line_color=COLORS['text_secondary'], line_width=1)

        layout = self._get_layout_template('Portfolio Positions')
        layout.update(height=300)
        fig.update_layout(**layout)
        return fig

    def plot_rolling_sharpe(self, results: pd.DataFrame, window: int = 60) -> go.Figure:
        """Interactive rolling Sharpe ratio."""
        returns = results['portfolio_return']
        rolling_sharpe = returns.rolling(window=window).mean() / returns.rolling(window=window).std() * np.sqrt(252)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=results['date'],
            y=rolling_sharpe,
            name=f'{window}-Day Rolling Sharpe',
            mode='lines',
            line=dict(color=COLORS['accent'], width=2.5),
            fill='tozeroy',
            fillcolor=f'rgba(217, 119, 87, 0.1)',
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Sharpe: %{y:.2f}<extra></extra>',
        ))

        fig.add_hline(y=0, line_color=COLORS['text_secondary'], line_width=1)
        fig.add_hline(y=1, line_dash='dash', line_color=COLORS['profit'],
                     annotation_text='Sharpe = 1.0', annotation_position='right')

        layout = self._get_layout_template(f'{window}-Day Rolling Sharpe Ratio')
        layout.update(height=300)
        fig.update_layout(**layout)
        return fig

    def create_full_dashboard(self, results: pd.DataFrame, commodities: list,
                            metrics: Optional[dict] = None) -> go.Figure:
        """Create comprehensive interactive dashboard."""

        # Create subplots
        fig = make_subplots(
            rows=6, cols=2,
            subplot_titles=(
                'Equity Curve', 'Metrics Summary',
                'Underwater Plot', '',
                'Returns Distribution', 'Rolling Sharpe',
                'Trading Signals', '',
                'Portfolio Positions', '',
                'Monthly Returns', '',
            ),
            specs=[
                [{"secondary_y": False}, {"type": "indicator"}],
                [{"secondary_y": False}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, None],
                [{"secondary_y": False}, None],
                [{"secondary_y": False}, None],
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12,
        )

        # Equity curve (full width)
        x = results['date']
        y = results['portfolio_value']
        initial = y.iloc[0]

        fig.add_trace(
            go.Scatter(x=x, y=y, name='Portfolio Value', line=dict(color=COLORS['accent'], width=2),
                      fill='tozeroy', fillcolor=f'rgba(217, 119, 87, 0.1)',
                      hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>'),
            row=1, col=1
        )

        # Metrics box
        if metrics:
            summary_text = (
                f"<b>Return:</b> {metrics.get('total_return', 0):.1%}<br>"
                f"<b>Sharpe:</b> {metrics.get('sharpe_ratio', 0):.2f}<br>"
                f"<b>Drawdown:</b> {metrics.get('max_drawdown', 0):.1%}<br>"
                f"<b>Win Rate:</b> {metrics.get('win_rate', 0):.1%}"
            )
            fig.add_trace(
                go.Indicator(mode='number+delta', value=metrics.get('total_return', 0)*100,
                            title={'text': 'Total Return (%)', 'font': {'size': 12}},
                            domain={'x': [0, 1], 'y': [0, 1]}),
                row=1, col=2
            )

        # Drawdown
        cumulative = results['portfolio_value']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100

        colors = [COLORS['loss'] if dd < -20 else COLORS['warning'] if dd < -10 else COLORS['neutral'] for dd in drawdown]

        fig.add_trace(
            go.Bar(x=x, y=drawdown, name='Drawdown', marker=dict(color=colors), showlegend=False,
                  hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}%<extra></extra>'),
            row=2, col=1
        )

        # Returns histogram
        returns = results['portfolio_return'].dropna() * 100
        fig.add_trace(
            go.Histogram(x=returns, name='Returns', nbinsx=50, marker=dict(color=COLORS['accent']),
                        showlegend=False, hovertemplate='Frequency: %{y}<extra></extra>'),
            row=3, col=1
        )

        # Rolling Sharpe
        rolling_sharpe = results['portfolio_return'].rolling(window=60).mean() / results['portfolio_return'].rolling(window=60).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=x, y=rolling_sharpe, name='Rolling Sharpe', line=dict(color=COLORS['accent'], width=2),
                      fill='tozeroy', fillcolor=f'rgba(217, 119, 87, 0.1)', showlegend=False,
                      hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:.2f}<extra></extra>'),
            row=3, col=2
        )

        # Signals
        for commodity in commodities:
            signal_col = f'{commodity}_signal'
            if signal_col in results.columns:
                fig.add_trace(
                    go.Scatter(x=x, y=results[signal_col], name=commodity.capitalize(), mode='lines',
                              hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + commodity + ': %{y:.3f}<extra></extra>'),
                    row=4, col=1
                )

        # Positions
        for commodity in commodities:
            position_col = f'{commodity}_position'
            if position_col in results.columns:
                fig.add_trace(
                    go.Scatter(x=x, y=results[position_col], name=commodity.capitalize(), mode='lines', fill='tozeroy',
                              hovertemplate='<b>%{x|%Y-%m-%d}</b><br>' + commodity + ': %{y:.3f}<extra></extra>'),
                    row=5, col=1
                )

        # Update layout
        fig.update_layout(
            title_text='<b>Backtest Performance Dashboard</b>',
            template='plotly_dark',
            plot_bgcolor=COLORS['bg_deep'],
            paper_bgcolor=COLORS['bg_primary'],
            hovermode='x unified',
            height=1600,
            font=dict(family='Arial, sans-serif', size=11, color=COLORS['text_primary']),
            showlegend=True,
            legend=dict(x=0.01, y=0.99, bgcolor=f'rgba(32, 32, 40, 0.8)', bordercolor=COLORS['border'], borderwidth=1),
        )

        # Update all axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['border'], showline=True, linewidth=1, linecolor=COLORS['border'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=COLORS['border'], showline=True, linewidth=1, linecolor=COLORS['border'])
=======
                ax.plot(results['date'], results[position_col],
                       label=commodity.capitalize(), alpha=0.8, linewidth=1.5)

        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax.set_xlabel('Trading Day', fontsize=12, color='white')
        ax.set_ylabel('Position Size (% of Portfolio)', fontsize=12, color='white')
        ax.set_title('Portfolio Positions', fontsize=14, fontweight='bold', color='white')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        self._format_xaxis_as_trading_days(ax, results)

        return ax

    def plot_rolling_sharpe(self, results: pd.DataFrame, window: int = 60,
                           ax: Optional[plt.Axes] = None):
        """Plot rolling Sharpe ratio."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))

        returns = results['portfolio_return']
        rolling_sharpe = (
            returns.rolling(window=window).mean()
            / returns.rolling(window=window).std()
            * np.sqrt(252)
        )

        ax.plot(results['date'], rolling_sharpe,
               color=self.colors['signal'], linewidth=2)
        ax.axhline(0, color='white', linestyle='-', alpha=0.3)
        ax.axhline(1, color='yellow', linestyle='--', alpha=0.5, label='Sharpe = 1')

        ax.set_xlabel('Trading Day', fontsize=12, color='white')
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12, color='white')
        ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=14, fontweight='bold', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.tick_params(colors='white')
        self._format_xaxis_as_trading_days(ax, results)

        return ax

    def create_full_report(self, results: pd.DataFrame, commodities: list,
                          save_path: Optional[Path] = None):
        """Create comprehensive backtest report with all plots."""
        fig = plt.figure(figsize=self.figsize)
        fig.patch.set_facecolor('#0a0a0a')

        gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3)

        ax1 = fig.add_subplot(gs[0:2, :])
        self.plot_equity_curve(results, ax=ax1)

        ax2 = fig.add_subplot(gs[2, :])
        self.plot_drawdown(results, ax=ax2)

        ax3 = fig.add_subplot(gs[3, 0])
        self.plot_returns_distribution(results, ax=ax3)

        ax4 = fig.add_subplot(gs[3, 1])
        self.plot_rolling_sharpe(results, ax=ax4)

        ax5 = fig.add_subplot(gs[4, :])
        self.plot_signals(results, commodities, ax=ax5)

        ax6 = fig.add_subplot(gs[5, :])
        self.plot_positions(results, commodities, ax=ax6)

        fig.suptitle('Backtest Performance Report',
                    fontsize=16, fontweight='bold', color='white', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#0a0a0a',
                       edgecolor='none', bbox_inches='tight')
            logger.info(f"Saved backtest visualization to {save_path}")
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b

        return fig


def plot_backtest_results(results: pd.DataFrame,
                         commodities: list,
                         save_path: Optional[str] = None,
                         trade_months: Optional[list] = None,
                         metrics: Optional[dict] = None):
    """
<<<<<<< HEAD
    Create interactive backtest dashboard.

    Args:
        results: Backtest results DataFrame
        commodities: List of commodities
        save_path: Path to save HTML
        trade_months: List of months to include
        metrics: Performance metrics dict
=======
    Convenience function to create full backtest visualization.
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
    """
    # Filter to trade months if specified
    results_filtered = results.copy()
    if trade_months:
        results_filtered = results[results['date'].dt.month.isin(trade_months)].copy()
        logger.info(f"Filtered to {len(results_filtered)} trading days in months {trade_months}")

    visualizer = BacktestVisualizer()
<<<<<<< HEAD
    fig = visualizer.create_full_dashboard(results_filtered, commodities, metrics=metrics)

    if save_path:
        # Save as HTML instead of static image for interactivity
        html_path = str(save_path).replace('.png', '.html')
        fig.write_html(html_path)
        logger.info(f"Saved interactive dashboard to {html_path}")

    fig.show()
    return fig
=======
    fig = visualizer.create_full_report(results, commodities, save_path)
    plt.show()
    return fig
>>>>>>> be3033f6e43f86a6455c13948f714e91c8606a8b
