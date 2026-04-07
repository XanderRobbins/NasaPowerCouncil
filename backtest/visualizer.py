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
                'Daily P&L', '',
            ),
            specs=[
                [{"secondary_y": False}, {"type": "table"}],
                [{"secondary_y": False}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, None],
                [{"secondary_y": False}, None],
                [{"secondary_y": False}, None],
            ],
            vertical_spacing=0.07,
            horizontal_spacing=0.12,
        )

        # Equity curve (full width)
        x = results['date']
        y = results['portfolio_value']
        initial = y.iloc[0]

        fig.add_trace(
            go.Scatter(x=x, y=y, name='Portfolio Value', line=dict(color=COLORS['accent'], width=2.5),
                      fill='tozeroy', fillcolor=f'rgba(217, 119, 87, 0.1)',
                      hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>'),
            row=1, col=1
        )

        # Metrics table
        if metrics:
            metric_rows = [
                ['Total Return', f"{metrics.get('total_return', 0):.1%}"],
                ['Annual Return', f"{metrics.get('annualized_return', 0):.2%}"],
                ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"],
                ['Max Drawdown', f"{metrics.get('max_drawdown', 0):.1%}"],
                ['Win Rate', f"{metrics.get('win_rate', 0):.1%}"],
                ['Profit Factor', f"{metrics.get('profit_factor', 0):.2f}"],
            ]
            fig.add_trace(
                go.Table(
                    header=dict(values=['<b>Metric</b>', '<b>Value</b>'],
                               fill_color=COLORS['accent'],
                               font=dict(color=COLORS['text_primary'], size=11),
                               align='left'),
                    cells=dict(values=list(zip(*metric_rows)),
                              fill_color=COLORS['bg_deep'],
                              font=dict(color=COLORS['text_primary'], size=10),
                              align='left',
                              height=25),
                    name='Metrics',
                ),
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
            go.Scatter(x=x, y=rolling_sharpe, name='Rolling Sharpe', line=dict(color=COLORS['accent'], width=2.5),
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
                              line=dict(width=2),
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

        # Daily P&L
        fig.add_trace(
            go.Bar(x=x, y=results['daily_pnl'], name='Daily P&L',
                  marker=dict(color=[COLORS['profit'] if p > 0 else COLORS['loss'] for p in results['daily_pnl']]),
                  showlegend=False,
                  hovertemplate='<b>%{x|%Y-%m-%d}</b><br>$%{y:,.0f}<extra></extra>'),
            row=6, col=1
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

        return fig


def plot_backtest_results(results: pd.DataFrame,
                         commodities: list,
                         save_path: Optional[str] = None,
                         trade_months: Optional[list] = None,
                         metrics: Optional[dict] = None):
    """
    Create interactive backtest dashboard.

    Args:
        results: Backtest results DataFrame
        commodities: List of commodities
        save_path: Path to save HTML
        trade_months: List of months to include
        metrics: Performance metrics dict
    """
    # Filter to trade months if specified
    results_filtered = results.copy()
    if trade_months:
        results_filtered = results[results['date'].dt.month.isin(trade_months)].copy()
        logger.info(f"Filtered to {len(results_filtered)} trading days in months {trade_months}")

    visualizer = BacktestVisualizer()
    fig = visualizer.create_full_dashboard(results_filtered, commodities, metrics=metrics)

    if save_path:
        # Save as HTML for interactivity
        html_path = str(save_path).replace('.png', '.html')
        fig.write_html(html_path)
        logger.info(f"Saved interactive dashboard to {html_path}")

    fig.show()
    return fig
