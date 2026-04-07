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

        for commodity in commodities:
            signal_col = f'{commodity}_signal'
            if signal_col in results.columns:
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

        for commodity in commodities:
            position_col = f'{commodity}_position'
            if position_col in results.columns:
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
        # Save as HTML instead of static image for interactivity
        html_path = str(save_path).replace('.png', '.html')
        fig.write_html(html_path)
        logger.info(f"Saved interactive dashboard to {html_path}")

    fig.show()
    return fig
