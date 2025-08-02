""
Visualization utilities for Brent oil price analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set default style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)


def plot_price_series(
    prices: pd.Series,
    change_points: Optional[List[Dict]] = None,
    events: Optional[pd.DataFrame] = None,
    title: str = 'Brent Crude Oil Price History',
    figsize: Tuple[int, int] = (14, 7)
) -> plt.Figure:
    """
    Plot the price series with optional change points and events.
    
    Args:
        prices: Pandas Series with datetime index and price values
        change_points: List of change point dictionaries with 'time_index' and 'probability'
        events: DataFrame with columns ['event_date', 'event_name', 'event_type']
        title: Plot title
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price series
    ax.plot(prices.index, prices.values, label='Price (USD/barrel)', linewidth=1.5)
    
    # Add change points
    if change_points:
        for cp in change_points:
            if 'time_index' in cp and cp['time_index'] < len(prices):
                cp_date = prices.index[cp['time_index']]
                cp_prob = cp.get('probability', 1.0)
                ax.axvline(
                    x=cp_date, 
                    color='red', 
                    linestyle='--', 
                    alpha=0.7,
                    label=f"Change Point (p={cp_prob:.2f})" if cp == change_points[0] else ""
                )
    
    # Add events
    if events is not None and not events.empty:
        event_types = events['event_type'].unique()
        colors = plt.cm.tab10(range(len(event_types)))
        
        for i, event_type in enumerate(event_types):
            type_events = events[events['event_type'] == event_type]
            for _, event in type_events.iterrows():
                event_date = pd.to_datetime(event['event_date'])
                if event_date >= prices.index[0] and event_date <= prices.index[-1]:
                    ax.axvline(
                        x=event_date,
                        color=colors[i],
                        linestyle=':',
                        alpha=0.7,
                        label=f"{event_type} Event" if i == 0 else ""
                    )
    
    # Customize plot
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD/barrel)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_interactive_price_series(
    prices: pd.Series,
    change_points: Optional[List[Dict]] = None,
    events: Optional[pd.DataFrame] = None,
    title: str = 'Brent Crude Oil Price History',
    height: int = 600
) -> go.Figure:
    """
    Create an interactive price series plot with Plotly.
    
    Args:
        prices: Pandas Series with datetime index and price values
        change_points: List of change point dictionaries
        events: DataFrame with event information
        title: Plot title
        height: Figure height in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add price series
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode='lines',
            name='Price (USD/barrel)',
            line=dict(width=1.5, color='#1f77b4'),
            hovertemplate='%{x|%b %d, %Y}<br>Price: %{y:.2f} USD/barrel<extra></extra>'
        )
    )
    
    # Add change points
    if change_points:
        for cp in change_points:
            if 'time_index' in cp and cp['time_index'] < len(prices):
                cp_date = prices.index[cp['time_index']]
                cp_prob = cp.get('probability', 1.0)
                
                fig.add_vline(
                    x=cp_date,
                    line=dict(color='red', dash='dash', width=1.5),
                    opacity=0.7,
                    annotation=dict(
                        text=f"Change Point (p={cp_prob:.2f})",
                        showarrow=True,
                        arrowhead=1,
                        ax=0,
                        ay=-40
                    )
                )
    
    # Add events
    if events is not None and not events.empty:
        event_types = events['event_type'].unique()
        colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, event_type in enumerate(event_types):
            type_events = events[events['event_type'] == event_type]
            color = colors[i % len(colors)]
            
            for _, event in type_events.iterrows():
                event_date = pd.to_datetime(event['event_date'])
                if event_date >= prices.index[0] and event_date <= prices.index[-1]:
                    fig.add_vline(
                        x=event_date,
                        line=dict(color=color, dash='dot', width=1.5),
                        opacity=0.7,
                        annotation=dict(
                            text=event['event_name'],
                            textangle=-90,
                            y=1.1,
                            yref='paper',
                            showarrow=False,
                            font=dict(size=10, color=color)
                        )
                    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        xaxis_title='Date',
        yaxis_title='Price (USD/barrel)',
        hovermode='x unified',
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    
    return fig


def plot_segment_statistics(segments: pd.DataFrame) -> go.Figure:
    """
    Plot statistics for each segment between change points.
    
    Args:
        segments: DataFrame with segment statistics
        
    Returns:
        Plotly Figure object
    """
    if segments.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Segment Mean Price', 'Segment Volatility'),
        vertical_spacing=0.15
    )
    
    # Add mean price
    fig.add_trace(
        go.Bar(
            x=segments.index,
            y=segments['mean'],
            name='Mean Price',
            marker_color='#1f77b4',
            hovertemplate='Segment %{x}<br>Mean: %{y:.2f} USD/barrel<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add volatility (standard deviation)
    fig.add_trace(
        go.Bar(
            x=segments.index,
            y=segments['std'],
            name='Volatility',
            marker_color='#ff7f0e',
            hovertemplate='Segment %{x}<br>Volatility: %{y:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Segment Statistics',
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        height=700,
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update y-axes
    fig.update_yaxes(title_text='Price (USD/barrel)', row=1, col=1)
    fig.update_yaxes(title_text='Standard Deviation', row=2, col=1)
    fig.update_xaxes(title_text='Segment', row=2, col=1)
    
    return fig


def plot_posterior_distributions(trace, var_names=None):
    """
    Plot posterior distributions of model parameters.
    
    Args:
        trace: PyMC3 trace or ArviZ InferenceData
        var_names: List of variable names to plot (if None, plot all)
        
    Returns:
        Matplotlib Figure object
    """
    import arviz as az
    
    # Use ArviZ for nice plotting
    az_style = {
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    }
    
    with az.rc_context(az_style):
        axes = az.plot_posterior(
            trace,
            var_names=var_names,
            textsize=10,
            hdi_prob=0.95,
            figsize=(14, 3 * len(var_names) if var_names else 10)
        )
    
    fig = axes.ravel()[0].figure
    plt.tight_layout()
    return fig
