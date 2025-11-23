"""
Plotly visualization functions for the REIT/InvIT dashboard.

All functions return plotly.graph_objects.Figure objects that can be
directly displayed in Streamlit using st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict


def plot_sample_paths(
    paths: np.ndarray,
    title: str,
    n_samples: int = 10,
    x_label: str = "Time Step",
    y_label: str = "Value"
) -> go.Figure:
    """
    Plot sample simulation paths with mean path highlighted.
    
    Args:
        paths: Array of shape (n_sims, time_steps) containing simulation paths
        title: Plot title
        n_samples: Number of individual paths to display (for clarity)
        x_label: X-axis label
        y_label: Y-axis label
        
    Returns:
        plotly.graph_objects.Figure
    """
    n_sims, time_steps = paths.shape
    time_axis = np.arange(time_steps)
    
    fig = go.Figure()
    
    # Plot sample individual paths (light, semi-transparent)
    n_display = min(n_samples, n_sims)
    sample_indices = np.linspace(0, n_sims - 1, n_display, dtype=int)
    
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=paths[idx, :],
            mode='lines',
            name=f'Path {idx+1}',
            line=dict(color='lightblue', width=1),
            opacity=0.3,
            showlegend=False
        ))
    
    # Plot mean path (bold, prominent)
    mean_path = np.mean(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color='darkblue', width=3),
        showlegend=True
    ))
    
    # Add confidence bands (optional: 5th and 95th percentiles)
    percentile_5 = np.percentile(paths, 5, axis=0)
    percentile_95 = np.percentile(paths, 95, axis=0)
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentile_95,
        mode='lines',
        name='95th Percentile',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=percentile_5,
        mode='lines',
        name='5th Percentile',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,100,200,0.1)',
        showlegend=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_distribution_histogram(
    values: np.ndarray,
    title: str,
    var_line: Optional[float] = None,
    cvar_line: Optional[float] = None,
    x_label: str = "Value"
) -> go.Figure:
    """
    Plot histogram of distribution with optional VaR and CVaR markers.
    
    Args:
        values: Array of values to plot
        title: Plot title
        var_line: Optional VaR value to mark on plot
        cvar_line: Optional CVaR value to mark on plot
        x_label: X-axis label
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=50,
        name='Distribution',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    # Add VaR line if provided
    if var_line is not None:
        fig.add_vline(
            x=var_line,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR: {var_line:.2f}",
            annotation_position="top"
        )
    
    # Add CVaR line if provided
    if cvar_line is not None:
        fig.add_vline(
            x=cvar_line,
            line_dash="dot",
            line_color="darkred",
            annotation_text=f"CVaR: {cvar_line:.2f}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Frequency",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    optimal_portfolio: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None
) -> go.Figure:
    """
    Plot efficient frontier with optional optimal portfolio highlighted.
    
    Args:
        frontier_df: DataFrame with columns 'return', 'risk', 'sharpe_ratio'
        optimal_portfolio: Optional dict with 'expected_return', 'risk', 'sharpe_ratio'
        asset_names: Optional list of asset names for individual asset points
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Plot efficient frontier
    fig.add_trace(go.Scatter(
        x=frontier_df['risk'],
        y=frontier_df['return'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue', width=2),
        hovertemplate='Risk: %{x:.4f}<br>Return: %{y:.4f}<extra></extra>'
    ))
    
    # Highlight optimal portfolio if provided
    if optimal_portfolio is not None:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['risk']],
            y=[optimal_portfolio['expected_return']],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            hovertemplate=(
                f"Risk: {optimal_portfolio['risk']:.4f}<br>"
                f"Return: {optimal_portfolio['expected_return']:.4f}<br>"
                f"Sharpe: {optimal_portfolio.get('sharpe_ratio', 0):.4f}"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Risk (Standard Deviation)',
        yaxis_title='Expected Return',
        template='plotly_white',
        height=500,
        hovermode='closest'
    )
    
    return fig


def plot_portfolio_weights(
    weights: np.ndarray,
    asset_names: List[str],
    title: str = "Portfolio Weights"
) -> go.Figure:
    """
    Plot portfolio weights as bar chart.
    
    Args:
        weights: Array of portfolio weights (should sum to ~1.0)
        asset_names: List of asset names corresponding to weights
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Sort by weight for better visualization
    sorted_indices = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_indices]
    sorted_names = [asset_names[i] for i in sorted_indices]
    
    # Color scale: darker for higher weights
    colors = ['steelblue' if w > 0.01 else 'lightgray' for w in sorted_weights]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sorted_names,
        y=sorted_weights,
        marker_color=colors,
        text=[f'{w:.1%}' for w in sorted_weights],
        textposition='outside',
        hovertemplate='%{x}<br>Weight: %{y:.2%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Asset',
        yaxis_title='Weight',
        yaxis=dict(tickformat='.0%'),
        template='plotly_white',
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def plot_cashflow_comparison(
    cashflow_paths: Dict[str, np.ndarray],
    title: str = "Cashflow Comparison"
) -> go.Figure:
    """
    Plot comparison of cashflow paths across multiple assets.
    
    Args:
        cashflow_paths: Dictionary mapping asset names to cashflow arrays
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    for asset_name, paths in cashflow_paths.items():
        mean_path = np.mean(paths, axis=0)
        time_axis = np.arange(len(mean_path))
        
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=mean_path,
            mode='lines',
            name=asset_name,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time Step',
        yaxis_title='Cashflow',
        template='plotly_white',
        height=500,
        hovermode='x unified'
    )
    
    return fig

