"""
Streamlit web dashboard for REIT/InvIT Quantitative Cashflow & NAV Simulation.

This is the main entry point for the application. Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import project modules
from utils.data_loader import load_assets, validate_asset_data
from models.cashflow_simulator import CashflowSimulator
from models.risk_metrics import (
    compute_var,
    compute_cvar,
    compute_downside_risk,
    compute_sharpe_ratio
)
from models.portfolio_optimization import (
    calculate_portfolio_stats,
    mean_variance_optimization,
    generate_efficient_frontier
)
from utils.plotting import (
    plot_sample_paths,
    plot_distribution_histogram,
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_cashflow_comparison
)
from utils.config import DEFAULT_SIMULATION_PARAMS

# Page configuration
st.set_page_config(
    page_title="REIT/InvIT Quantitative Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 REIT/InvIT Quantitative Cashflow & NAV Simulation")
st.markdown("**Monte Carlo Simulation + Portfolio Optimization Tool**")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Parameters")

# Load asset data
@st.cache_data
def load_asset_data():
    """Load and cache asset data."""
    # Try multiple path resolution methods for Streamlit compatibility
    try:
        # Method 1: Use __file__ (works when running as script)
        data_path = Path(__file__).parent / "data" / "sample_assets.csv"
    except NameError:
        # Method 2: Use current working directory (works in Streamlit)
        data_path = Path("data") / "sample_assets.csv"
    
    # If file doesn't exist, try alternative paths
    if not data_path.exists():
        # Try relative to current working directory
        alt_path = Path.cwd() / "data" / "sample_assets.csv"
        if alt_path.exists():
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find sample_assets.csv. Tried: {data_path} and {alt_path}")
    
    df = load_assets(str(data_path))
    validate_asset_data(df)
    return df

try:
    assets_df = load_asset_data()
except Exception as e:
    st.error(f"Error loading asset data: {e}")
    st.error("Please ensure data/sample_assets.csv exists in the project directory.")
    st.stop()

# Simulation parameters
horizon_years = st.sidebar.slider(
    "Horizon (years)",
    min_value=1,
    max_value=20,
    value=DEFAULT_SIMULATION_PARAMS['horizon_years'],
    step=1
)

steps_per_year = st.sidebar.selectbox(
    "Steps per year",
    options=[1, 4, 12],
    index=2,  # Default to 12 (monthly)
    format_func=lambda x: {1: "Annual", 4: "Quarterly", 12: "Monthly"}[x]
)

n_sims = st.sidebar.slider(
    "Number of simulations",
    min_value=100,
    max_value=10000,
    value=DEFAULT_SIMULATION_PARAMS['n_sims'],
    step=100
)

var_confidence = st.sidebar.slider(
    "VaR confidence level",
    min_value=0.90,
    max_value=0.99,
    value=DEFAULT_SIMULATION_PARAMS['var_confidence'],
    step=0.01,
    format="%.2f"
)

risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (annual)",
    min_value=0.0,
    max_value=0.20,
    value=DEFAULT_SIMULATION_PARAMS['risk_free_rate'],
    step=0.001,
    format="%.3f"
)

# Asset selection
st.sidebar.header("📋 Asset Selection")
available_assets = assets_df['name'].tolist()
selected_assets = st.sidebar.multiselect(
    "Select assets to analyze",
    options=available_assets,
    default=available_assets[:5]  # Default to first 5
)

if not selected_assets:
    st.warning("Please select at least one asset from the sidebar.")
    st.stop()

# Filter assets DataFrame
selected_asset_ids = assets_df[assets_df['name'].isin(selected_assets)]['asset_id'].tolist()
filtered_assets_df = assets_df[assets_df['asset_id'].isin(selected_asset_ids)]

# Portfolio optimization options
st.sidebar.header("🎯 Portfolio Options")
enable_optimization = st.sidebar.checkbox("Enable Portfolio Optimization", value=False)

target_return = None
if enable_optimization:
    target_return = st.sidebar.number_input(
        "Target return (annual, optional)",
        min_value=0.0,
        max_value=0.50,
        value=None,
        step=0.001,
        format="%.3f",
        help="Leave empty for maximum Sharpe ratio"
    )
    if target_return == 0.0:
        target_return = None

# Run simulation
@st.cache_data
def run_simulation(_assets_df, horizon_years, steps_per_year, n_sims, random_state=42):
    """Run simulation and cache results."""
    simulator = CashflowSimulator(
        assets_df=_assets_df,
        horizon_years=horizon_years,
        steps_per_year=steps_per_year,
        n_sims=n_sims,
        random_state=random_state
    )
    return simulator.simulate_all_assets()

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "🔍 Single Asset Analysis",
    "💼 Portfolio Simulation",
    "⚖️ Portfolio Optimization"
])

# Tab 1: Overview
with tab1:
    st.header("Asset Overview")
    
    # Display summary table
    display_cols = [
        'name', 'asset_type', 'current_nav', 'base_rent_or_tariff',
        'current_occupancy_or_utilization', 'rent_tariff_growth_mu',
        'rent_tariff_growth_sigma', 'payout_ratio', 'debt_ratio', 'beta'
    ]
    
    summary_df = filtered_assets_df[display_cols].copy()
    summary_df.columns = [
        'Name', 'Type', 'Current NAV', 'Base Rent/Tariff',
        'Occupancy/Utilization', 'Growth Rate (μ)',
        'Growth Volatility (σ)', 'Payout Ratio', 'Debt Ratio', 'Beta'
    ]
    
    # Format numbers
    summary_df['Current NAV'] = summary_df['Current NAV'].apply(lambda x: f"${x:,.0f}")
    summary_df['Base Rent/Tariff'] = summary_df['Base Rent/Tariff'].apply(lambda x: f"${x:.2f}")
    summary_df['Occupancy/Utilization'] = summary_df['Occupancy/Utilization'].apply(lambda x: f"{x:.1%}")
    summary_df['Growth Rate (μ)'] = summary_df['Growth Rate (μ)'].apply(lambda x: f"{x:.2%}")
    summary_df['Growth Volatility (σ)'] = summary_df['Growth Volatility (σ)'].apply(lambda x: f"{x:.2%}")
    summary_df['Payout Ratio'] = summary_df['Payout Ratio'].apply(lambda x: f"{x:.1%}")
    summary_df['Debt Ratio'] = summary_df['Debt Ratio'].apply(lambda x: f"{x:.1%}")
    summary_df['Beta'] = summary_df['Beta'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(summary_df, width='stretch', hide_index=True)
    
    # Key statistics
    st.subheader("Portfolio Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    total_nav = filtered_assets_df['current_nav'].sum()
    avg_occupancy = filtered_assets_df['current_occupancy_or_utilization'].mean()
    avg_growth = filtered_assets_df['rent_tariff_growth_mu'].mean()
    avg_payout = filtered_assets_df['payout_ratio'].mean()
    
    col1.metric("Total NAV", f"${total_nav:,.0f}")
    col2.metric("Avg Occupancy", f"{avg_occupancy:.1%}")
    col3.metric("Avg Growth Rate", f"{avg_growth:.2%}")
    col4.metric("Avg Payout Ratio", f"{avg_payout:.1%}")

# Tab 2: Single Asset Analysis
with tab2:
    st.header("Single Asset Deep Dive")
    
    # Select asset
    selected_asset_name = st.selectbox(
        "Select asset to analyze",
        options=selected_assets,
        key="single_asset_select"
    )
    
    selected_asset_id = assets_df[assets_df['name'] == selected_asset_name]['asset_id'].iloc[0]
    
    # Run simulation for selected asset
    with st.spinner("Running simulation..."):
        simulation_results = run_simulation(
            filtered_assets_df[filtered_assets_df['asset_id'] == selected_asset_id],
            horizon_years,
            steps_per_year,
            n_sims
        )
    
    asset_result = simulation_results[selected_asset_id]
    stats = asset_result['summary_stats']
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{stats['annualized_return']:.2%}")
    col2.metric("Avg Annual Yield", f"{stats['avg_annual_yield']:.2%}")
    col3.metric("Final NAV (Mean)", f"${stats['final_nav_mean']:,.0f}")
    col4.metric("IRR (Approx)", f"{stats['irr_approx']:.2%}")
    
    # NAV paths
    st.subheader("NAV Evolution Paths")
    nav_paths = asset_result['nav_paths']
    fig_nav = plot_sample_paths(
        nav_paths,
        title=f"NAV Simulation Paths - {selected_asset_name}",
        y_label="NAV ($)"
    )
    st.plotly_chart(fig_nav, width='stretch')
    
    # Distribution paths
    st.subheader("Distribution/Cashflow Paths")
    dist_paths = asset_result['distribution_paths']
    fig_dist = plot_sample_paths(
        dist_paths,
        title=f"Distribution Paths - {selected_asset_name}",
        y_label="Distribution ($)"
    )
    st.plotly_chart(fig_dist, width='stretch')
    
    # Ending NAV distribution
    st.subheader("Ending NAV Distribution")
    final_nav = nav_paths[:, -1]
    initial_nav = nav_paths[:, 0]
    nav_returns = (final_nav - initial_nav) / initial_nav
    
    # Calculate risk metrics
    var_nav = compute_var(nav_returns, var_confidence)
    cvar_nav = compute_cvar(nav_returns, var_confidence)
    downside_risk_nav = compute_downside_risk(nav_returns)
    sharpe_nav = compute_sharpe_ratio(nav_returns, risk_free_rate, steps_per_year)
    
    fig_hist = plot_distribution_histogram(
        final_nav,
        title=f"Final NAV Distribution - {selected_asset_name}",
        x_label="Final NAV ($)"
    )
    st.plotly_chart(fig_hist, width='stretch')
    
    # Risk metrics table
    st.subheader("Risk Metrics")
    risk_metrics_df = pd.DataFrame({
        'Metric': ['VaR', 'CVaR', 'Downside Risk', 'Sharpe Ratio'],
        'Value': [
            f"{var_nav:.2%}",
            f"{cvar_nav:.2%}",
            f"{downside_risk_nav:.2%}",
            f"{sharpe_nav:.2f}"
        ]
    })
    st.dataframe(risk_metrics_df, width='stretch', hide_index=True)

# Tab 3: Portfolio Simulation
with tab3:
    st.header("Portfolio-Level Analysis")
    
    # Run simulation for all selected assets
    with st.spinner("Running portfolio simulation..."):
        portfolio_results = run_simulation(
            filtered_assets_df,
            horizon_years,
            steps_per_year,
            n_sims
        )
    
    # Get asset IDs from the results (to ensure they match)
    result_asset_ids = list(portfolio_results.keys())
    n_assets = len(result_asset_ids)
    
    if n_assets == 0:
        st.error("No simulation results available. Please check asset selection.")
        st.stop()
    
    equal_weight = 1.0 / n_assets
    
    # Aggregate portfolio NAV
    portfolio_nav_paths = np.zeros((n_sims, horizon_years * steps_per_year + 1))
    portfolio_dist_paths = np.zeros((n_sims, horizon_years * steps_per_year))
    
    for asset_id in result_asset_ids:
        if asset_id in portfolio_results:
            portfolio_nav_paths += portfolio_results[asset_id]['nav_paths'] * equal_weight
            portfolio_dist_paths += portfolio_results[asset_id]['distribution_paths'] * equal_weight
    
    # Portfolio NAV evolution
    st.subheader("Portfolio NAV Evolution")
    fig_port_nav = plot_sample_paths(
        portfolio_nav_paths,
        title="Portfolio NAV Paths (Equal-Weighted)",
        y_label="Portfolio NAV ($)"
    )
    st.plotly_chart(fig_port_nav, width='stretch')
    
    # Portfolio cashflow comparison
    st.subheader("Asset Cashflow Comparison")
    cashflow_dict = {
        assets_df[assets_df['asset_id'] == aid]['name'].iloc[0]: 
        portfolio_results[aid]['distribution_paths']
        for aid in result_asset_ids
        if aid in portfolio_results
    }
    fig_cf = plot_cashflow_comparison(
        cashflow_dict,
        title="Mean Distribution Paths by Asset"
    )
    st.plotly_chart(fig_cf, width='stretch')
    
    # Portfolio risk metrics
    st.subheader("Portfolio Risk Metrics")
    portfolio_final_nav = portfolio_nav_paths[:, -1]
    portfolio_initial_nav = portfolio_nav_paths[:, 0]
    portfolio_returns = (portfolio_final_nav - portfolio_initial_nav) / portfolio_initial_nav
    
    port_var = compute_var(portfolio_returns, var_confidence)
    port_cvar = compute_cvar(portfolio_returns, var_confidence)
    port_downside = compute_downside_risk(portfolio_returns)
    port_sharpe = compute_sharpe_ratio(portfolio_returns, risk_free_rate, steps_per_year)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_port_hist = plot_distribution_histogram(
            portfolio_final_nav,
            title="Portfolio Final NAV Distribution",
            var_line=portfolio_initial_nav[0] * (1 + port_var),
            cvar_line=portfolio_initial_nav[0] * (1 + port_cvar),
            x_label="Final Portfolio NAV ($)"
        )
        st.plotly_chart(fig_port_hist, width='stretch')
    
    with col2:
        port_risk_df = pd.DataFrame({
            'Metric': ['VaR', 'CVaR', 'Downside Risk', 'Sharpe Ratio'],
            'Value': [
                f"{port_var:.2%}",
                f"{port_cvar:.2%}",
                f"{port_downside:.2%}",
                f"{port_sharpe:.2f}"
            ]
        })
        st.dataframe(port_risk_df, width='stretch', hide_index=True)
        
        # Portfolio summary stats
        avg_annual_dist = np.mean(np.sum(portfolio_dist_paths, axis=1)) / horizon_years
        st.metric("Avg Annual Distribution", f"${avg_annual_dist:,.0f}")

# Tab 4: Portfolio Optimization
with tab4:
    st.header("Portfolio Optimization")
    
    if not enable_optimization:
        st.info("💡 Enable Portfolio Optimization in the sidebar to use this feature.")
    else:
        # Check if we have at least 2 assets for optimization
        if len(selected_asset_ids) < 2:
            st.warning("⚠️ Portfolio optimization requires at least 2 assets. Please select more assets from the sidebar.")
        else:
            # Run simulation for optimization
            with st.spinner("Calculating optimal portfolio..."):
                opt_portfolio_results = run_simulation(
                    filtered_assets_df,
                    horizon_years,
                    steps_per_year,
                    n_sims
                )
                
                # Get asset IDs from results
                opt_asset_ids = list(opt_portfolio_results.keys())
                
                if len(opt_asset_ids) < 2:
                    st.error("Portfolio optimization requires at least 2 assets with simulation results.")
                else:
                    # Calculate portfolio statistics
                    expected_returns, cov_matrix = calculate_portfolio_stats(opt_portfolio_results)
                    
                    asset_names = [
                        assets_df[assets_df['asset_id'] == aid]['name'].iloc[0]
                        for aid in opt_asset_ids
                        if aid in assets_df['asset_id'].values
                    ]
                    
                    # Generate efficient frontier
                    frontier_df = generate_efficient_frontier(
                        expected_returns,
                        cov_matrix,
                        n_points=50,
                        risk_free_rate=risk_free_rate
                    )
                    
                    # Find optimal portfolio
                    optimal_portfolio = mean_variance_optimization(
                        expected_returns,
                        cov_matrix,
                        target_return=target_return,
                        risk_free_rate=risk_free_rate
                    )
                    
                    # Display efficient frontier
                    st.subheader("Efficient Frontier")
                    fig_frontier = plot_efficient_frontier(
                        frontier_df,
                        optimal_portfolio=optimal_portfolio,
                        asset_names=asset_names
                    )
                    st.plotly_chart(fig_frontier, width='stretch')
                    
                    # Optimal portfolio weights
                    st.subheader("Optimal Portfolio Weights")
                    fig_weights = plot_portfolio_weights(
                        optimal_portfolio['weights'],
                        asset_names,
                        title="Optimal Portfolio Allocation"
                    )
                    st.plotly_chart(fig_weights, width='stretch')
                    
                    # Optimal portfolio metrics
                    st.subheader("Optimal Portfolio Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Expected Return", f"{optimal_portfolio['expected_return']:.2%}")
                    col2.metric("Risk (Std Dev)", f"{optimal_portfolio['risk']:.2%}")
                    col3.metric("Sharpe Ratio", f"{optimal_portfolio['sharpe_ratio']:.2f}")
                    
                    # Calculate VaR/CVaR for optimal portfolio
                    # Approximate using portfolio return distribution
                    optimal_returns = np.zeros(n_sims)
                    for i, asset_id in enumerate(opt_asset_ids):
                        if asset_id in opt_portfolio_results:
                            nav_paths = opt_portfolio_results[asset_id]['nav_paths']
                            initial_nav = nav_paths[:, 0]
                            final_nav = nav_paths[:, -1]
                            asset_returns = (final_nav - initial_nav) / initial_nav
                            optimal_returns += asset_returns * optimal_portfolio['weights'][i]
                    
                    opt_var = compute_var(optimal_returns, var_confidence)
                    opt_cvar = compute_cvar(optimal_returns, var_confidence)
                    col4.metric("VaR", f"{opt_var:.2%}")
                    
                    # Detailed weights table
                    weights_df = pd.DataFrame({
                        'Asset': asset_names,
                        'Weight': optimal_portfolio['weights'],
                        'Expected Return': expected_returns
                    })
                    weights_df = weights_df.sort_values('Weight', ascending=False)
                    weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
                    weights_df['Expected Return'] = weights_df['Expected Return'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(weights_df, width='stretch', hide_index=True)
                    
                    # Additional risk metrics
                    st.subheader("Optimal Portfolio Risk Metrics")
                    opt_risk_df = pd.DataFrame({
                        'Metric': ['VaR', 'CVaR'],
                        'Value': [f"{opt_var:.2%}", f"{opt_cvar:.2%}"]
                    })
                    st.dataframe(opt_risk_df, width='stretch', hide_index=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**REIT/InvIT Quantitative Analysis Tool**")
st.sidebar.markdown("Built with Streamlit, NumPy, and Plotly")

