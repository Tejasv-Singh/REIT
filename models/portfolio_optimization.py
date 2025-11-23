"""
Portfolio optimization using Markowitz mean-variance framework.

This module implements:
- Portfolio statistics calculation (expected returns, covariance)
- Mean-variance optimization
- Efficient frontier generation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    from scipy.optimize import minimize


def calculate_portfolio_stats(simulation_results: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract expected returns and covariance matrix from simulation results.
    
    This function aggregates individual asset simulations to compute:
    - Expected returns: Mean annualized return for each asset
    - Covariance matrix: Covariance of returns across assets
    
    Args:
        simulation_results: Dictionary mapping asset_id to simulation results
            Each result should have 'summary_stats' with 'annualized_return'
            and 'nav_paths' for covariance calculation
            
    Returns:
        Tuple of (expected_returns, cov_matrix)
        - expected_returns: Array of shape (n_assets,)
        - cov_matrix: Array of shape (n_assets, n_assets)
    """
    asset_ids = list(simulation_results.keys())
    n_assets = len(asset_ids)
    
    if n_assets == 0:
        raise ValueError("No assets in simulation results")
    
    # Extract expected returns from summary stats
    expected_returns = np.array([
        simulation_results[aid]['summary_stats']['annualized_return']
        for aid in asset_ids
    ])
    
    # Handle single asset case
    if n_assets == 1:
        # For single asset, return a 1x1 covariance matrix (variance of that asset)
        nav_paths = simulation_results[asset_ids[0]]['nav_paths']
        initial_nav = nav_paths[:, 0]
        final_nav = nav_paths[:, -1]
        total_returns = (final_nav - initial_nav) / initial_nav
        variance = np.var(total_returns)
        cov_matrix = np.array([[variance]])
    else:
        # Calculate covariance matrix from NAV paths
        # Use final NAV returns for covariance
        nav_returns_list = []
        for asset_id in asset_ids:
            nav_paths = simulation_results[asset_id]['nav_paths']
            initial_nav = nav_paths[:, 0]
            final_nav = nav_paths[:, -1]
            # Total return over simulation period
            total_returns = (final_nav - initial_nav) / initial_nav
            nav_returns_list.append(total_returns)
        
        # Stack into matrix: (n_assets, n_sims)
        returns_matrix = np.array(nav_returns_list)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_matrix)
        
        # Ensure it's 2D (handle edge case where np.cov might return 1D)
        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])
        elif cov_matrix.ndim == 1:
            cov_matrix = cov_matrix.reshape(1, -1) if n_assets == 1 else cov_matrix.reshape(-1, 1)
        
        # Ensure positive semi-definite (add small regularization if needed)
        if cov_matrix.ndim >= 2 and cov_matrix.shape[0] > 0:
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals < -1e-10):  # Negative eigenvalues indicate numerical issues
                # Add small regularization
                cov_matrix += np.eye(n_assets) * 1e-8
    
    return expected_returns, cov_matrix


def mean_variance_optimization(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Solve Markowitz mean-variance optimization problem.
    
    If target_return is None, solves for maximum Sharpe ratio portfolio.
    Otherwise, minimizes portfolio variance subject to target return.
    
    Optimization problem:
        minimize: w^T Σ w  (portfolio variance)
        subject to:
            - sum(w) = 1  (fully invested)
            - w >= 0  (no short selling)
            - [optional] w^T μ >= target_return
    
    Args:
        expected_returns: Array of expected returns for each asset
        cov_matrix: Covariance matrix of returns
        target_return: Optional target return constraint
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        
    Returns:
        Dictionary with:
        - weights: Optimal portfolio weights (array)
        - expected_return: Portfolio expected return
        - risk: Portfolio standard deviation
        - sharpe_ratio: Sharpe ratio of portfolio
    """
    n_assets = len(expected_returns)
    
    if CVXPY_AVAILABLE:
        return _optimize_with_cvxpy(
            expected_returns, cov_matrix, target_return, risk_free_rate
        )
    else:
        return _optimize_with_scipy(
            expected_returns, cov_matrix, target_return, risk_free_rate
        )


def _optimize_with_cvxpy(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: Optional[float],
    risk_free_rate: float
) -> Dict:
    """Optimize using cvxpy (preferred method)."""
    n_assets = len(expected_returns)
    
    # Decision variable: portfolio weights
    w = cp.Variable(n_assets)
    
    # Portfolio return and risk
    portfolio_return = expected_returns @ w
    portfolio_risk = cp.quad_form(w, cov_matrix)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Fully invested
        w >= 0  # No short selling
    ]
    
    if target_return is not None:
        constraints.append(portfolio_return >= target_return)
    
    # Objective: minimize variance (or maximize Sharpe if no target)
    if target_return is None:
        # Maximize Sharpe ratio = (return - rf) / risk
        # Equivalent to minimizing -Sharpe or maximizing (return - rf) / sqrt(risk)
        objective = cp.Maximize((portfolio_return - risk_free_rate) / cp.sqrt(portfolio_risk))
    else:
        # Minimize variance
        objective = cp.Minimize(portfolio_risk)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Optimization failed with status: {problem.status}")
    
    weights = w.value
    expected_return = float(portfolio_return.value)
    risk = float(np.sqrt(portfolio_risk.value))
    sharpe = (expected_return - risk_free_rate) / risk if risk > 0 else 0.0
    
    return {
        'weights': weights,
        'expected_return': expected_return,
        'risk': risk,
        'sharpe_ratio': sharpe
    }


def _optimize_with_scipy(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: Optional[float],
    risk_free_rate: float
) -> Dict:
    """Optimize using scipy.optimize (fallback method)."""
    n_assets = len(expected_returns)
    
    # Objective function: portfolio variance
    def objective(w):
        return w @ cov_matrix @ w
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
    ]
    
    if target_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: expected_returns @ w - target_return
        })
    
    # Bounds: no short selling
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Initial guess: equal weights
    x0 = np.ones(n_assets) / n_assets
    
    # Solve
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    
    weights = result.x
    expected_return = expected_returns @ weights
    risk = np.sqrt(objective(weights))
    sharpe = (expected_return - risk_free_rate) / risk if risk > 0 else 0.0
    
    return {
        'weights': weights,
        'expected_return': expected_return,
        'risk': risk,
        'sharpe_ratio': sharpe
    }


def generate_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    risk_free_rate: float = 0.0
) -> pd.DataFrame:
    """
    Generate efficient frontier points for plotting.
    
    The efficient frontier shows the set of optimal portfolios that:
    - Maximize return for a given level of risk, or
    - Minimize risk for a given level of return
    
    Args:
        expected_returns: Array of expected returns
        cov_matrix: Covariance matrix
        n_points: Number of points on the frontier
        risk_free_rate: Risk-free rate
        
    Returns:
        DataFrame with columns: return, risk, weights (as list)
    """
    # Find min and max returns
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    
    # Generate target returns
    target_returns = np.linspace(min_return, max_return, n_points)
    
    frontier_points = []
    
    for target_ret in target_returns:
        try:
            result = mean_variance_optimization(
                expected_returns,
                cov_matrix,
                target_return=target_ret,
                risk_free_rate=risk_free_rate
            )
            frontier_points.append({
                'return': result['expected_return'],
                'risk': result['risk'],
                'sharpe_ratio': result['sharpe_ratio'],
                'weights': result['weights'].tolist()
            })
        except:
            # Skip if optimization fails for this target
            continue
    
    return pd.DataFrame(frontier_points)

