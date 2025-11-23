"""
Risk metrics calculation for portfolio analysis.

This module implements statistical risk measures including:
- Value-at-Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Downside risk
- Sharpe ratio
"""

import numpy as np
from typing import Optional


def compute_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate Value-at-Risk (VaR) at specified confidence level.
    
    VaR is the maximum loss expected at a given confidence level.
    For example, VaR(0.95) means we expect losses to exceed this value
    only 5% of the time.
    
    Args:
        returns: Array of returns (can be 1D or 2D)
        alpha: Confidence level (e.g., 0.95 for 95% confidence)
        
    Returns:
        VaR value (negative number representing potential loss)
        
    Example:
        >>> returns = np.random.normal(0.05, 0.15, 1000)
        >>> var_95 = compute_var(returns, alpha=0.95)
        >>> # var_95 will be negative, representing potential loss
    """
    # Flatten if 2D
    if returns.ndim > 1:
        returns = returns.flatten()
    
    # VaR is the (1-alpha) percentile
    percentile = (1 - alpha) * 100
    var = np.percentile(returns, percentile)
    
    return var


def compute_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """
    Calculate Conditional Value-at-Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    It provides a more conservative risk measure than VaR by considering
    the tail of the distribution.
    
    Args:
        returns: Array of returns (can be 1D or 2D)
        alpha: Confidence level (e.g., 0.95 for 95% confidence)
        
    Returns:
        CVaR value (negative number representing expected loss in tail)
        
    Example:
        >>> returns = np.random.normal(0.05, 0.15, 1000)
        >>> cvar_95 = compute_cvar(returns, alpha=0.95)
        >>> # cvar_95 <= var_95 (more negative)
    """
    # Flatten if 2D
    if returns.ndim > 1:
        returns = returns.flatten()
    
    # Calculate VaR first
    var = compute_var(returns, alpha)
    
    # CVaR is the mean of returns below VaR
    tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        return var  # Fallback to VaR if no tail
    
    cvar = np.mean(tail_returns)
    
    return cvar


def compute_downside_risk(returns: np.ndarray, mar: float = 0.0) -> float:
    """
    Calculate downside deviation below Minimum Acceptable Return (MAR).
    
    Downside risk only considers returns below the MAR threshold,
    which is more relevant for investors who are primarily concerned
    with losses rather than volatility in both directions.
    
    Args:
        returns: Array of returns (can be 1D or 2D)
        mar: Minimum Acceptable Return threshold (default 0.0)
        
    Returns:
        Downside deviation (standard deviation of returns below MAR)
        
    Example:
        >>> returns = np.random.normal(0.05, 0.15, 1000)
        >>> downside = compute_downside_risk(returns, mar=0.0)
        >>> # Measures volatility of negative returns only
    """
    # Flatten if 2D
    if returns.ndim > 1:
        returns = returns.flatten()
    
    # Only consider returns below MAR
    downside_returns = returns[returns < mar]
    
    if len(downside_returns) == 0:
        return 0.0  # No downside if all returns >= MAR
    
    # Calculate standard deviation of downside returns
    downside_risk = np.std(downside_returns)
    
    return downside_risk


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 1
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).
    
    Sharpe ratio = (mean_return - risk_free_rate) / std_deviation
    
    Higher Sharpe ratio indicates better risk-adjusted performance.
    Typically annualized by multiplying by sqrt(periods_per_year).
    
    Args:
        returns: Array of returns (can be 1D or 2D)
        risk_free_rate: Annual risk-free rate (default 0.0)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Sharpe ratio (annualized if periods_per_year > 1)
        
    Example:
        >>> monthly_returns = np.random.normal(0.01, 0.03, 120)
        >>> sharpe = compute_sharpe_ratio(monthly_returns, 
        ...                               risk_free_rate=0.03, 
        ...                               periods_per_year=12)
    """
    # Flatten if 2D
    if returns.ndim > 1:
        returns = returns.flatten()
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate mean and std
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0  # Avoid division by zero
    
    # Annualize if needed
    if periods_per_year > 1:
        mean_return_annual = mean_return * periods_per_year
        std_return_annual = std_return * np.sqrt(periods_per_year)
    else:
        mean_return_annual = mean_return
        std_return_annual = std_return
    
    # Calculate Sharpe ratio
    sharpe = (mean_return_annual - risk_free_rate) / std_return_annual
    
    return sharpe

