"""
Stochastic process implementations for financial modeling.

This module provides implementations of:
- Geometric Brownian Motion (GBM) for rent/tariff growth
- Ornstein-Uhlenbeck (OU) process for occupancy/utilization
- Poisson jump process for infrastructure shocks
"""

import numpy as np
from typing import Optional, Tuple


def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    n_sims: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Simulate paths of Geometric Brownian Motion (GBM).
    
    The GBM process follows: dS_t = μ S_t dt + σ S_t dW_t
    
    This is used to model rent/tariff growth over time, where:
    - S0: Initial value (starting rent/tariff)
    - mu: Drift (expected annual growth rate)
    - sigma: Volatility (growth uncertainty)
    - T: Time horizon in years
    - steps: Number of time steps
    - n_sims: Number of simulation paths
    
    Args:
        S0: Initial value of the process
        mu: Drift parameter (annual growth rate)
        sigma: Volatility parameter (annual standard deviation)
        T: Time horizon in years
        steps: Number of discrete time steps
        n_sims: Number of independent simulation paths
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (n_sims, steps+1) containing simulation paths.
        First column contains initial value S0 for all paths.
        
    Example:
        >>> paths = simulate_gbm_paths(S0=100, mu=0.05, sigma=0.15, 
        ...                            T=10, steps=120, n_sims=1000)
        >>> paths.shape
        (1000, 121)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    dt = T / steps
    # Generate random increments: dW ~ N(0, dt)
    dW = np.random.normal(0, np.sqrt(dt), size=(n_sims, steps))
    
    # Initialize paths with S0
    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = S0
    
    # Euler-Maruyama discretization for GBM
    # S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*dW_t)
    drift_term = (mu - 0.5 * sigma ** 2) * dt
    diffusion_term = sigma * dW
    
    for t in range(steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift_term + diffusion_term[:, t])
    
    return paths


def simulate_ou_paths(
    X0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    steps: int,
    n_sims: int,
    bounds: Tuple[float, float] = (0, 1),
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Simulate paths of Ornstein-Uhlenbeck (OU) process.
    
    The OU process follows: dX_t = κ(θ - X_t)dt + σ dW_t
    
    This is used to model occupancy/utilization rates, which:
    - Mean-revert to a long-term level (theta)
    - Have mean reversion speed (kappa)
    - Are bounded between 0 and 1 (for occupancy/utilization)
    
    Args:
        X0: Initial value of the process
        kappa: Mean reversion speed (higher = faster reversion)
        theta: Long-term mean (target occupancy/utilization)
        sigma: Volatility parameter
        T: Time horizon in years
        steps: Number of discrete time steps
        n_sims: Number of independent simulation paths
        bounds: Tuple (min, max) to clip the process values
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (n_sims, steps+1) containing simulation paths.
        Values are clipped to the specified bounds.
        
    Example:
        >>> paths = simulate_ou_paths(X0=0.85, kappa=2.0, theta=0.90,
        ...                           sigma=0.1, T=10, steps=120, n_sims=1000)
        >>> assert np.all(paths >= 0) and np.all(paths <= 1)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    dt = T / steps
    # Generate random increments
    dW = np.random.normal(0, np.sqrt(dt), size=(n_sims, steps))
    
    # Initialize paths
    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = X0
    
    # Euler-Maruyama discretization for OU
    # X_{t+1} = X_t + kappa*(theta - X_t)*dt + sigma*dW_t
    for t in range(steps):
        mean_reversion = kappa * (theta - paths[:, t]) * dt
        diffusion = sigma * dW[:, t]
        paths[:, t + 1] = paths[:, t] + mean_reversion + diffusion
        
        # Clip to bounds (e.g., occupancy must be between 0 and 1)
        paths[:, t + 1] = np.clip(paths[:, t + 1], bounds[0], bounds[1])
    
    return paths


def apply_poisson_jumps(
    base_paths: np.ndarray,
    lambda_rate: float,
    jump_size_mean: float,
    T: float
) -> np.ndarray:
    """
    Apply Poisson jump process to existing paths.
    
    This models infrastructure shocks (e.g., natural disasters, regulatory changes)
    that cause sudden downward jumps in cashflows.
    
    The process adds random downward jumps:
    - Jump times follow Poisson process with rate lambda
    - Jump sizes are negative (downward) with mean jump_size_mean
    
    Args:
        base_paths: Array of shape (n_sims, time_steps) - base paths to modify
        lambda_rate: Annual jump arrival rate (expected jumps per year)
        jump_size_mean: Mean jump size (negative for downward jumps)
        T: Time horizon in years (to scale the Poisson rate)
        
    Returns:
        Modified paths with jumps applied. Same shape as base_paths.
        
    Example:
        >>> base = np.ones((1000, 121)) * 100
        >>> jumped = apply_poisson_jumps(base, lambda_rate=0.1, 
        ...                              jump_size_mean=-0.05, T=10)
        >>> # Some paths will have downward jumps
    """
    n_sims, time_steps = base_paths.shape
    dt = T / (time_steps - 1)
    
    # Scale lambda to per-step rate
    lambda_per_step = lambda_rate * dt
    
    # Generate jump indicators for each path and time step
    # 1 if jump occurs, 0 otherwise
    jump_indicators = np.random.poisson(lambda_per_step, size=(n_sims, time_steps))
    jump_indicators = (jump_indicators > 0).astype(float)
    
    # Generate jump sizes (negative for downward jumps)
    # Using exponential distribution for jump magnitudes
    jump_sizes = np.random.exponential(
        abs(jump_size_mean),
        size=(n_sims, time_steps)
    ) * np.sign(jump_size_mean)
    
    # Apply jumps multiplicatively: new_value = old_value * (1 + jump)
    jump_effects = 1 + (jump_indicators * jump_sizes)
    
    # Apply cumulative effect along time dimension
    # Each jump affects all future values
    modified_paths = base_paths.copy()
    for t in range(time_steps):
        # Apply jump effect from this time step forward
        if t < time_steps - 1:
            modified_paths[:, t + 1:] *= jump_effects[:, t:t + 1]
    
    return modified_paths

