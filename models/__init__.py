"""
Quantitative models for REIT/InvIT cashflow simulation and portfolio optimization.
"""

from .stochastic_processes import (
    simulate_gbm_paths,
    simulate_ou_paths,
    apply_poisson_jumps
)
from .cashflow_simulator import CashflowSimulator
from .risk_metrics import (
    compute_var,
    compute_cvar,
    compute_downside_risk,
    compute_sharpe_ratio
)
from .portfolio_optimization import (
    calculate_portfolio_stats,
    mean_variance_optimization,
    generate_efficient_frontier
)

__all__ = [
    'simulate_gbm_paths',
    'simulate_ou_paths',
    'apply_poisson_jumps',
    'CashflowSimulator',
    'compute_var',
    'compute_cvar',
    'compute_downside_risk',
    'compute_sharpe_ratio',
    'calculate_portfolio_stats',
    'mean_variance_optimization',
    'generate_efficient_frontier'
]

