"""
Configuration constants for the REIT/InvIT simulation tool.
"""

# Default simulation parameters
DEFAULT_SIMULATION_PARAMS = {
    'horizon_years': 10,
    'steps_per_year': 12,  # Monthly steps
    'n_sims': 1000,
    'var_confidence': 0.95,
    'risk_free_rate': 0.03,  # 3% annual risk-free rate
    'min_occupancy': 0.0,
    'max_occupancy': 1.0,
    'ou_kappa': 2.0,  # Mean reversion speed for occupancy
    'ou_theta': 0.90,  # Long-term occupancy target
    'ou_sigma': 0.1,  # Occupancy volatility
    'jump_lambda': 0.05,  # Annual jump rate (infrequent shocks)
    'jump_size_mean': -0.10,  # 10% downward jump on average
    'capex_ratio': 0.02,  # 2% of NAV per year for capex
}

