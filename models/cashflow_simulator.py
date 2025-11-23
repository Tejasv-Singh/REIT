"""
Cashflow and NAV simulation engine for REIT/InvIT assets.

This module implements the core simulation logic that:
1. Simulates revenue using stochastic processes
2. Calculates operating income and debt service
3. Computes distributable cashflow and distributions
4. Evolves NAV over time
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .stochastic_processes import simulate_gbm_paths, simulate_ou_paths, apply_poisson_jumps

# Import config - handle both relative and absolute imports
try:
    from ..utils.config import DEFAULT_SIMULATION_PARAMS
except (ImportError, ValueError):
    # Fallback for when running as script or from app.py
    try:
        from utils.config import DEFAULT_SIMULATION_PARAMS
    except ImportError:
        # Last resort: define defaults
        DEFAULT_SIMULATION_PARAMS = {
            'horizon_years': 10,
            'steps_per_year': 12,
            'n_sims': 1000,
            'var_confidence': 0.95,
            'risk_free_rate': 0.03,
            'min_occupancy': 0.0,
            'max_occupancy': 1.0,
            'ou_kappa': 2.0,
            'ou_theta': 0.90,
            'ou_sigma': 0.1,
            'jump_lambda': 0.05,
            'jump_size_mean': -0.10,
            'capex_ratio': 0.02,
        }


class CashflowSimulator:
    """
    Simulates future cashflows and NAV for REIT/InvIT assets.
    
    The simulation process:
    1. Models rent/tariff growth using Geometric Brownian Motion
    2. Models occupancy/utilization using Ornstein-Uhlenbeck process
    3. Calculates revenue, operating expenses, and NOI
    4. Accounts for debt service and distributable cashflow
    5. Evolves NAV over time based on retained cashflow
    """
    
    def __init__(
        self,
        assets_df: pd.DataFrame,
        horizon_years: int = 10,
        steps_per_year: int = 12,
        n_sims: int = 1000,
        random_state: Optional[int] = None
    ):
        """
        Initialize the cashflow simulator.
        
        Args:
            assets_df: DataFrame containing asset parameters
            horizon_years: Simulation time horizon in years
            steps_per_year: Number of time steps per year (e.g., 12 for monthly)
            n_sims: Number of Monte Carlo simulation paths
            random_state: Random seed for reproducibility
        """
        self.assets_df = assets_df
        self.horizon_years = horizon_years
        self.steps_per_year = steps_per_year
        self.n_sims = n_sims
        self.random_state = random_state
        self.total_steps = horizon_years * steps_per_year
        
        # Get config parameters
        self.config = DEFAULT_SIMULATION_PARAMS
        
    def simulate_asset(self, asset_id: str) -> Dict:
        """
        Simulate cashflows and NAV for a single asset.
        
        Args:
            asset_id: Unique identifier for the asset
            
        Returns:
            Dictionary containing:
            - nav_paths: Array (n_sims, time_steps) of NAV evolution
            - distribution_paths: Array (n_sims, time_steps) of distributions
            - revenue_paths: Array (n_sims, time_steps) of revenue
            - dcf_paths: Array (n_sims, time_steps) of distributable cashflow
            - summary_stats: Dictionary with IRR, avg_yield, final_nav_mean, etc.
        """
        # Get asset data
        asset = self.assets_df[self.assets_df['asset_id'] == asset_id].iloc[0]
        
        # Initialize arrays
        nav_paths = np.zeros((self.n_sims, self.total_steps + 1))
        distribution_paths = np.zeros((self.n_sims, self.total_steps))
        revenue_paths = np.zeros((self.n_sims, self.total_steps))
        dcf_paths = np.zeros((self.n_sims, self.total_steps))
        
        # Set initial NAV for all simulations
        nav_paths[:, 0] = asset['current_nav']
        
        # Step 1: Simulate rent/tariff growth using GBM
        rent_paths = simulate_gbm_paths(
            S0=asset['base_rent_or_tariff'],
            mu=asset['rent_tariff_growth_mu'],
            sigma=asset['rent_tariff_growth_sigma'],
            T=self.horizon_years,
            steps=self.total_steps,
            n_sims=self.n_sims,
            random_state=self.random_state
        )
        
        # Step 2: Simulate occupancy/utilization using OU process
        # Use config defaults for OU parameters, but can be customized per asset
        occupancy_paths = simulate_ou_paths(
            X0=asset['current_occupancy_or_utilization'],
            kappa=self.config['ou_kappa'],
            theta=self.config['ou_theta'],
            sigma=self.config['ou_sigma'],
            T=self.horizon_years,
            steps=self.total_steps,
            n_sims=self.n_sims,
            bounds=(self.config['min_occupancy'], self.config['max_occupancy']),
            random_state=self.random_state
        )
        
        # Step 3: Calculate revenue
        # Revenue = rent/tariff * occupancy * scale_factor
        # Scale factor converts per-unit rent to total revenue
        # We use a simple scaling: assume base revenue = base_rent * occupancy * units
        # For simplicity, we scale by current_nav / (base_rent * current_occupancy)
        # This gives us a reasonable revenue magnitude
        scale_factor = asset['current_nav'] / (
            asset['base_rent_or_tariff'] * asset['current_occupancy_or_utilization'] * 1000
        )
        revenue_paths = rent_paths[:, 1:] * occupancy_paths[:, 1:] * scale_factor
        
        # Step 4: Calculate operating income
        opex_paths = revenue_paths * asset['operating_expense_ratio']
        noi_paths = revenue_paths - opex_paths  # Net Operating Income
        
        # Step 5: Calculate debt service
        # Interest expense = debt_ratio * initial_nav * debt_cost (annualized)
        # Convert to per-step
        annual_interest = asset['debt_ratio'] * asset['current_nav'] * asset['debt_cost']
        interest_per_step = annual_interest / self.steps_per_year
        interest_paths = np.full((self.n_sims, self.total_steps), interest_per_step)
        
        # Step 6: Calculate distributable cashflow (DCF)
        dcf_paths = np.maximum(0, noi_paths - interest_paths)
        
        # Step 7: Calculate distributions
        distribution_paths = dcf_paths * asset['payout_ratio']
        
        # Step 8: Evolve NAV over time
        # NAV_{t+1} = NAV_t + (DCF_t - distribution_t) - capex_t
        capex_per_step = asset['current_nav'] * self.config['capex_ratio'] / self.steps_per_year
        
        for t in range(self.total_steps):
            # Retained cashflow = DCF - distributions
            retained_cashflow = dcf_paths[:, t] - distribution_paths[:, t]
            
            # Update NAV
            nav_paths[:, t + 1] = nav_paths[:, t] + retained_cashflow - capex_per_step
            
            # Ensure NAV doesn't go negative (liquidation scenario)
            nav_paths[:, t + 1] = np.maximum(nav_paths[:, t + 1], 0)
        
        # Step 9: Calculate summary statistics
        summary_stats = self._calculate_summary_stats(
            nav_paths=nav_paths,
            distribution_paths=distribution_paths,
            revenue_paths=revenue_paths,
            dcf_paths=dcf_paths,
            initial_nav=asset['current_nav']
        )
        
        return {
            'nav_paths': nav_paths,
            'distribution_paths': distribution_paths,
            'revenue_paths': revenue_paths,
            'dcf_paths': dcf_paths,
            'summary_stats': summary_stats,
            'asset_info': asset.to_dict()
        }
    
    def _calculate_summary_stats(
        self,
        nav_paths: np.ndarray,
        distribution_paths: np.ndarray,
        revenue_paths: np.ndarray,
        dcf_paths: np.ndarray,
        initial_nav: float
    ) -> Dict:
        """
        Calculate summary statistics from simulation results.
        
        Args:
            nav_paths: NAV evolution paths
            distribution_paths: Distribution paths
            revenue_paths: Revenue paths
            dcf_paths: DCF paths
            initial_nav: Initial NAV value
            
        Returns:
            Dictionary of summary statistics
        """
        # Final NAV statistics
        final_nav = nav_paths[:, -1]
        final_nav_mean = np.mean(final_nav)
        final_nav_std = np.std(final_nav)
        final_nav_median = np.median(final_nav)
        
        # Total return
        total_return = (final_nav - initial_nav) / initial_nav
        mean_total_return = np.mean(total_return)
        std_total_return = np.std(total_return)
        
        # Annualized return (assuming horizon_years)
        annualized_return = (1 + mean_total_return) ** (1 / self.horizon_years) - 1
        
        # Distribution yield (annual average)
        # Sum distributions over all periods, divide by initial NAV, annualize
        total_distributions = np.sum(distribution_paths, axis=1)
        avg_annual_yield = np.mean(total_distributions) / initial_nav / self.horizon_years
        
        # IRR approximation (simplified)
        # Use average annual distribution and final NAV
        avg_annual_dist = np.mean(total_distributions) / self.horizon_years
        # Simple IRR approximation: solve for r where NPV = 0
        # For simplicity, use a rough approximation
        try:
            # Use average cashflows for IRR calculation
            avg_final_nav = np.mean(final_nav)
            cashflows = [-initial_nav] + [avg_annual_dist] * self.horizon_years
            cashflows[-1] += avg_final_nav  # Add final NAV to last period
            
            # Simple IRR using numpy financial (if available) or approximation
            irr_approx = self._approximate_irr(cashflows)
        except:
            irr_approx = annualized_return  # Fallback
        
        # Average revenue
        avg_annual_revenue = np.mean(np.sum(revenue_paths, axis=1)) / self.horizon_years
        
        # Average DCF
        avg_annual_dcf = np.mean(np.sum(dcf_paths, axis=1)) / self.horizon_years
        
        return {
            'final_nav_mean': final_nav_mean,
            'final_nav_std': final_nav_std,
            'final_nav_median': final_nav_median,
            'mean_total_return': mean_total_return,
            'std_total_return': std_total_return,
            'annualized_return': annualized_return,
            'avg_annual_yield': avg_annual_yield,
            'irr_approx': irr_approx,
            'avg_annual_revenue': avg_annual_revenue,
            'avg_annual_dcf': avg_annual_dcf,
        }
    
    def _approximate_irr(self, cashflows: list, guess: float = 0.1) -> float:
        """
        Approximate IRR using Newton-Raphson method.
        
        Args:
            cashflows: List of cashflows (negative initial, then positive)
            guess: Initial guess for IRR
            
        Returns:
            Approximated IRR
        """
        from scipy.optimize import fsolve
        
        def npv(r):
            return sum(cf / (1 + r) ** i for i, cf in enumerate(cashflows))
        
        try:
            irr = fsolve(npv, guess)[0]
            # Ensure reasonable range
            if -0.99 < irr < 10:
                return irr
            else:
                return guess
        except:
            return guess
    
    def simulate_all_assets(self) -> Dict[str, Dict]:
        """
        Simulate all assets in the DataFrame.
        
        Returns:
            Dictionary mapping asset_id to simulation results
        """
        results = {}
        for asset_id in self.assets_df['asset_id']:
            results[asset_id] = self.simulate_asset(asset_id)
        return results

