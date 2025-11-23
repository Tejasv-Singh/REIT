"""
Unit tests for cashflow simulation.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cashflow_simulator import CashflowSimulator


@pytest.fixture
def sample_assets_df():
    """Create sample asset DataFrame for testing."""
    return pd.DataFrame({
        'asset_id': ['TEST001', 'TEST002'],
        'name': ['Test Asset 1', 'Test Asset 2'],
        'asset_type': ['Office REIT', 'Retail REIT'],
        'base_rent_or_tariff': [100.0, 80.0],
        'current_occupancy_or_utilization': [0.85, 0.90],
        'rent_tariff_growth_mu': [0.04, 0.03],
        'rent_tariff_growth_sigma': [0.12, 0.15],
        'operating_expense_ratio': [0.25, 0.30],
        'debt_ratio': [0.45, 0.50],
        'debt_cost': [0.055, 0.060],
        'payout_ratio': [0.90, 0.95],
        'current_nav': [100000000, 80000000],
        'beta': [0.85, 0.90]
    })


class TestCashflowSimulator:
    """Tests for CashflowSimulator class."""
    
    def test_simulator_initialization(self, sample_assets_df):
        """Test simulator initialization."""
        simulator = CashflowSimulator(
            sample_assets_df,
            horizon_years=10,
            steps_per_year=12,
            n_sims=100,
            random_state=42
        )
        
        assert simulator.horizon_years == 10
        assert simulator.steps_per_year == 12
        assert simulator.n_sims == 100
        assert simulator.total_steps == 120
    
    def test_simulate_asset_output_shape(self, sample_assets_df):
        """Test that simulation outputs have correct shapes."""
        simulator = CashflowSimulator(
            sample_assets_df,
            horizon_years=10,
            steps_per_year=12,
            n_sims=100,
            random_state=42
        )
        
        result = simulator.simulate_asset('TEST001')
        
        # Check output structure
        assert 'nav_paths' in result
        assert 'distribution_paths' in result
        assert 'revenue_paths' in result
        assert 'dcf_paths' in result
        assert 'summary_stats' in result
        
        # Check shapes
        assert result['nav_paths'].shape == (100, 121), "NAV paths shape incorrect"
        assert result['distribution_paths'].shape == (100, 120), "Distribution paths shape incorrect"
        assert result['revenue_paths'].shape == (100, 120), "Revenue paths shape incorrect"
        assert result['dcf_paths'].shape == (100, 120), "DCF paths shape incorrect"
    
    def test_distributions_non_negative(self, sample_assets_df):
        """Test that distributions are non-negative."""
        simulator = CashflowSimulator(
            sample_assets_df,
            horizon_years=10,
            steps_per_year=12,
            n_sims=100,
            random_state=42
        )
        
        result = simulator.simulate_asset('TEST001')
        distributions = result['distribution_paths']
        
        assert np.all(distributions >= 0), "Distributions should be non-negative"
    
    def test_nav_non_negative(self, sample_assets_df):
        """Test that NAV remains non-negative."""
        simulator = CashflowSimulator(
            sample_assets_df,
            horizon_years=10,
            steps_per_year=12,
            n_sims=100,
            random_state=42
        )
        
        result = simulator.simulate_asset('TEST001')
        nav_paths = result['nav_paths']
        
        assert np.all(nav_paths >= 0), "NAV should remain non-negative"
    
    def test_summary_stats_structure(self, sample_assets_df):
        """Test that summary stats contain expected keys."""
        simulator = CashflowSimulator(
            sample_assets_df,
            horizon_years=10,
            steps_per_year=12,
            n_sims=100,
            random_state=42
        )
        
        result = simulator.simulate_asset('TEST001')
        stats = result['summary_stats']
        
        expected_keys = [
            'final_nav_mean', 'final_nav_std', 'final_nav_median',
            'mean_total_return', 'std_total_return', 'annualized_return',
            'avg_annual_yield', 'irr_approx', 'avg_annual_revenue', 'avg_annual_dcf'
        ]
        
        for key in expected_keys:
            assert key in stats, f"Missing key in summary_stats: {key}"
    
    def test_simulate_all_assets(self, sample_assets_df):
        """Test simulation of all assets."""
        simulator = CashflowSimulator(
            sample_assets_df,
            horizon_years=10,
            steps_per_year=12,
            n_sims=100,
            random_state=42
        )
        
        results = simulator.simulate_all_assets()
        
        assert len(results) == 2, "Should simulate all assets"
        assert 'TEST001' in results
        assert 'TEST002' in results
        
        # Check that each result has required keys
        for asset_id, result in results.items():
            assert 'nav_paths' in result
            assert 'summary_stats' in result

