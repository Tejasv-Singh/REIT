"""
Unit tests for portfolio optimization.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.portfolio_optimization import (
    calculate_portfolio_stats,
    mean_variance_optimization,
    generate_efficient_frontier
)


@pytest.fixture
def mock_simulation_results():
    """Create mock simulation results for testing."""
    n_sims = 1000
    
    # Create mock results for 3 assets
    results = {}
    for i, asset_id in enumerate(['ASSET1', 'ASSET2', 'ASSET3']):
        # Create NAV paths
        nav_paths = np.random.normal(100, 10, size=(n_sims, 121))
        nav_paths[:, 0] = 100  # Initial NAV
        
        # Create summary stats
        results[asset_id] = {
            'nav_paths': nav_paths,
            'summary_stats': {
                'annualized_return': 0.05 + i * 0.01  # Different returns
            }
        }
    
    return results


class TestPortfolioOptimization:
    """Tests for portfolio optimization functions."""
    
    def test_calculate_portfolio_stats_shape(self, mock_simulation_results):
        """Test that portfolio stats have correct shapes."""
        expected_returns, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        n_assets = len(mock_simulation_results)
        assert expected_returns.shape == (n_assets,), \
            f"Expected returns shape should be ({n_assets},), got {expected_returns.shape}"
        assert cov_matrix.shape == (n_assets, n_assets), \
            f"Covariance matrix shape should be ({n_assets}, {n_assets}), got {cov_matrix.shape}"
    
    def test_covariance_positive_semi_definite(self, mock_simulation_results):
        """Test that covariance matrix is positive semi-definite."""
        _, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        eigenvals = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvals >= -1e-10), \
            "Covariance matrix should be positive semi-definite"
    
    def test_optimization_weights_sum_to_one(self, mock_simulation_results):
        """Test that optimal weights sum to 1.0."""
        expected_returns, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        result = mean_variance_optimization(
            expected_returns,
            cov_matrix,
            target_return=None
        )
        
        weights_sum = np.sum(result['weights'])
        assert np.isclose(weights_sum, 1.0, atol=1e-6), \
            f"Weights should sum to 1.0, got {weights_sum}"
    
    def test_optimization_weights_non_negative(self, mock_simulation_results):
        """Test that all weights are non-negative (no short selling)."""
        expected_returns, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        result = mean_variance_optimization(
            expected_returns,
            cov_matrix,
            target_return=None
        )
        
        assert np.all(result['weights'] >= -1e-6), \
            "All weights should be non-negative (no short selling)"
    
    def test_optimization_with_target_return(self, mock_simulation_results):
        """Test optimization with target return constraint."""
        expected_returns, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        target_return = np.mean(expected_returns)
        result = mean_variance_optimization(
            expected_returns,
            cov_matrix,
            target_return=target_return
        )
        
        # Portfolio return should be >= target (with small tolerance)
        assert result['expected_return'] >= target_return - 1e-6, \
            "Portfolio return should meet target return constraint"
    
    def test_efficient_frontier_generation(self, mock_simulation_results):
        """Test efficient frontier generation."""
        expected_returns, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        frontier_df = generate_efficient_frontier(
            expected_returns,
            cov_matrix,
            n_points=20
        )
        
        # Check DataFrame structure
        assert 'return' in frontier_df.columns
        assert 'risk' in frontier_df.columns
        assert 'sharpe_ratio' in frontier_df.columns
        assert 'weights' in frontier_df.columns
        
        # Check that risk increases with return (generally)
        # (Efficient frontier should be upward-sloping)
        sorted_df = frontier_df.sort_values('return')
        assert len(sorted_df) > 0, "Frontier should have points"
        
        # Check that weights in each row sum to ~1.0
        for _, row in frontier_df.iterrows():
            weights = np.array(row['weights'])
            assert np.isclose(np.sum(weights), 1.0, atol=1e-5), \
                "Weights in frontier should sum to 1.0"
    
    def test_optimization_output_structure(self, mock_simulation_results):
        """Test that optimization returns expected structure."""
        expected_returns, cov_matrix = calculate_portfolio_stats(mock_simulation_results)
        
        result = mean_variance_optimization(
            expected_returns,
            cov_matrix,
            target_return=None
        )
        
        required_keys = ['weights', 'expected_return', 'risk', 'sharpe_ratio']
        for key in required_keys:
            assert key in result, f"Missing key in optimization result: {key}"
        
        # Check types
        assert isinstance(result['weights'], np.ndarray)
        assert isinstance(result['expected_return'], (float, np.floating))
        assert isinstance(result['risk'], (float, np.floating))
        assert isinstance(result['sharpe_ratio'], (float, np.floating))

