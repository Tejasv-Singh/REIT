"""
Unit tests for stochastic process implementations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.stochastic_processes import simulate_gbm_paths, simulate_ou_paths, apply_poisson_jumps


class TestGBM:
    """Tests for Geometric Brownian Motion."""
    
    def test_deterministic_growth_zero_volatility(self):
        """Test GBM with zero volatility should give deterministic exponential growth."""
        S0 = 100.0
        mu = 0.05
        sigma = 0.0  # Zero volatility
        T = 1.0
        steps = 12
        n_sims = 100
        
        paths = simulate_gbm_paths(S0, mu, sigma, T, steps, n_sims, random_state=42)
        
        # All paths should be identical (deterministic)
        assert np.allclose(paths, paths[0, :]), "All paths should be identical with zero volatility"
        
        # Final value should be approximately S0 * exp(mu * T)
        expected_final = S0 * np.exp(mu * T)
        actual_final = paths[0, -1]
        assert np.isclose(actual_final, expected_final, rtol=1e-3), \
            f"Final value should be {expected_final}, got {actual_final}"
    
    def test_gbm_shape(self):
        """Test that GBM returns correct array shape."""
        paths = simulate_gbm_paths(100, 0.05, 0.15, 10, 120, 1000, random_state=42)
        assert paths.shape == (1000, 121), f"Expected shape (1000, 121), got {paths.shape}"
    
    def test_gbm_initial_value(self):
        """Test that all paths start at S0."""
        S0 = 100.0
        paths = simulate_gbm_paths(S0, 0.05, 0.15, 10, 120, 1000, random_state=42)
        assert np.allclose(paths[:, 0], S0), "All paths should start at S0"
    
    def test_gbm_positive_values(self):
        """Test that GBM paths remain positive."""
        paths = simulate_gbm_paths(100, 0.05, 0.15, 10, 120, 1000, random_state=42)
        assert np.all(paths > 0), "GBM paths should always be positive"


class TestOU:
    """Tests for Ornstein-Uhlenbeck process."""
    
    def test_ou_bounds(self):
        """Test that OU process respects bounds."""
        X0 = 0.5
        kappa = 2.0
        theta = 0.8
        sigma = 0.1
        bounds = (0, 1)
        
        paths = simulate_ou_paths(
            X0, kappa, theta, sigma, 10, 120, 1000,
            bounds=bounds, random_state=42
        )
        
        assert np.all(paths >= bounds[0]), "OU paths should respect lower bound"
        assert np.all(paths <= bounds[1]), "OU paths should respect upper bound"
    
    def test_ou_mean_reversion(self):
        """Test that OU process mean-reverts with high kappa."""
        X0 = 0.2  # Start far from theta
        kappa = 10.0  # High mean reversion speed
        theta = 0.8
        sigma = 0.05  # Low volatility
        T = 5.0
        steps = 60
        
        paths = simulate_ou_paths(
            X0, kappa, theta, sigma, T, steps, 100,
            bounds=(0, 1), random_state=42
        )
        
        # After some time, paths should be closer to theta
        later_values = paths[:, steps // 2:]
        mean_later = np.mean(later_values)
        
        # Mean should be closer to theta than to X0
        assert abs(mean_later - theta) < abs(mean_later - X0), \
            "OU process should mean-revert toward theta"
    
    def test_ou_shape(self):
        """Test that OU returns correct array shape."""
        paths = simulate_ou_paths(0.5, 2.0, 0.8, 0.1, 10, 120, 1000, random_state=42)
        assert paths.shape == (1000, 121), f"Expected shape (1000, 121), got {paths.shape}"
    
    def test_ou_initial_value(self):
        """Test that all paths start at X0."""
        X0 = 0.5
        paths = simulate_ou_paths(X0, 2.0, 0.8, 0.1, 10, 120, 1000, random_state=42)
        assert np.allclose(paths[:, 0], X0), "All paths should start at X0"


class TestPoissonJumps:
    """Tests for Poisson jump process."""
    
    def test_jump_shape(self):
        """Test that jump process preserves array shape."""
        base_paths = np.ones((100, 121)) * 100
        jumped = apply_poisson_jumps(base_paths, 0.1, -0.05, 10)
        assert jumped.shape == base_paths.shape, "Jump process should preserve shape"
    
    def test_jump_downward(self):
        """Test that negative jump size causes downward jumps."""
        base_paths = np.ones((1000, 121)) * 100
        jumped = apply_poisson_jumps(base_paths, 0.5, -0.10, 10)  # High jump rate
        
        # Some paths should have decreased
        final_values = jumped[:, -1]
        assert np.any(final_values < 100), "Some paths should have downward jumps"
    
    def test_jump_non_negative(self):
        """Test that jumps don't make values negative."""
        base_paths = np.ones((100, 121)) * 100
        jumped = apply_poisson_jumps(base_paths, 0.1, -0.05, 10)
        assert np.all(jumped >= 0), "Jumped paths should remain non-negative"

