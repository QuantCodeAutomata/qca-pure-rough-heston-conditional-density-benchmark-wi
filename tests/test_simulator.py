"""
Tests for src/simulator.py and src/config.py.

Validates:
  - Parameter validation (edge cases and invalid inputs).
  - Simulation output shapes and types.
  - Variance positivity (full-truncation property).
  - Reproducibility with fixed seeds.
  - Negative-V fraction reporting.
  - Idiosyncratic spot simulation output shape.
"""
from __future__ import annotations

import pytest
import numpy as np

from src.config import RoughHestonParams, make_param_grid
from src.simulator import simulate_rough_heston, simulate_idiosyncratic_paths


# ──────────────────────────────────────────────
# RoughHestonParams validation
# ──────────────────────────────────────────────

class TestRoughHestonParamsValidation:
    def test_valid_params_no_error(self) -> None:
        p = RoughHestonParams(H=0.1, V0=0.04, kappa=1.0, theta=0.04,
                              nu=0.5, rho=-0.7, S0=1.0, T=1.0, n_steps=10)
        p.validate()  # should not raise

    def test_H_out_of_range_below(self) -> None:
        with pytest.raises(ValueError, match="H"):
            RoughHestonParams(H=0.0)

    def test_H_out_of_range_above(self) -> None:
        with pytest.raises(ValueError, match="H"):
            RoughHestonParams(H=0.5)

    def test_V0_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="V0"):
            RoughHestonParams(V0=0.0)

    def test_kappa_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="kappa"):
            RoughHestonParams(kappa=0.0)

    def test_theta_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="theta"):
            RoughHestonParams(theta=0.0)

    def test_nu_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="nu"):
            RoughHestonParams(nu=0.0)

    def test_S0_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="S0"):
            RoughHestonParams(S0=0.0)

    def test_rho_too_large(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            RoughHestonParams(rho=1.0)

    def test_rho_too_small(self) -> None:
        with pytest.raises(ValueError, match="rho"):
            RoughHestonParams(rho=-1.0)

    def test_dt_property(self) -> None:
        p = RoughHestonParams(T=2.0, n_steps=100)
        assert abs(p.dt - 0.02) < 1e-12

    def test_time_grid_endpoints(self) -> None:
        p = RoughHestonParams(T=1.0, n_steps=50)
        tg = p.time_grid
        assert tg[0] == pytest.approx(0.0)
        assert tg[-1] == pytest.approx(1.0)
        assert len(tg) == 51


class TestMakeParamGrid:
    def test_grid_size(self) -> None:
        grid = make_param_grid(H_values=[0.1, 0.2], rho_values=[-0.5, 0.5])
        assert len(grid) == 4

    def test_grid_values(self) -> None:
        grid = make_param_grid(H_values=[0.15], rho_values=[-0.3])
        assert grid[0].H == pytest.approx(0.15)
        assert grid[0].rho == pytest.approx(-0.3)


# ──────────────────────────────────────────────
# simulate_rough_heston
# ──────────────────────────────────────────────

class TestSimulateRoughHeston:
    PARAMS = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.5, rho=-0.7, r=0.05, d=0.0, T=1.0, n_steps=50, seed=42,
    )

    def test_output_keys(self) -> None:
        result = simulate_rough_heston(self.PARAMS, N=3)
        for key in ["V", "dZ", "Z", "time_grid", "negative_V_fraction", "params", "scheme"]:
            assert key in result

    def test_shapes(self) -> None:
        N, n = 5, self.PARAMS.n_steps
        result = simulate_rough_heston(self.PARAMS, N)
        assert result["V"].shape == (N, n + 1)
        assert result["dZ"].shape == (N, n)
        assert result["Z"].shape == (N, n + 1)
        assert result["time_grid"].shape == (n + 1,)

    def test_V_nonnegative(self) -> None:
        """After full truncation, V must be >= 0."""
        result = simulate_rough_heston(self.PARAMS, N=20)
        assert np.all(result["V"] >= 0.0)

    def test_V0_initial_condition(self) -> None:
        result = simulate_rough_heston(self.PARAMS, N=10)
        assert np.all(result["V"][:, 0] == pytest.approx(self.PARAMS.V0))

    def test_Z0_initial_condition(self) -> None:
        result = simulate_rough_heston(self.PARAMS, N=10)
        assert np.all(result["Z"][:, 0] == pytest.approx(0.0))

    def test_reproducibility_with_seed(self) -> None:
        r1 = simulate_rough_heston(self.PARAMS, N=5, seed=99)
        r2 = simulate_rough_heston(self.PARAMS, N=5, seed=99)
        np.testing.assert_array_equal(r1["V"], r2["V"])
        np.testing.assert_array_equal(r1["dZ"], r2["dZ"])

    def test_different_seeds_differ(self) -> None:
        r1 = simulate_rough_heston(self.PARAMS, N=5, seed=1)
        r2 = simulate_rough_heston(self.PARAMS, N=5, seed=2)
        assert not np.allclose(r1["V"], r2["V"])

    def test_negative_V_fraction_in_range(self) -> None:
        result = simulate_rough_heston(self.PARAMS, N=10)
        frac = result["negative_V_fraction"]
        assert 0.0 <= frac <= 1.0

    def test_dZ_increments_scaled_by_sqrt_dt(self) -> None:
        """dZ should have variance ~ dt."""
        result = simulate_rough_heston(self.PARAMS, N=2000, seed=1)
        dZ = result["dZ"]
        empirical_var = np.var(dZ.ravel())
        assert abs(empirical_var - self.PARAMS.dt) < 0.005, (
            f"Expected dZ variance ~ dt={self.PARAMS.dt:.4f}, got {empirical_var:.4f}"
        )

    def test_small_H_runs_without_error(self) -> None:
        p = RoughHestonParams(H=0.05, n_steps=30)
        result = simulate_rough_heston(p, N=3)
        assert result["V"].shape[0] == 3

    def test_large_H_runs_without_error(self) -> None:
        p = RoughHestonParams(H=0.45, n_steps=30)
        result = simulate_rough_heston(p, N=3)
        assert result["V"].shape[0] == 3


class TestSimulateIdiosyncraticPaths:
    PARAMS = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.5, rho=-0.7, r=0.05, d=0.0, T=0.5, n_steps=50, seed=42,
    )

    def test_shape(self) -> None:
        sim = simulate_rough_heston(self.PARAMS, N=2)
        S = simulate_idiosyncratic_paths(
            self.PARAMS, sim["V"][0], sim["dZ"][0], M=10, seed=1
        )
        assert S.shape == (10, self.PARAMS.n_steps + 1)

    def test_S0_initial_condition(self) -> None:
        sim = simulate_rough_heston(self.PARAMS, N=2)
        S = simulate_idiosyncratic_paths(
            self.PARAMS, sim["V"][0], sim["dZ"][0], M=5, seed=1
        )
        assert np.all(S[:, 0] == pytest.approx(self.PARAMS.S0))

    def test_S_positive(self) -> None:
        """All spot prices should be positive (log-Euler method)."""
        sim = simulate_rough_heston(self.PARAMS, N=2)
        S = simulate_idiosyncratic_paths(
            self.PARAMS, sim["V"][0], sim["dZ"][0], M=100, seed=7
        )
        assert np.all(S > 0.0)
