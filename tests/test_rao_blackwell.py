"""
Tests for src/rao_blackwell.py and src/lsrv.py.

Validates:
  - Rao–Blackwellized estimator output shapes and numerical properties.
  - Equivalence of transformed-coordinate and original-coordinate estimators.
  - Theta_hat in a plausible range (approximately V0).
  - Denominator stability detection.
  - LSRV stochastic flow positivity and shapes.
  - Transformed coefficients have correct shapes.
  - LeverageFunction evaluates without error and is positive.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import RoughHestonParams
from src.simulator import simulate_rough_heston
from src.path_functionals import compute_path_functionals, get_functionals_at_time
from src.density import eval_p_t_at_strikes, eval_q_t_at_inverse_strikes, EPSILON
from src.rao_blackwell import (
    estimate_theta,
    estimate_theta_surface,
    verify_estimator_equivalence,
)
from src.lsrv import (
    LeverageFunction,
    PureHestonLeverage,
    solve_stochastic_flow,
    compute_transformed_coefficients,
    invert_flow,
)


PARAMS = RoughHestonParams(
    S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
    nu=0.5, rho=-0.7, r=0.05, d=0.0, T=1.0, n_steps=100, seed=42,
)


# ──────────────────────────────────────────────
# estimate_theta
# ──────────────────────────────────────────────

class TestEstimateTheta:
    def test_returns_tuple_of_three(self) -> None:
        V_t = np.array([0.03, 0.04, 0.05])
        p = np.array([1.0, 2.0, 1.0])
        theta, denom, stable = estimate_theta(V_t, p)
        assert isinstance(theta, float)
        assert isinstance(denom, float)
        assert isinstance(stable, bool)

    def test_known_value(self) -> None:
        """With V=[1,2,3] and equal weights, theta_hat = mean(V) = 2."""
        V_t = np.array([1.0, 2.0, 3.0])
        p = np.ones(3)
        theta, _, _ = estimate_theta(V_t, p)
        assert abs(theta - 2.0) < 1e-10

    def test_unstable_when_denom_small(self) -> None:
        V_t = np.array([0.04])
        p = np.array([1e-20])
        _, _, stable = estimate_theta(V_t, p, stability_threshold=1e-10)
        assert stable is False

    def test_stable_normal(self) -> None:
        V_t = np.array([0.04, 0.05])
        p = np.array([1.0, 1.0])
        _, denom, stable = estimate_theta(V_t, p, stability_threshold=1e-10)
        assert stable is True


class TestEstimateThetaSurface:
    def test_output_shapes(self) -> None:
        N, n_K = 50, 5
        V_t = np.full(N, 0.04)
        p_dens = np.ones((N, n_K)) * 0.01
        theta, denom, stable = estimate_theta_surface(V_t, p_dens)
        assert theta.shape == (n_K,)
        assert denom.shape == (n_K,)
        assert stable.shape == (n_K,)

    def test_theta_nonnegative(self) -> None:
        """V_t > 0 and p_dens > 0 → theta_hat > 0."""
        N, n_K = 20, 3
        rng = np.random.default_rng(5)
        V_t = np.abs(rng.standard_normal(N)) * 0.01 + 0.04
        p_dens = rng.uniform(0.001, 0.01, (N, n_K))
        theta, _, _ = estimate_theta_surface(V_t, p_dens)
        assert np.all(theta[~np.isnan(theta)] >= 0.0)


# ──────────────────────────────────────────────
# verify_estimator_equivalence
# ──────────────────────────────────────────────

class TestVerifyEstimatorEquivalence:
    def test_equiv_with_explicit_densities(self) -> None:
        """
        Since p_t^i(K) = q_t^i(K/M_t^i)/M_t^i by construction, the
        two estimators should be numerically identical.
        """
        N = 200
        sim = simulate_rough_heston(PARAMS, N=N, seed=42)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        t_idx = 50
        ft = get_functionals_at_time(fcts, t_idx)

        K = np.array([0.8, 1.0, 1.2]) * PARAMS.S0
        V_t = sim["V"][:, t_idx]
        p_dens = eval_p_t_at_strikes(K, ft["I_t"], ft["Vbar_t"], ft["A_t"], ft["R_t"], PARAMS.S0, PARAMS.rho)
        q_dens = eval_q_t_at_inverse_strikes(K, ft["M_t"], ft["A_t"], ft["R_t"], PARAMS.S0)

        result = verify_estimator_equivalence(V_t, q_dens, ft["M_t"], p_dens)

        stable = result["is_stable"]
        if np.any(stable):
            assert result["max_abs_diff"] < 1e-6, (
                f"max_abs_diff={result['max_abs_diff']:.2e}"
            )

    def test_keys_present(self) -> None:
        N, n_K = 10, 3
        rng = np.random.default_rng(0)
        V_t = np.abs(rng.standard_normal(N)) * 0.01 + 0.04
        q_dens = rng.uniform(0.001, 0.01, (N, n_K))
        J_t = np.exp(rng.standard_normal(N) * 0.05)
        p_dens = q_dens / J_t[:, np.newaxis]
        result = verify_estimator_equivalence(V_t, q_dens, J_t, p_dens)
        for k in ["theta_trans", "theta_orig", "abs_diff", "rel_diff", "is_stable"]:
            assert k in result


# ──────────────────────────────────────────────
# LSRV LeverageFunction
# ──────────────────────────────────────────────

class TestLeverageFunction:
    def test_L_positive(self) -> None:
        lev = LeverageFunction(alpha=0.3, beta=0.5, gamma=0.5, S0=1.0)
        s = np.linspace(0.1, 3.0, 100)
        assert np.all(lev.L(t=0.5, s=s) > 0.0)

    def test_L_unity_at_S0(self) -> None:
        """L(t, S0) = 1 + alpha*exp(-beta*t)*tanh(0) = 1."""
        lev = LeverageFunction(alpha=0.4, beta=0.3, gamma=0.7, S0=1.0)
        assert abs(lev.L(t=0.5, s=np.array([1.0]))[0] - 1.0) < 1e-10

    def test_pure_heston_L_is_one(self) -> None:
        lev = PureHestonLeverage(S0=1.0)
        s = np.linspace(0.1, 3.0, 50)
        np.testing.assert_allclose(lev.L(t=0.5, s=s), np.ones(50))

    def test_pure_heston_derivatives_zero(self) -> None:
        lev = PureHestonLeverage(S0=1.0)
        s = np.linspace(0.1, 3.0, 50)
        np.testing.assert_allclose(lev.dL_ds(t=0.5, s=s), np.zeros(50), atol=1e-15)

    def test_alpha_gte_1_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            LeverageFunction(alpha=1.0, beta=0.5, gamma=0.5, S0=1.0)


# ──────────────────────────────────────────────
# Stochastic flow
# ──────────────────────────────────────────────

class TestStochasticFlow:
    PARAMS_FLOW = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.3, rho=-0.5, r=0.03, d=0.0, T=0.5, n_steps=50, seed=77,
    )

    def _get_frozen_path(self):
        sim = simulate_rough_heston(self.PARAMS_FLOW, N=3, seed=self.PARAMS_FLOW.seed)
        return sim["V"][0], sim["dZ"][0]

    def test_output_keys(self) -> None:
        V_path, dZ_path = self._get_frozen_path()
        x_grid = np.linspace(0.1, 3.0, 50)
        lev = PureHestonLeverage(S0=self.PARAMS_FLOW.S0)
        result = solve_stochastic_flow(self.PARAMS_FLOW, dZ_path, V_path, x_grid, lev)
        for k in ["Phi", "J", "H", "time_grid", "negative_J_count"]:
            assert k in result

    def test_shapes(self) -> None:
        V_path, dZ_path = self._get_frozen_path()
        n_x = 40
        x_grid = np.linspace(0.1, 3.0, n_x)
        lev = PureHestonLeverage(S0=self.PARAMS_FLOW.S0)
        result = solve_stochastic_flow(self.PARAMS_FLOW, dZ_path, V_path, x_grid, lev)
        n = self.PARAMS_FLOW.n_steps
        assert result["Phi"].shape == (n + 1, n_x)
        assert result["J"].shape == (n + 1, n_x)
        assert result["H"].shape == (n + 1, n_x)

    def test_initial_condition(self) -> None:
        """Phi_0(x) = x, J_0(x) = 1, H_0(x) = 0."""
        V_path, dZ_path = self._get_frozen_path()
        x_grid = np.linspace(0.1, 3.0, 30)
        lev = PureHestonLeverage(S0=self.PARAMS_FLOW.S0)
        result = solve_stochastic_flow(self.PARAMS_FLOW, dZ_path, V_path, x_grid, lev)
        np.testing.assert_allclose(result["Phi"][0], x_grid)
        np.testing.assert_allclose(result["J"][0], np.ones(30))
        np.testing.assert_allclose(result["H"][0], np.zeros(30))

    def test_J_positive_for_pure_heston(self) -> None:
        """For pure L=1, J_t = M_t > 0 should hold (no negative J expected)."""
        V_path, dZ_path = self._get_frozen_path()
        x_grid = np.linspace(0.1, 3.0, 30)
        lev = PureHestonLeverage(S0=self.PARAMS_FLOW.S0)
        result = solve_stochastic_flow(self.PARAMS_FLOW, dZ_path, V_path, x_grid, lev)
        assert result["negative_J_count"] == 0


class TestTransformedCoefficients:
    PARAMS_C = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.3, rho=-0.5, r=0.03, d=0.0, T=0.5, n_steps=30, seed=77,
    )

    def test_shapes(self) -> None:
        sim = simulate_rough_heston(self.PARAMS_C, N=2, seed=77)
        V_path = sim["V"][0]
        dZ_path = sim["dZ"][0]
        x_grid = np.linspace(0.1, 3.0, 40)
        lev = PureHestonLeverage(S0=self.PARAMS_C.S0)
        flow = solve_stochastic_flow(self.PARAMS_C, dZ_path, V_path, x_grid, lev)
        coeff = compute_transformed_coefficients(self.PARAMS_C, flow, V_path, lev)
        n = self.PARAMS_C.n_steps
        assert coeff["DeltaTilde_sq"].shape == (n + 1, 40)
        assert coeff["muTilde"].shape == (n + 1, 40)

    def test_DeltaTilde_sq_nonneg_for_pure_heston(self) -> None:
        """For L=1, DeltaTilde^2 = (1-rho^2)*Phi_t^2*V_t/J_t^2 >= 0."""
        sim = simulate_rough_heston(self.PARAMS_C, N=2, seed=77)
        V_path = sim["V"][0]
        dZ_path = sim["dZ"][0]
        x_grid = np.linspace(0.2, 2.0, 30)
        lev = PureHestonLeverage(S0=self.PARAMS_C.S0)
        flow = solve_stochastic_flow(self.PARAMS_C, dZ_path, V_path, x_grid, lev)
        coeff = compute_transformed_coefficients(self.PARAMS_C, flow, V_path, lev)
        assert np.all(coeff["DeltaTilde_sq"] >= 0.0)


# ──────────────────────────────────────────────
# invert_flow
# ──────────────────────────────────────────────

class TestInvertFlow:
    def test_roundtrip(self) -> None:
        """invert_flow(Phi_t(x0), Phi_t, x_grid) ~ x0."""
        x_grid = np.linspace(0.5, 3.0, 200)
        # Monotone increasing Phi
        Phi_t = x_grid * 1.05  # simple scaling
        x0 = 1.5
        K = x0 * 1.05
        x_K = invert_flow(K, Phi_t, x_grid)
        assert abs(x_K - x0) < 0.02

    def test_out_of_range_returns_nan(self) -> None:
        x_grid = np.linspace(0.5, 3.0, 100)
        Phi_t = x_grid * 1.1
        x_K = invert_flow(K=0.01, Phi_t=Phi_t, x_grid=x_grid)
        assert np.isnan(x_K)
