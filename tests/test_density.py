"""
Tests for src/density.py and src/path_functionals.py.

Validates:
  - Lognormal log-PDF numerical correctness.
  - q_t and p_t positivity, normalization (~1 to tolerance), and shape.
  - Inverse-flow identity: p_t(K) = q_t(K/M_t) / M_t.
  - Transformation identity: q_t(x) = p_t(x*M_t) * M_t.
  - Batch evaluators produce consistent shapes.
  - Path functionals: I_t, Vbar_t, A_t shapes, A_t >= 0.
  - M_t > 0 always (J_t > 0 identity).
  - Small A_t degenerate case returns finite values.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import lognorm

from src.config import RoughHestonParams
from src.simulator import simulate_rough_heston
from src.path_functionals import compute_path_functionals, get_functionals_at_time
from src.density import (
    lognormal_log_pdf,
    eval_q_t,
    eval_p_t,
    eval_q_t_batch,
    eval_p_t_batch,
    eval_p_t_at_strikes,
    eval_q_t_at_inverse_strikes,
    inverse_flow_recovery,
    verify_transformation_identity,
    normalization_error,
    check_positivity,
    compute_first_moment,
    EPSILON,
)


PARAMS = RoughHestonParams(
    S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
    nu=0.5, rho=-0.7, r=0.05, d=0.0, T=1.0, n_steps=200, seed=42,
)


# ──────────────────────────────────────────────
# Lognormal log-PDF
# ──────────────────────────────────────────────

class TestLognormalLogPDF:
    def test_matches_scipy(self) -> None:
        x = np.array([0.5, 1.0, 1.5, 2.0])
        mu, var = 0.1, 0.25
        log_pdf_custom = lognormal_log_pdf(x, mu, var)
        # scipy.stats.lognorm(s=sqrt(var), scale=exp(mu)).logpdf(x)
        log_pdf_scipy = lognorm(s=np.sqrt(var), scale=np.exp(mu)).logpdf(x)
        np.testing.assert_allclose(log_pdf_custom, log_pdf_scipy, atol=1e-10)

    def test_output_shape(self) -> None:
        x = np.linspace(0.1, 3.0, 100)
        out = lognormal_log_pdf(x, 0.0, 0.5)
        assert out.shape == (100,)

    def test_finite_for_positive_x(self) -> None:
        x = np.array([0.01, 0.1, 1.0, 10.0])
        out = lognormal_log_pdf(x, 0.0, 1.0)
        assert np.all(np.isfinite(out))


# ──────────────────────────────────────────────
# Explicit density evaluators
# ──────────────────────────────────────────────

class TestEvalQt:
    def test_positivity(self) -> None:
        x = np.linspace(0.01, 5.0, 300)
        q = eval_q_t(x, A_t=0.1, R_t=0.05, S0=1.0)
        assert np.all(q >= 0.0)

    def test_normalization_approx(self) -> None:
        """Integral of q_t over positive x-grid should be close to 1."""
        x = np.linspace(0.001, 10.0, 10000)
        q = eval_q_t(x, A_t=0.2, R_t=0.02, S0=1.0)
        mass = np.trapz(q, x)
        assert abs(mass - 1.0) < 0.01, f"mass={mass}"

    def test_degenerate_A_returns_zeros(self) -> None:
        """If A_t < EPSILON, density should be treated as degenerate (zero)."""
        x = np.linspace(0.1, 3.0, 100)
        q = eval_q_t(x, A_t=0.0, R_t=0.0, S0=1.0)
        assert np.all(q == 0.0)

    def test_lognormal_first_moment(self) -> None:
        """E[X] = S0 * exp(R_t) for q_t (lognormal mean)."""
        x = np.linspace(0.001, 10.0, 10000)
        A_t, R_t, S0 = 0.3, 0.05, 1.0
        q = eval_q_t(x, A_t=A_t, R_t=R_t, S0=S0)
        moment = np.trapz(x * q, x)
        expected = S0 * np.exp(R_t)
        assert abs(moment - expected) < 0.03, f"moment={moment}, expected={expected}"


class TestEvalPt:
    def test_positivity(self) -> None:
        s = np.linspace(0.01, 5.0, 300)
        p = eval_p_t(s, I_t=0.2, Vbar_t=0.5, A_t=0.15, R_t=0.05, S0=1.0, rho=-0.7)
        assert np.all(p >= 0.0)

    def test_normalization_approx(self) -> None:
        s = np.linspace(0.001, 10.0, 10000)
        p = eval_p_t(s, I_t=0.1, Vbar_t=0.3, A_t=0.09, R_t=0.02, S0=1.0, rho=-0.5)
        mass = np.trapz(p, s)
        assert abs(mass - 1.0) < 0.01, f"mass={mass}"

    def test_degenerate_A_returns_zeros(self) -> None:
        s = np.linspace(0.1, 3.0, 100)
        p = eval_p_t(s, I_t=0.1, Vbar_t=0.0, A_t=0.0, R_t=0.0, S0=1.0, rho=-0.5)
        assert np.all(p == 0.0)


# ──────────────────────────────────────────────
# Inverse-flow identity: p_t(K) = q_t(K/M_t) / M_t
# ──────────────────────────────────────────────

class TestInverseFlowRecovery:
    def test_recovery_exact(self) -> None:
        """Recovered p_t(K) should equal direct p_t(K) to machine precision."""
        I_t, Vbar_t = 0.3, 0.5
        rho = -0.7
        A_t = (1 - rho ** 2) * Vbar_t
        R_t = 0.05
        S0 = 1.0
        M_t = np.exp(rho * I_t - 0.5 * rho ** 2 * Vbar_t)

        K = np.linspace(0.3, 2.5, 200)
        p_direct = eval_p_t(K, I_t, Vbar_t, A_t, R_t, S0, rho)
        p_rec = inverse_flow_recovery(K, M_t, A_t, R_t, S0)

        np.testing.assert_allclose(p_direct, p_rec, rtol=1e-10, atol=1e-12)

    def test_multiple_paths(self) -> None:
        """Verify inverse-flow recovery for N=50 simulated paths."""
        sim = simulate_rough_heston(PARAMS, N=50, seed=77)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        ft = get_functionals_at_time(fcts, t_idx=100)

        K = np.array([0.8, 1.0, 1.2]) * PARAMS.S0

        for i in range(min(10, 50)):
            I0 = float(ft["I_t"][i])
            Vbar0 = float(ft["Vbar_t"][i])
            A0 = float(ft["A_t"][i])
            M0 = float(ft["M_t"][i])
            R_t = float(ft["R_t"])

            p_dir = eval_p_t(K, I0, Vbar0, A0, R_t, PARAMS.S0, PARAMS.rho)
            p_rec = inverse_flow_recovery(K, M0, A0, R_t, PARAMS.S0)
            np.testing.assert_allclose(p_dir, p_rec, rtol=1e-8)


# ──────────────────────────────────────────────
# Transformation identity
# ──────────────────────────────────────────────

class TestTransformationIdentity:
    def test_identity_holds(self) -> None:
        """q_t(x) = p_t(x*M_t) * M_t to machine precision."""
        I_t, Vbar_t = 0.2, 0.4
        rho = -0.6
        A_t = (1 - rho ** 2) * Vbar_t
        R_t = 0.03
        S0 = 1.0
        M_t = np.exp(rho * I_t - 0.5 * rho ** 2 * Vbar_t)

        x = np.linspace(0.1, 3.0, 200)
        result = verify_transformation_identity(x, M_t, I_t, Vbar_t, A_t, R_t, S0, rho)
        assert result["max_rel_error"] < 1e-8, (
            f"max_rel_error={result['max_rel_error']:.2e}"
        )


# ──────────────────────────────────────────────
# Batch evaluators
# ──────────────────────────────────────────────

class TestBatchEvaluators:
    def test_q_batch_shape(self) -> None:
        N, n_x = 10, 150
        A_t = np.abs(np.random.default_rng(0).standard_normal(N)) * 0.1 + 0.05
        x = np.linspace(0.1, 3.0, n_x)
        q = eval_q_t_batch(x, A_t, R_t=0.02, S0=1.0)
        assert q.shape == (N, n_x)

    def test_p_batch_shape(self) -> None:
        N, n_s = 10, 150
        rng = np.random.default_rng(1)
        I_t = rng.standard_normal(N) * 0.1
        Vbar_t = np.abs(rng.standard_normal(N)) * 0.2 + 0.1
        rho = -0.7
        A_t = (1 - rho ** 2) * Vbar_t
        s = np.linspace(0.1, 3.0, n_s)
        p = eval_p_t_batch(s, I_t, Vbar_t, A_t, R_t=0.02, S0=1.0, rho=rho)
        assert p.shape == (N, n_s)

    def test_q_batch_nonnegative(self) -> None:
        A_t = np.array([0.1, 0.2, 0.3])
        x = np.linspace(0.01, 5.0, 100)
        q = eval_q_t_batch(x, A_t, 0.0, 1.0)
        assert np.all(q >= 0.0)

    def test_p_at_strikes_shape(self) -> None:
        N, n_K = 8, 5
        rng = np.random.default_rng(2)
        I_t = rng.standard_normal(N) * 0.05
        Vbar_t = np.abs(rng.standard_normal(N)) * 0.1 + 0.05
        rho = -0.5
        A_t = (1 - rho ** 2) * Vbar_t
        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        p = eval_p_t_at_strikes(K, I_t, Vbar_t, A_t, R_t=0.0, S0=1.0, rho=rho)
        assert p.shape == (N, n_K)

    def test_q_at_inverse_strikes_shape(self) -> None:
        N, n_K = 8, 5
        rng = np.random.default_rng(3)
        M_t = np.exp(rng.standard_normal(N) * 0.1)
        A_t = np.abs(rng.standard_normal(N)) * 0.1 + 0.05
        K = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        q = eval_q_t_at_inverse_strikes(K, M_t, A_t, R_t=0.0, S0=1.0)
        assert q.shape == (N, n_K)


# ──────────────────────────────────────────────
# Normalization and positivity checkers
# ──────────────────────────────────────────────

class TestNormalizationAndPositivity:
    def test_normalization_error_perfect(self) -> None:
        x = np.linspace(0.01, 10.0, 10000)
        q = eval_q_t(x, A_t=0.25, R_t=0.05, S0=1.0)
        err = normalization_error(q, x)
        assert err < 0.02

    def test_check_positivity_all_positive(self) -> None:
        x = np.linspace(0.1, 3.0, 100)
        q = eval_q_t(x, A_t=0.2, R_t=0.0, S0=1.0)
        result = check_positivity(q)
        assert result["all_positive"] is True
        assert result["n_negative"] == 0

    def test_check_positivity_with_negatives(self) -> None:
        arr = np.array([1.0, -0.1, 2.0, -0.5])
        result = check_positivity(arr)
        assert result["all_positive"] is False
        assert result["n_negative"] == 2

    def test_first_moment_nonnegative(self) -> None:
        x = np.linspace(0.01, 5.0, 1000)
        q = eval_q_t(x, A_t=0.2, R_t=0.0, S0=1.0)
        m1 = compute_first_moment(q, x)
        assert m1 > 0.0


# ──────────────────────────────────────────────
# Path functionals
# ──────────────────────────────────────────────

class TestPathFunctionals:
    def test_shapes(self) -> None:
        N, n = 10, PARAMS.n_steps
        sim = simulate_rough_heston(PARAMS, N, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        assert fcts["I_t"].shape == (N, n + 1)
        assert fcts["Vbar_t"].shape == (N, n + 1)
        assert fcts["A_t"].shape == (N, n + 1)
        assert fcts["R_t"].shape == (n + 1,)
        assert fcts["M_t"].shape == (N, n + 1)

    def test_initial_conditions(self) -> None:
        sim = simulate_rough_heston(PARAMS, N=5, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        assert np.all(fcts["I_t"][:, 0] == pytest.approx(0.0))
        assert np.all(fcts["Vbar_t"][:, 0] == pytest.approx(0.0))
        assert np.all(fcts["A_t"][:, 0] == pytest.approx(0.0))
        assert fcts["R_t"][0] == pytest.approx(0.0)
        assert np.all(fcts["M_t"][:, 0] == pytest.approx(1.0))

    def test_A_t_nonnegative(self) -> None:
        """A_t = (1-rho^2)*Vbar_t >= 0."""
        sim = simulate_rough_heston(PARAMS, N=5, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        assert np.all(fcts["A_t"] >= 0.0)

    def test_M_t_positive(self) -> None:
        """M_t = exp(...) > 0 always."""
        sim = simulate_rough_heston(PARAMS, N=10, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        assert np.all(fcts["M_t"] > 0.0)

    def test_Vbar_t_nondecreasing(self) -> None:
        """Vbar_t is cumulative integral of V >= 0, so must be nondecreasing."""
        sim = simulate_rough_heston(PARAMS, N=3, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        diffs = np.diff(fcts["Vbar_t"], axis=1)
        assert np.all(diffs >= -1e-12)  # small tolerance for floating-point

    def test_R_t_equals_rate_times_time(self) -> None:
        """R_t = (r-d)*t for constant rates."""
        sim = simulate_rough_heston(PARAMS, N=2, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        expected = (PARAMS.r - PARAMS.d) * PARAMS.time_grid
        np.testing.assert_allclose(fcts["R_t"], expected, rtol=1e-10)

    def test_get_functionals_at_time(self) -> None:
        sim = simulate_rough_heston(PARAMS, N=5, seed=0)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], PARAMS)
        t_idx = 50
        ft = get_functionals_at_time(fcts, t_idx)
        assert ft["I_t"].shape == (5,)
        assert ft["Vbar_t"].shape == (5,)
        assert ft["A_t"].shape == (5,)
        assert isinstance(ft["R_t"], float) or ft["R_t"].shape == ()
        assert ft["M_t"].shape == (5,)
        np.testing.assert_array_equal(ft["I_t"], fcts["I_t"][:, t_idx])
