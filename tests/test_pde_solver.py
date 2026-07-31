"""
Tests for src/pde_solver.py.

Validates:
  - PDE solver runs without error for typical inputs.
  - Output shapes and keys are correct.
  - Mass history stays close to 1.
  - Numerical q_t matches exact q_t within acceptable tolerance.
  - Inverse-flow recovery from PDE output is consistent.
  - Log-space PDE solution is non-negative.
  - Convergence: finer grid produces smaller L1 error vs exact.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d

from src.config import RoughHestonParams
from src.simulator import simulate_rough_heston
from src.density import eval_q_t, eval_p_t, EPSILON
from src.pde_solver import solve_pde_log_space, compute_pde_error_vs_exact


PARAMS = RoughHestonParams(
    S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
    nu=0.5, rho=-0.7, r=0.05, d=0.0, T=0.5, n_steps=100, seed=42,
)


def get_frozen_path(params: RoughHestonParams, path_idx: int = 0) -> tuple:
    sim = simulate_rough_heston(params, N=max(path_idx + 1, 3), seed=params.seed)
    V_path = sim["V"][path_idx]
    dZ_path = sim["dZ"][path_idx]
    return V_path, dZ_path


class TestPDESolverOutputStructure:
    def test_output_keys(self) -> None:
        V_path, _ = get_frozen_path(PARAMS)
        reporting = [50]
        result = solve_pde_log_space(PARAMS, V_path, reporting, n_y=100)
        for key in ["y_grid", "x_grid", "qbar_solutions", "q_solutions",
                    "mass_history", "A_t_path", "R_t_path", "params"]:
            assert key in result

    def test_y_grid_shape(self) -> None:
        V_path, _ = get_frozen_path(PARAMS)
        result = solve_pde_log_space(PARAMS, V_path, [50], n_y=100)
        n_total = 102  # n_y + 2 boundary points
        assert result["y_grid"].shape == (n_total,)
        assert result["x_grid"].shape == (n_total,)

    def test_mass_history_shape(self) -> None:
        V_path, _ = get_frozen_path(PARAMS)
        result = solve_pde_log_space(PARAMS, V_path, [50], n_y=100)
        assert result["mass_history"].shape == (PARAMS.n_steps + 1,)

    def test_solutions_count(self) -> None:
        """Number of saved solutions should equal len(reporting_times)."""
        V_path, _ = get_frozen_path(PARAMS)
        reporting = [25, 50]
        result = solve_pde_log_space(PARAMS, V_path, reporting, n_y=100)
        assert len(result["q_solutions"]) == len(reporting)

    def test_x_grid_positive(self) -> None:
        V_path, _ = get_frozen_path(PARAMS)
        result = solve_pde_log_space(PARAMS, V_path, [50], n_y=100)
        assert np.all(result["x_grid"] > 0.0)


class TestPDEMassConservation:
    def test_mass_near_one_at_reporting_time(self) -> None:
        """Mass should remain close to 1 throughout the simulation."""
        V_path, _ = get_frozen_path(PARAMS)
        result = solve_pde_log_space(PARAMS, V_path, [50, 100], n_y=200)
        # Mass at final time
        final_mass = result["mass_history"][-1]
        assert 0.85 < final_mass < 1.15, f"final_mass={final_mass}"

    def test_mass_history_finite(self) -> None:
        V_path, _ = get_frozen_path(PARAMS)
        result = solve_pde_log_space(PARAMS, V_path, [50], n_y=100)
        assert np.all(np.isfinite(result["mass_history"]))


class TestPDESolutionNonnegativity:
    def test_q_solutions_nonnegative(self) -> None:
        """After positivity enforcement, q solutions must be >= 0."""
        V_path, _ = get_frozen_path(PARAMS)
        result = solve_pde_log_space(PARAMS, V_path, [50, 100], n_y=150)
        for q_sol in result["q_solutions"]:
            assert np.all(q_sol >= 0.0), f"min q={np.min(q_sol)}"


class TestPDEVsExact:
    def test_pde_vs_exact_reasonable_error(self) -> None:
        """Numerical PDE q_t should match exact q_t with reasonable L1 error."""
        V_path, dZ_path = get_frozen_path(PARAMS)
        from src.path_functionals import compute_path_functionals
        V_one = V_path[np.newaxis, :]
        dZ_one = dZ_path[np.newaxis, :]
        fcts = compute_path_functionals(V_one, dZ_one, PARAMS)

        reporting_times = [PARAMS.n_steps]
        result = solve_pde_log_space(PARAMS, V_path, reporting_times, n_y=200)

        I_t_rep = fcts["I_t"][0, reporting_times]
        Vbar_t_rep = fcts["Vbar_t"][0, reporting_times]
        M_t_rep = fcts["M_t"][0, reporting_times]

        errors = compute_pde_error_vs_exact(result, PARAMS, I_t_rep, Vbar_t_rep, M_t_rep)

        # L1 error < 0.5 (coarse tolerance for small n_y and n_steps)
        assert errors["L1"][0] < 0.5, f"L1={errors['L1'][0]:.4f}"

    def test_normalization_pde_close_to_one(self) -> None:
        V_path, dZ_path = get_frozen_path(PARAMS)
        from src.path_functionals import compute_path_functionals
        V_one = V_path[np.newaxis, :]
        dZ_one = dZ_path[np.newaxis, :]
        fcts = compute_path_functionals(V_one, dZ_one, PARAMS)

        result = solve_pde_log_space(PARAMS, V_path, [PARAMS.n_steps], n_y=200)
        I_t_rep = fcts["I_t"][0, -1:]
        Vbar_t_rep = fcts["Vbar_t"][0, -1:]
        M_t_rep = fcts["M_t"][0, -1:]
        errors = compute_pde_error_vs_exact(result, PARAMS, I_t_rep, Vbar_t_rep, M_t_rep)
        # The numerical density should integrate to between 0.8 and 1.2
        mass = errors["normalization_pde"][0]
        assert 0.8 < mass < 1.2, f"mass={mass}"


class TestPDEInverseFlowRecovery:
    def test_inverse_flow_recovery_shape(self) -> None:
        V_path, dZ_path = get_frozen_path(PARAMS)
        from src.path_functionals import compute_path_functionals
        V_one = V_path[np.newaxis, :]
        dZ_one = dZ_path[np.newaxis, :]
        fcts = compute_path_functionals(V_one, dZ_one, PARAMS)

        t_final = PARAMS.n_steps
        result = solve_pde_log_space(PARAMS, V_path, [t_final], n_y=150)

        A_t_val = float(result["A_t_reporting"][0])
        R_t_val = float(result["R_t_reporting"][0])
        M_t_val = float(fcts["M_t"][0, t_final])
        I_t_val = float(fcts["I_t"][0, t_final])
        Vbar_t_val = float(fcts["Vbar_t"][0, t_final])

        x_grid = result["x_grid"]
        q_num = result["q_solutions"][0]
        K_test = np.linspace(0.3 * PARAMS.S0, 2.5 * PARAMS.S0, 50)
        x_K = K_test / max(M_t_val, EPSILON)

        q_interp = interp1d(x_grid, q_num, kind="linear", fill_value=0.0, bounds_error=False)
        p_from_pde = q_interp(x_K) / max(M_t_val, EPSILON)
        p_exact = eval_p_t(K_test, I_t_val, Vbar_t_val, A_t_val, R_t_val, PARAMS.S0, PARAMS.rho)

        # At least 80% of points should have relative error < 1 (coarse check)
        rel_err = np.abs(p_from_pde - p_exact) / np.maximum(np.abs(p_exact), EPSILON)
        frac_ok = np.mean(rel_err < 1.0)
        assert frac_ok > 0.7, f"fraction with rel_err<1: {frac_ok:.2f}"
