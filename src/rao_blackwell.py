"""
Rao–Blackwellized estimator for Theta(t,K) = E[V_t | S_t = K] in pure rough Heston.

In pure rough Heston with f(V) = sqrt(V), the calibration ratio target is:
  Theta(t,K) = E[f^2(V_t) | S_t = K] = E[V_t | S_t = K]

Two equivalent finite-sample estimators (paper equations 30 and 31):

Transformed-coordinate (eq. 30):
  Theta_hat_trans(t,K) = sum_i V_t^i * q_t^i(K/M_t^i) / M_t^i
                         -------------------------------------------------
                         sum_i q_t^i(K/M_t^i) / M_t^i

Original-coordinate (eq. 31):
  Theta_hat_orig(t,K) = sum_i V_t^i * p_t^i(K)
                        --------------------------
                        sum_i p_t^i(K)

These should be numerically identical since p_t^i(K) = q_t^i(K/M_t^i) / M_t^i.

Custom — Context7 found no library equivalent for Rao-Blackwellized LSRV estimator.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import RoughHestonParams
from .density import eval_p_t_at_strikes, eval_q_t_at_inverse_strikes, EPSILON
from .path_functionals import compute_path_functionals
from .simulator import simulate_rough_heston


# ──────────────────────────────────────────────
# Core estimator functions
# ──────────────────────────────────────────────

def estimate_theta(
    V_t: np.ndarray,
    p_density: np.ndarray,
    stability_threshold: float = 1e-10,
) -> tuple[float, float, bool]:
    """Compute Rao–Blackwellized conditional variance estimator at a single strike.

    Theta_hat = sum(V_t^i * p_t^i) / sum(p_t^i)

    Parameters
    ----------
    V_t : np.ndarray, shape (N,)
        Variance values at time t for N outer paths.
    p_density : np.ndarray, shape (N,)
        p_t^i(K) or q_t^i(K/M_t^i)/M_t^i for each path.
    stability_threshold : float
        Minimum denominator value for stable estimation.

    Returns
    -------
    theta_hat : float
    denominator : float
    is_stable : bool
    """
    numerator = float(np.sum(V_t * p_density))
    denominator = float(np.sum(p_density))
    is_stable = denominator > stability_threshold
    theta_hat = numerator / denominator if is_stable else np.nan
    return theta_hat, denominator, is_stable


def estimate_theta_surface(
    V_t: np.ndarray,
    p_densities: np.ndarray,
    stability_threshold: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Rao–Blackwellized estimator at multiple strikes simultaneously.

    Parameters
    ----------
    V_t : np.ndarray, shape (N,)
    p_densities : np.ndarray, shape (N, n_strikes)
        p_t^i(K_l) for all paths i and strikes l.
    stability_threshold : float

    Returns
    -------
    theta_hat : np.ndarray, shape (n_strikes,)
    denominators : np.ndarray, shape (n_strikes,)
    is_stable : np.ndarray of bool, shape (n_strikes,)
    """
    # numerator[l] = sum_i V_t^i * p_t^i(K_l)
    numerator = V_t @ p_densities         # (n_strikes,)
    denominator = np.sum(p_densities, axis=0)  # (n_strikes,)
    is_stable = denominator > stability_threshold
    theta_hat = np.where(is_stable, numerator / np.maximum(denominator, EPSILON), np.nan)
    return theta_hat, denominator, is_stable


def verify_estimator_equivalence(
    V_t: np.ndarray,
    q_densities: np.ndarray,
    J_t: np.ndarray,
    p_densities: np.ndarray,
    stability_threshold: float = 1e-10,
) -> dict:
    """Verify that equations (30) and (31) produce identical estimates.

    Equation (30): uses q_t^i(K/M_t^i) / J_t^i  (where J_t^i = M_t^i)
    Equation (31): uses p_t^i(K)

    Since p_t^i(K) = q_t^i(K/M_t^i) / M_t^i = q_t^i(K/M_t^i) / J_t^i,
    both should be numerically equal.

    Parameters
    ----------
    V_t : np.ndarray, shape (N,)
    q_densities : np.ndarray, shape (N, n_strikes)
        q_t^i(K/M_t^i) for each path and strike.
    J_t : np.ndarray, shape (N,)
        J_t^i = M_t^i for each path.
    p_densities : np.ndarray, shape (N, n_strikes)
        p_t^i(K) for each path and strike.
    stability_threshold : float

    Returns
    -------
    dict with:
        'theta_trans'   : np.ndarray, shape (n_strikes,) from eq. (30)
        'theta_orig'    : np.ndarray, shape (n_strikes,) from eq. (31)
        'abs_diff'      : np.ndarray, shape (n_strikes,)
        'rel_diff'      : np.ndarray, shape (n_strikes,)
        'max_abs_diff'  : float
        'max_rel_diff'  : float
        'is_stable'     : np.ndarray of bool
    """
    # Eq. (30): q(K/M)/J = q(K/M)/M
    w_trans = q_densities / np.maximum(J_t[:, np.newaxis], EPSILON)  # (N, n_strikes)
    theta_trans, denom_trans, stable_trans = estimate_theta_surface(
        V_t, w_trans, stability_threshold
    )

    # Eq. (31): p(K)
    theta_orig, denom_orig, stable_orig = estimate_theta_surface(
        V_t, p_densities, stability_threshold
    )

    is_stable = stable_trans & stable_orig
    abs_diff = np.where(is_stable, np.abs(theta_trans - theta_orig), np.nan)
    denom_check = np.maximum(np.abs(theta_trans), EPSILON)
    rel_diff = np.where(is_stable, abs_diff / denom_check, np.nan)

    return {
        "theta_trans": theta_trans,
        "theta_orig": theta_orig,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "max_abs_diff": float(np.nanmax(abs_diff)) if np.any(is_stable) else np.nan,
        "max_rel_diff": float(np.nanmax(rel_diff)) if np.any(is_stable) else np.nan,
        "is_stable": is_stable,
    }


# ──────────────────────────────────────────────
# Convergence study
# ──────────────────────────────────────────────

def _run_single_estimate(
    params: RoughHestonParams,
    N: int,
    t_idx: int,
    K_values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one Rao–Blackwellized estimation for given N and seed.

    Returns (theta_hat, is_stable) each of shape (n_strikes,).
    """
    sim = simulate_rough_heston(params, N, seed=seed)
    fcts = compute_path_functionals(sim["V"], sim["dZ"], params)

    I_t = fcts["I_t"][:, t_idx]
    Vbar_t = fcts["Vbar_t"][:, t_idx]
    A_t = fcts["A_t"][:, t_idx]
    M_t = fcts["M_t"][:, t_idx]
    R_t = fcts["R_t"][t_idx]
    V_t = sim["V"][:, t_idx]

    p_dens = eval_p_t_at_strikes(K_values, I_t, Vbar_t, A_t, R_t, params.S0, params.rho)
    theta_hat, denom, is_stable = estimate_theta_surface(V_t, p_dens)
    return theta_hat, is_stable


def convergence_study(
    params: RoughHestonParams,
    N_values: list[int],
    t_idx: int,
    K_values: np.ndarray,
    N_ref: int = 20000,
    n_runs: int = 5,
    seed: int = 42,
) -> dict:
    """Study convergence of Theta_hat(t,K) as outer sample size N increases.

    Parameters
    ----------
    params : RoughHestonParams
    N_values : list of int
        Sequence of N to test, e.g. [100, 300, 1000, 3000].
    t_idx : int
        Time-step index.
    K_values : np.ndarray, shape (n_strikes,)
    N_ref : int
        Reference estimator sample size.
    n_runs : int
        Number of independent seeds per N (for standard-error estimate).
    seed : int
        Base seed.

    Returns
    -------
    dict with keys:
        'N_values', 'K_values', 't_idx',
        'theta_ref'      : np.ndarray, shape (n_strikes,)
        'mean_error'     : np.ndarray, shape (len(N_values), n_strikes)
        'std_error'      : np.ndarray, shape (len(N_values), n_strikes)
        'rmse'           : np.ndarray, shape (len(N_values), n_strikes)
        'stable_fraction': np.ndarray, shape (len(N_values), n_strikes)
    """
    rng = np.random.default_rng(seed)

    # High-precision reference
    ref_seed = int(rng.integers(0, 2 ** 31))
    theta_ref, _ = _run_single_estimate(params, N_ref, t_idx, K_values, ref_seed)

    n_K = len(K_values)
    mean_error = np.zeros((len(N_values), n_K))
    std_error = np.zeros((len(N_values), n_K))
    rmse = np.zeros((len(N_values), n_K))
    stable_fraction = np.zeros((len(N_values), n_K))

    for i, N in enumerate(N_values):
        run_estimates = []
        run_stable = []
        for run in range(n_runs):
            run_seed = int(rng.integers(0, 2 ** 31))
            theta_hat, is_stable = _run_single_estimate(
                params, N, t_idx, K_values, run_seed
            )
            run_estimates.append(theta_hat)
            run_stable.append(is_stable.astype(float))

        run_estimates = np.array(run_estimates)  # (n_runs, n_K)
        run_stable = np.array(run_stable)

        errors = run_estimates - theta_ref[np.newaxis, :]
        mean_error[i] = np.nanmean(errors, axis=0)
        std_error[i] = np.nanstd(errors, axis=0)
        rmse[i] = np.sqrt(np.nanmean(errors ** 2, axis=0))
        stable_fraction[i] = np.mean(run_stable, axis=0)

    return {
        "N_values": N_values,
        "K_values": K_values,
        "t_idx": t_idx,
        "theta_ref": theta_ref,
        "mean_error": mean_error,
        "std_error": std_error,
        "rmse": rmse,
        "stable_fraction": stable_fraction,
    }
