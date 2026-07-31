"""
Path functional calculator for pure rough Heston.

Computes the following for each simulated path:
  I_t   = int_0^t sqrt(V_u) dZ_u         (Ito stochastic integral, left-point)
  Vbar_t = int_0^t V_u du                 (Riemann integral, left-point)
  A_t   = (1 - rho^2) * Vbar_t
  R_t   = int_0^t (r_u - d_u) du          (deterministic, constant rates)
  M_t   = exp(rho * I_t - 0.5 * rho^2 * Vbar_t)

These path functionals define the stochastic flow and conditional densities
as per equations (33)–(39) in the paper.
"""
from __future__ import annotations

import numpy as np

from .config import RoughHestonParams


def compute_path_functionals(
    V: np.ndarray,
    dZ: np.ndarray,
    params: RoughHestonParams,
) -> dict:
    """Compute path functionals for all N outer paths at all time steps.

    Quadrature: left-point Riemann / Ito sum.  For step k (k = 0,...,n-1):
      Contribution to I_{t_{k+1}} relative to I_{t_k}:  sqrt(V_{t_k}) * dZ_k
      Contribution to Vbar_{t_{k+1}} relative to Vbar_{t_k}: V_{t_k} * dt

    Parameters
    ----------
    V : np.ndarray, shape (N, n_steps+1)
        Variance paths (full truncated, so V >= 0).
    dZ : np.ndarray, shape (N, n_steps)
        Common noise increments.
    params : RoughHestonParams

    Returns
    -------
    dict with keys:
        'I_t'    : np.ndarray, shape (N, n_steps+1)   — stochastic integral
        'Vbar_t' : np.ndarray, shape (N, n_steps+1)   — time integral of V
        'A_t'    : np.ndarray, shape (N, n_steps+1)   — (1-rho^2)*Vbar_t
        'R_t'    : np.ndarray, shape (n_steps+1,)     — deterministic rate integral
        'M_t'    : np.ndarray, shape (N, n_steps+1)   — exp(rho*I_t - 0.5*rho^2*Vbar_t)
    """
    N, n_plus_1 = V.shape
    n = n_plus_1 - 1
    dt = params.dt
    rho = params.rho
    r_minus_d = params.r - params.d

    # I_t: left-point Ito integral  — cumulative sum of sqrt(V_{t_k}) * dZ_k
    # sqrt of V (already >= 0 due to full truncation)
    sqrt_V = np.sqrt(np.maximum(V[:, :-1], 0.0))          # (N, n)
    dI = sqrt_V * dZ                                        # (N, n)
    I_t = np.zeros((N, n + 1))
    I_t[:, 1:] = np.cumsum(dI, axis=1)

    # Vbar_t: left-point Riemann sum of V_{t_k} * dt
    dVbar = V[:, :-1] * dt                                  # (N, n)
    Vbar_t = np.zeros((N, n + 1))
    Vbar_t[:, 1:] = np.cumsum(dVbar, axis=1)

    # A_t = (1 - rho^2) * Vbar_t
    A_t = (1.0 - rho ** 2) * Vbar_t                        # (N, n+1)

    # R_t = (r - d) * t  (constant rates)
    time_grid = params.time_grid                            # (n+1,)
    R_t = r_minus_d * time_grid                            # (n+1,)

    # M_t = exp(rho * I_t - 0.5 * rho^2 * Vbar_t)
    log_M = rho * I_t - 0.5 * rho ** 2 * Vbar_t           # (N, n+1)
    M_t = np.exp(log_M)                                    # (N, n+1)

    return {
        "I_t": I_t,
        "Vbar_t": Vbar_t,
        "A_t": A_t,
        "R_t": R_t,
        "M_t": M_t,
    }


def get_functionals_at_time(
    functionals: dict,
    t_idx: int,
) -> dict:
    """Extract path functional arrays at a single time index.

    Parameters
    ----------
    functionals : dict
        Output of compute_path_functionals.
    t_idx : int
        Time index (column index).

    Returns
    -------
    dict with scalar (per-path) arrays for I_t, Vbar_t, A_t, R_t, M_t.
    """
    return {
        "I_t": functionals["I_t"][:, t_idx],     # (N,)
        "Vbar_t": functionals["Vbar_t"][:, t_idx],  # (N,)
        "A_t": functionals["A_t"][:, t_idx],     # (N,)
        "R_t": functionals["R_t"][t_idx],         # scalar
        "M_t": functionals["M_t"][:, t_idx],     # (N,)
    }
