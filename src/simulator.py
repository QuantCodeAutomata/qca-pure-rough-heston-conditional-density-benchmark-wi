"""
Rough Heston path simulator using Euler–Maruyama discretization
of the stochastic Volterra equation.

Variance process (Volterra–Heston):
  V_t = V_0 + int_0^t K(t-s) [kappa*(theta-V_s) ds + nu*sqrt(max(V_s,0)) dZ_s]
  K(u) = u^(H - 1/2) / Gamma(H + 1/2),   H in (0, 1/2)

Discretization:
  V_{t_n} ≈ V_0 + sum_{k=0}^{n-1} K(t_n - t_k) * [kappa*(theta - V_{t_k})*dt
                                                    + nu*sqrt(max(V_{t_k},0)) * dZ_k]

Positivity: full truncation  V_{t_k} -> max(V_{t_k}, 0) under sqrt.
Complexity: O(N * n_steps^2 / 2) — vectorised over outer paths but sequential in time.
For n_steps = 500 and N = 200 this is approximately 25 M kernel evaluations, feasible in numpy.

Custom — Context7 found no library equivalent for rough Volterra-Heston discrete simulation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gamma as gamma_func

from .config import RoughHestonParams


def _build_kernel_matrix(H: float, time_grid: np.ndarray) -> np.ndarray:
    """Precompute lower-triangular Volterra kernel matrix.

    K[n, k] = (t_n - t_k)^(H-0.5) / Gamma(H+0.5) for n > k, else 0.

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 0.5).
    time_grid : np.ndarray
        Uniform time grid of shape (n_steps+1,).

    Returns
    -------
    K : np.ndarray of shape (n_steps, n_steps)
        K[i, j] gives the kernel weight for the contribution at time j
        to the integral up to time t_{i+1} (0-indexed for the increment row).
    """
    n = len(time_grid) - 1  # number of steps
    gH = gamma_func(H + 0.5)
    # Build (n, n) matrix; we need K[step_i, step_j] for step_i > step_j
    # where step_i corresponds to t_{step_i+1} and step_j to t_{step_j}
    # Actually we compute V[:, i] using past steps 0..i-1
    # Let i = 1..n (index in V), j = 0..i-1 (index in V)
    # kernel(i, j) = (t_grid[i] - t_grid[j])^(H-0.5) / gH
    rows = np.arange(1, n + 1)          # shape (n,)
    cols = np.arange(0, n)              # shape (n,)
    t_diff = time_grid[rows[:, None]] - time_grid[cols[None, :]]  # (n, n)
    # Only lower-triangular part is valid (i > j, i.e., t_diff > 0)
    K = np.where(t_diff > 0.0, t_diff ** (H - 0.5) / gH, 0.0)
    return K  # shape (n, n)


def simulate_rough_heston(
    params: RoughHestonParams,
    N: int,
    seed: Optional[int] = None,
) -> dict:
    """Simulate N iid common-environment paths (Z^i, V^i) for pure rough Heston.

    Uses Euler–Maruyama discretization of the Volterra integral with full truncation
    (V -> max(V, 0)) to preserve positivity under the square root.

    Parameters
    ----------
    params : RoughHestonParams
        Model parameters (H, kappa, theta, nu, V0, rho, T, n_steps).
    N : int
        Number of outer paths.
    seed : int, optional
        Override random seed.  Falls back to params.seed.

    Returns
    -------
    dict with keys:
        'V'                   : np.ndarray, shape (N, n_steps+1)   — variance paths
        'dZ'                  : np.ndarray, shape (N, n_steps)     — Z increments
        'Z'                   : np.ndarray, shape (N, n_steps+1)   — cumulative Z path
        'time_grid'           : np.ndarray, shape (n_steps+1,)
        'negative_V_fraction' : float — fraction of raw V values < 0 before truncation
        'params'              : RoughHestonParams
        'scheme'              : str — discretization scheme label
    """
    rng = np.random.default_rng(seed if seed is not None else params.seed)

    n = params.n_steps
    dt = params.dt
    time_grid = params.time_grid  # shape (n+1,)

    H = params.H
    kappa = params.kappa
    theta = params.theta
    nu = params.nu
    V0 = params.V0

    # Precompute full kernel matrix K of shape (n, n)
    # K[i-1, j] corresponds to the weight for step j contributing to V[:,i]
    K = _build_kernel_matrix(H, time_grid)  # (n, n), lower-triangular

    # Generate all Z increments: shape (N, n)
    dZ = rng.standard_normal((N, n)) * np.sqrt(dt)
    Z = np.zeros((N, n + 1))
    Z[:, 1:] = np.cumsum(dZ, axis=1)

    # Simulate variance paths
    V = np.zeros((N, n + 1))
    V[:, 0] = V0

    neg_V_count = 0
    total_V_computed = 0

    for i in range(1, n + 1):
        # Past V and dZ values: columns 0..i-1
        V_past = V[:, :i]                            # (N, i)
        V_past_pos = np.maximum(V_past, 0.0)         # full truncation for sqrt
        dZ_past = dZ[:, :i]                          # (N, i)

        # Kernel weights for this step (row i-1 of K, first i columns)
        kw = K[i - 1, :i]                            # (i,)

        # Drift integral: sum_j K[i-1,j] * kappa*(theta - V_j) * dt
        drift = np.dot(kappa * (theta - V_past) * dt, kw)  # (N,)

        # Diffusion integral: sum_j K[i-1,j] * nu * sqrt(V_j) * dZ_j
        diffusion = np.dot(nu * np.sqrt(V_past_pos) * dZ_past, kw)  # (N,)

        V_raw = V0 + drift + diffusion
        neg_V_count += int(np.sum(V_raw < 0.0))
        total_V_computed += N

        V[:, i] = np.maximum(V_raw, 0.0)

    negative_V_fraction = neg_V_count / total_V_computed if total_V_computed > 0 else 0.0

    return {
        "V": V,
        "dZ": dZ,
        "Z": Z,
        "time_grid": time_grid,
        "negative_V_fraction": negative_V_fraction,
        "params": params,
        "scheme": "Euler-Maruyama full-truncation Volterra",
    }


def simulate_idiosyncratic_paths(
    params: RoughHestonParams,
    frozen_V: np.ndarray,
    frozen_dZ: np.ndarray,
    M: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simulate M iid idiosyncratic spot paths conditional on a frozen (V, Z) path.

    Given frozen variance and common noise, the conditional spot process is:
      dS_t = (r-d) S_t dt + S_t sqrt(V_t) [rho dZ_t + sqrt(1-rho^2) d_zeta_t]
    Solved by:
      S_t = S_0 exp(R_t + rho I_t + sqrt(1-rho^2) zeta_t - 0.5 Vbar_t)
    where zeta_t = int_0^t sqrt(V_u) d_zeta_u ~ N(0, (1-rho^2) Vbar_t) conditionally.

    Parameters
    ----------
    params : RoughHestonParams
    frozen_V : np.ndarray, shape (n_steps+1,)
        Frozen variance path.
    frozen_dZ : np.ndarray, shape (n_steps,)
        Frozen Z increments.
    M : int
        Number of idiosyncratic paths.
    seed : int, optional

    Returns
    -------
    S : np.ndarray, shape (M, n_steps+1)
        Simulated spot paths.
    """
    rng = np.random.default_rng(seed)
    n = params.n_steps
    dt = params.dt
    time_grid = params.time_grid
    rho = params.rho
    S0 = params.S0
    r, d = params.r, params.d

    # Generate M idiosyncratic Brownian increments d_zeta: (M, n)
    d_zeta = rng.standard_normal((M, n)) * np.sqrt(dt)

    # Build S paths via Euler–Maruyama on log S
    log_S = np.zeros((M, n + 1))
    log_S[:, 0] = np.log(S0)

    for k in range(n):
        v_k = np.maximum(frozen_V[k], 0.0)
        sv = np.sqrt(v_k)
        r_d = r - d
        # Euler step in log S:
        # d log S = (r-d - 0.5 V_t) dt + sqrt(V_t) [rho dZ + sqrt(1-rho^2) d_zeta]
        dlog = (
            (r_d - 0.5 * v_k) * dt
            + sv * rho * frozen_dZ[k]
            + sv * np.sqrt(max(1.0 - rho ** 2, 0.0)) * d_zeta[:, k]
        )
        log_S[:, k + 1] = log_S[:, k] + dlog

    return np.exp(log_S)
