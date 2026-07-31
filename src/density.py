"""
Explicit conditional density evaluators for pure rough Heston.

In pure rough Heston (L=1, f(V)=sqrt(V)):
  Stochastic flow:   Phi_t(x) = x * M_t,  M_t = exp(rho*I_t - 0.5*rho^2*Vbar_t)
  Jacobian:          J_t(x) = M_t  (positive, independent of x)
  Second derivative: H_t(x) = 0

Transformed conditional density (equation 38):
  q_t(x) = LN(x; mu_q, sigma_q^2)
  where mu_q = log(S0) + R_t - 0.5*A_t,  sigma_q^2 = A_t

Original conditional density (equation 39):
  p_t(s) = LN(s; mu_p, sigma_p^2)
  where mu_p = log(S0) + R_t + rho*I_t - 0.5*Vbar_t,  sigma_p^2 = A_t

Inverse-flow recovery:  p_t(K) = q_t(K/M_t) / M_t
Transformation identity: q_t(x) = p_t(Phi_t(x)) * J_t(x) = p_t(x*M_t) * M_t

All computations in log-space to avoid underflow.
Safeguard: if A_t < EPSILON, density is treated as degenerate (concentrated at S0).

Custom — Context7 found no library equivalent for rough Heston explicit density formulas.
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm

from .config import RoughHestonParams

EPSILON = 1e-12  # minimum variance threshold


def lognormal_log_pdf(
    x: np.ndarray,
    log_mean: float | np.ndarray,
    variance: float | np.ndarray,
) -> np.ndarray:
    """Log-PDF of lognormal: log f(x) = -log(x) - 0.5*log(2*pi*var) - (log(x) - mu)^2/(2*var).

    Parameters
    ----------
    x : np.ndarray
        Positive evaluation points.
    log_mean : float or np.ndarray
        Mean of log(x).
    variance : float or np.ndarray
        Variance of log(x).

    Returns
    -------
    log_pdf : np.ndarray, same shape as x (broadcast with log_mean, variance).
    """
    log_x = np.log(np.maximum(x, EPSILON))
    log_pdf = (
        -log_x
        - 0.5 * np.log(2.0 * np.pi * np.maximum(variance, EPSILON))
        - (log_x - log_mean) ** 2 / (2.0 * np.maximum(variance, EPSILON))
    )
    return log_pdf


def eval_q_t(
    x: np.ndarray,
    A_t: float | np.ndarray,
    R_t: float,
    S0: float,
) -> np.ndarray:
    """Evaluate explicit transformed density q_t (equation 38).

    q_t(x) = 1/(x * sqrt(2*pi*A_t)) * exp(-(log(x/S0) - R_t + 0.5*A_t)^2 / (2*A_t))

    This is a lognormal density for x with:
      log(x) ~ N(log(S0) + R_t - 0.5*A_t,  A_t)

    Parameters
    ----------
    x : np.ndarray
        Positive evaluation points (last axis is the space axis).
    A_t : float or np.ndarray
        (1-rho^2)*Vbar_t.  Broadcast with x.
    R_t : float
        Deterministic rate integral int_0^t (r-d) du.
    S0 : float
        Initial spot.

    Returns
    -------
    q : np.ndarray  — density values (≥ 0; returns 0 where A_t < EPSILON).
    """
    A_safe = np.maximum(np.asarray(A_t, dtype=float), EPSILON)
    degenerate = np.asarray(A_t) < EPSILON
    log_mean = np.log(S0) + R_t - 0.5 * A_safe
    log_pdf = lognormal_log_pdf(x, log_mean, A_safe)
    q = np.exp(log_pdf)
    q = np.where(degenerate, 0.0, q)
    return q


def eval_p_t(
    s: np.ndarray,
    I_t: float | np.ndarray,
    Vbar_t: float | np.ndarray,
    A_t: float | np.ndarray,
    R_t: float,
    S0: float,
    rho: float,
) -> np.ndarray:
    """Evaluate explicit original conditional density p_t (equation 39).

    p_t(s) = 1/(s * sqrt(2*pi*A_t)) * exp(-(log(s/S0) - R_t - rho*I_t + 0.5*Vbar_t)^2 / (2*A_t))

    Parameters
    ----------
    s : np.ndarray
        Positive evaluation points.
    I_t : float or np.ndarray
        Ito integral int_0^t sqrt(V_u) dZ_u.
    Vbar_t : float or np.ndarray
        Time integral int_0^t V_u du.
    A_t : float or np.ndarray
        (1-rho^2)*Vbar_t.
    R_t : float
        Deterministic rate integral.
    S0 : float
        Initial spot.
    rho : float
        Correlation.

    Returns
    -------
    p : np.ndarray
    """
    A_safe = np.maximum(np.asarray(A_t, dtype=float), EPSILON)
    degenerate = np.asarray(A_t) < EPSILON
    log_mean = np.log(S0) + R_t + rho * np.asarray(I_t) - 0.5 * np.asarray(Vbar_t)
    log_pdf = lognormal_log_pdf(s, log_mean, A_safe)
    p = np.exp(log_pdf)
    p = np.where(degenerate, 0.0, p)
    return p


def inverse_flow_recovery(
    K: np.ndarray,
    M_t: float | np.ndarray,
    A_t: float | np.ndarray,
    R_t: float,
    S0: float,
) -> np.ndarray:
    """Recover p_t(K) from q_t via inverse flow: p_t(K) = q_t(K/M_t) / M_t.

    Parameters
    ----------
    K : np.ndarray
        Strike grid (positive).
    M_t : float or np.ndarray
        M_t = exp(rho*I_t - 0.5*rho^2*Vbar_t).  Broadcast with K.
    A_t : float or np.ndarray
        (1-rho^2)*Vbar_t.
    R_t : float
    S0 : float

    Returns
    -------
    p_recovered : np.ndarray
    """
    M = np.maximum(np.asarray(M_t), EPSILON)
    x_K = K / M                                      # inverse flow preimage
    q_val = eval_q_t(x_K, A_t, R_t, S0)
    return q_val / M


def verify_transformation_identity(
    x: np.ndarray,
    M_t: float | np.ndarray,
    I_t: float | np.ndarray,
    Vbar_t: float | np.ndarray,
    A_t: float | np.ndarray,
    R_t: float,
    S0: float,
    rho: float,
) -> dict:
    """Check q_t(x) == p_t(x*M_t) * M_t pathwise.

    Parameters
    ----------
    x : np.ndarray
        Positive x-grid.
    M_t, I_t, Vbar_t, A_t : float or np.ndarray (per-path scalars broadcast with x)
    R_t, S0, rho : float

    Returns
    -------
    dict with keys:
        'q_direct'     : np.ndarray — q_t evaluated directly
        'q_via_p'      : np.ndarray — p_t(x*M_t) * M_t
        'abs_error'    : np.ndarray
        'rel_error'    : np.ndarray
        'max_abs_error': float
        'max_rel_error': float
    """
    q_direct = eval_q_t(x, A_t, R_t, S0)
    s_vals = x * np.asarray(M_t)
    p_vals = eval_p_t(s_vals, I_t, Vbar_t, A_t, R_t, S0, rho)
    q_via_p = p_vals * np.asarray(M_t)

    abs_err = np.abs(q_direct - q_via_p)
    denom = np.maximum(np.abs(q_direct), EPSILON)
    rel_err = abs_err / denom

    return {
        "q_direct": q_direct,
        "q_via_p": q_via_p,
        "abs_error": abs_err,
        "rel_error": rel_err,
        "max_abs_error": float(np.max(abs_err)),
        "max_rel_error": float(np.max(rel_err)),
    }


def normalization_error(
    density: np.ndarray,
    grid: np.ndarray,
) -> float:
    """Estimate |int density dx - 1| using the trapezoidal rule.

    Parameters
    ----------
    density : np.ndarray, shape (..., n_grid)
    grid : np.ndarray, shape (n_grid,)

    Returns
    -------
    float — absolute normalization error.
    """
    mass = np.trapz(density, x=grid, axis=-1)
    return float(np.abs(mass - 1.0))


def check_positivity(density: np.ndarray) -> dict:
    """Check whether the density is non-negative everywhere.

    Returns
    -------
    dict with 'all_positive' (bool), 'n_negative' (int), 'min_value' (float).
    """
    return {
        "all_positive": bool(np.all(density >= 0.0)),
        "n_negative": int(np.sum(density < 0.0)),
        "min_value": float(np.min(density)),
    }


def compute_first_moment(
    density: np.ndarray,
    grid: np.ndarray,
) -> float:
    """Estimate E[S_t] = int s * p_t(s) ds via the trapezoidal rule.

    Parameters
    ----------
    density : np.ndarray, shape (n_grid,)
    grid : np.ndarray, shape (n_grid,)

    Returns
    -------
    float
    """
    return float(np.trapz(grid * density, x=grid))


def eval_q_t_batch(
    x: np.ndarray,
    A_t: np.ndarray,
    R_t: np.ndarray | float,
    S0: float,
) -> np.ndarray:
    """Evaluate q_t for N paths over a common x-grid.

    Parameters
    ----------
    x : np.ndarray, shape (n_x,)
    A_t : np.ndarray, shape (N,)
    R_t : float or np.ndarray, shape (N,)
    S0 : float

    Returns
    -------
    q : np.ndarray, shape (N, n_x)
    """
    # Broadcast: (N, 1) against (1, n_x)
    A_t_col = np.asarray(A_t)[:, np.newaxis]
    R_t_col = np.atleast_1d(R_t)[:, np.newaxis] if np.ndim(R_t) > 0 else float(R_t)
    x_row = x[np.newaxis, :]
    A_safe = np.maximum(A_t_col, EPSILON)
    log_mean = np.log(S0) + R_t_col - 0.5 * A_safe
    log_pdf = lognormal_log_pdf(x_row, log_mean, A_safe)
    q = np.exp(log_pdf)
    q = np.where(A_t_col < EPSILON, 0.0, q)
    return q


def eval_p_t_batch(
    s: np.ndarray,
    I_t: np.ndarray,
    Vbar_t: np.ndarray,
    A_t: np.ndarray,
    R_t: float | np.ndarray,
    S0: float,
    rho: float,
) -> np.ndarray:
    """Evaluate p_t for N paths over a common s-grid.

    Parameters
    ----------
    s : np.ndarray, shape (n_s,)
    I_t : np.ndarray, shape (N,)
    Vbar_t : np.ndarray, shape (N,)
    A_t : np.ndarray, shape (N,)
    R_t : float
    S0 : float
    rho : float

    Returns
    -------
    p : np.ndarray, shape (N, n_s)
    """
    I_col = I_t[:, np.newaxis]
    Vbar_col = Vbar_t[:, np.newaxis]
    A_col = A_t[:, np.newaxis]
    A_safe = np.maximum(A_col, EPSILON)
    R_val = float(R_t) if np.ndim(R_t) == 0 else np.asarray(R_t)[:, np.newaxis]
    log_mean = np.log(S0) + R_val + rho * I_col - 0.5 * Vbar_col
    s_row = s[np.newaxis, :]
    log_pdf = lognormal_log_pdf(s_row, log_mean, A_safe)
    p = np.exp(log_pdf)
    p = np.where(A_col < EPSILON, 0.0, p)
    return p


def eval_p_t_at_strikes(
    K: np.ndarray,
    I_t: np.ndarray,
    Vbar_t: np.ndarray,
    A_t: np.ndarray,
    R_t: float | np.ndarray,
    S0: float,
    rho: float,
) -> np.ndarray:
    """Evaluate p_t^i(K_l) for each path i and strike K_l.

    Returns shape (N, n_strikes).
    """
    return eval_p_t_batch(K, I_t, Vbar_t, A_t, R_t, S0, rho)


def eval_q_t_at_inverse_strikes(
    K: np.ndarray,
    M_t: np.ndarray,
    A_t: np.ndarray,
    R_t: float | np.ndarray,
    S0: float,
) -> np.ndarray:
    """Evaluate q_t^i(K/M_t^i) for each path i and strike K_l.

    Returns shape (N, n_strikes).
    """
    # x_K^i(l) = K_l / M_t^i — shape (N, n_strikes)
    M_col = np.maximum(M_t[:, np.newaxis], EPSILON)
    x_K = K[np.newaxis, :] / M_col  # (N, n_strikes)

    A_col = np.maximum(A_t[:, np.newaxis], EPSILON)
    R_val = float(R_t) if np.ndim(R_t) == 0 else np.asarray(R_t)[:, np.newaxis]
    log_mean = np.log(S0) + R_val - 0.5 * A_col
    log_pdf = lognormal_log_pdf(x_K, log_mean, A_col)
    q = np.exp(log_pdf)
    q = np.where(A_t[:, np.newaxis] < EPSILON, 0.0, q)
    return q
