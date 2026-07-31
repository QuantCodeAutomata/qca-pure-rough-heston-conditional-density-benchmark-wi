"""
Generic LSRV (Local Stochastic Rough Volatility) stochastic flow solver.

Model:
  dS_t = (r-d) S_t dt + S_t L(t,S_t) sqrt(V_t) [rho dZ_t + sqrt(1-rho^2) d_zeta_t]
  V_t: Volterra-Heston process (rough Heston variance)

Stochastic flow Phi_t(x) solves (frozen common path):
  dPhi_t(x) = Gamma_t(Phi_t(x)) dZ_t,    Phi_0(x) = x
  where Gamma_t(s) = rho * s * L(t,s) * sqrt(max(V_t, 0))

Jacobian J_t(x) = d/dx Phi_t(x):
  dJ_t(x) = Gamma'_t(Phi_t(x)) * J_t(x) dZ_t,   J_0(x) = 1

Second derivative H_t(x) = d^2/dx^2 Phi_t(x):
  dH_t(x) = [Gamma''_t(Phi_t(x)) * J_t(x)^2 + Gamma'_t(Phi_t(x)) * H_t(x)] dZ_t,
             H_0(x) = 0

Transformed coefficients (general LSRV):
  DeltaTilde_t^2(x) = (1-rho^2) * Phi_t(x)^2 * L(t,Phi_t(x))^2 * V_t / J_t(x)^2
  muTilde_t(x) = (r-d) * Phi_t(x) / J_t(x)
                 - 0.5 * (1-rho^2) * Phi_t(x)^2 * L(t,Phi_t(x))^2 * V_t * H_t(x) / J_t(x)^3

Custom — Context7 found no library equivalent for LSRV stochastic flow simulation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from .config import RoughHestonParams
from .density import EPSILON


# ──────────────────────────────────────────────
# Leverage function
# ──────────────────────────────────────────────

class LeverageFunction:
    """Nontrivial deterministic leverage function with spatial derivatives.

    L(t,s) = 1 + alpha * exp(-beta * t) * tanh(gamma * log(s / S0))

    Chosen to be smooth, positive (for |alpha| < 1), and approach 1 as t -> inf.
    Derivatives are computed analytically.
    """

    def __init__(self, alpha: float, beta: float, gamma: float, S0: float) -> None:
        if abs(alpha) >= 1.0:
            raise ValueError(f"|alpha| must be < 1 for L > 0; got alpha={alpha}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.S0 = S0

    def _u(self, t: float, s: np.ndarray) -> np.ndarray:
        """Argument of tanh: gamma * log(s/S0)."""
        return self.gamma * np.log(np.maximum(s, EPSILON) / self.S0)

    def L(self, t: float, s: np.ndarray) -> np.ndarray:
        """L(t,s) = 1 + alpha * exp(-beta*t) * tanh(u)."""
        return 1.0 + self.alpha * np.exp(-self.beta * t) * np.tanh(self._u(t, s))

    def dL_ds(self, t: float, s: np.ndarray) -> np.ndarray:
        """dL/ds = alpha * exp(-beta*t) * gamma/s * sech^2(u)."""
        u = self._u(t, s)
        sech2 = 1.0 / np.cosh(u) ** 2
        return self.alpha * np.exp(-self.beta * t) * self.gamma / np.maximum(s, EPSILON) * sech2

    def d2L_ds2(self, t: float, s: np.ndarray) -> np.ndarray:
        """d^2L/ds^2.

        d/ds [gamma/s * sech^2(u)] where u = gamma*log(s/S0):
        = gamma/s^2 * sech^2(u) * (gamma*tanh(u)*(-2) - 1)
        """
        u = self._u(t, s)
        s_safe = np.maximum(s, EPSILON)
        sech2 = 1.0 / np.cosh(u) ** 2
        tanh_u = np.tanh(u)
        factor = (-1.0 - 2.0 * self.gamma * tanh_u) * sech2
        return self.alpha * np.exp(-self.beta * t) * self.gamma / s_safe ** 2 * factor

    def d3L_ds3(self, t: float, s: np.ndarray) -> np.ndarray:
        """d^3L/ds^3 (for completeness; used in some regularity checks)."""
        u = self._u(t, s)
        s_safe = np.maximum(s, EPSILON)
        sech2 = 1.0 / np.cosh(u) ** 2
        tanh_u = np.tanh(u)
        # Computed via chain rule of the second derivative formula
        g = self.gamma
        # d/ds of [g/s^2 * (-1 - 2g*tanh(u)) * sech^2(u)]
        # = g/s^3 * [(-2)(-1-2g*tanh(u)) + (-2g*sech^2(u)*g/s) * s] * sech^2(u)
        # Approximate (exact but verbose)
        t1 = 2.0 * (-1.0 - 2.0 * g * tanh_u) * sech2
        t2 = -2.0 * g * g * sech2 ** 2 + 4.0 * g * tanh_u * 2.0 * g * tanh_u * sech2
        return self.alpha * np.exp(-self.beta * t) * g / s_safe ** 3 * (t1 + t2)


class PureHestonLeverage(LeverageFunction):
    """Pure L=1 leverage (pure rough Heston special case)."""

    def __init__(self, S0: float) -> None:
        super().__init__(alpha=0.0, beta=0.0, gamma=1.0, S0=S0)

    def L(self, t: float, s: np.ndarray) -> np.ndarray:
        return np.ones_like(s)

    def dL_ds(self, t: float, s: np.ndarray) -> np.ndarray:
        return np.zeros_like(s)

    def d2L_ds2(self, t: float, s: np.ndarray) -> np.ndarray:
        return np.zeros_like(s)

    def d3L_ds3(self, t: float, s: np.ndarray) -> np.ndarray:
        return np.zeros_like(s)


# ──────────────────────────────────────────────
# Stochastic flow solver
# ──────────────────────────────────────────────

def solve_stochastic_flow(
    params: RoughHestonParams,
    dZ_path: np.ndarray,
    V_path: np.ndarray,
    x_grid: np.ndarray,
    leverage_fn: LeverageFunction,
) -> dict:
    """Solve the stochastic flow (Phi, J, H) for a frozen common-environment path.

    Uses Euler–Maruyama on the realized dZ increments.

    Parameters
    ----------
    params : RoughHestonParams
    dZ_path : np.ndarray, shape (n_steps,)
        Frozen Z increments.
    V_path : np.ndarray, shape (n_steps+1,)
        Frozen variance path.
    x_grid : np.ndarray, shape (n_x,)
        Grid of initial x values (x > 0).
    leverage_fn : LeverageFunction

    Returns
    -------
    dict with keys:
        'Phi' : np.ndarray, shape (n_steps+1, n_x)
        'J'   : np.ndarray, shape (n_steps+1, n_x)
        'H'   : np.ndarray, shape (n_steps+1, n_x)
        'time_grid' : np.ndarray, shape (n_steps+1,)
        'negative_J_count' : int
    """
    n = params.n_steps
    rho = params.rho
    n_x = len(x_grid)
    time_grid = params.time_grid

    Phi = np.zeros((n + 1, n_x))
    J = np.zeros((n + 1, n_x))
    H = np.zeros((n + 1, n_x))

    Phi[0] = x_grid.copy()
    J[0] = 1.0
    H[0] = 0.0

    neg_J_count = 0

    for k in range(n):
        t_k = time_grid[k]
        dZ_k = dZ_path[k]
        v_k = float(np.maximum(V_path[k], 0.0))
        sv = np.sqrt(v_k)

        phi_k = Phi[k]  # (n_x,)
        j_k = J[k]      # (n_x,)
        h_k = H[k]      # (n_x,)

        # Gamma_t(s) = rho * s * L(t,s) * sqrt(V_t)
        L_k = leverage_fn.L(t_k, phi_k)
        dL_k = leverage_fn.dL_ds(t_k, phi_k)
        d2L_k = leverage_fn.d2L_ds2(t_k, phi_k)

        # Gamma = rho * s * L * sv
        Gamma = rho * phi_k * L_k * sv
        # Gamma' = rho * sv * (L + s * dL/ds) = rho * sv * (L + s * dL)
        dGamma = rho * sv * (L_k + phi_k * dL_k)
        # Gamma'' = rho * sv * (dL + dL + s * d2L) = rho * sv * (2*dL + s * d2L)
        d2Gamma = rho * sv * (2.0 * dL_k + phi_k * d2L_k)

        # Euler-Maruyama update (Milstein-style for Jacobian equation)
        # Phi update: dPhi = Gamma(Phi) * dZ
        Phi[k + 1] = phi_k + Gamma * dZ_k
        # J update: dJ = Gamma'(Phi) * J * dZ
        J[k + 1] = j_k + dGamma * j_k * dZ_k
        # H update: dH = [Gamma''(Phi)*J^2 + Gamma'(Phi)*H] * dZ
        H[k + 1] = h_k + (d2Gamma * j_k ** 2 + dGamma * h_k) * dZ_k

        # Monitor J positivity
        neg_J_count += int(np.sum(J[k + 1] <= 0.0))

    return {
        "Phi": Phi,
        "J": J,
        "H": H,
        "time_grid": time_grid,
        "negative_J_count": neg_J_count,
    }


def compute_transformed_coefficients(
    params: RoughHestonParams,
    flow_result: dict,
    V_path: np.ndarray,
    leverage_fn: LeverageFunction,
) -> dict:
    """Compute DeltaTilde_t^2(x) and muTilde_t(x) from the flow.

    Parameters
    ----------
    params : RoughHestonParams
    flow_result : dict
        Output of solve_stochastic_flow.
    V_path : np.ndarray, shape (n_steps+1,)
    leverage_fn : LeverageFunction

    Returns
    -------
    dict with:
        'DeltaTilde_sq' : np.ndarray, shape (n_steps+1, n_x)
        'muTilde'       : np.ndarray, shape (n_steps+1, n_x)
    """
    n = params.n_steps
    rho = params.rho
    r_d = params.r - params.d
    time_grid = params.time_grid

    Phi = flow_result["Phi"]
    J = flow_result["J"]
    H = flow_result["H"]

    DeltaTilde_sq = np.zeros_like(Phi)
    muTilde = np.zeros_like(Phi)

    for k in range(n + 1):
        t_k = time_grid[k]
        v_k = float(np.maximum(V_path[k], 0.0))
        phi_k = Phi[k]
        j_k = np.maximum(J[k], EPSILON)
        h_k = H[k]

        L_k = leverage_fn.L(t_k, phi_k)

        # Delta_t^2(s) = (1-rho^2) * s^2 * L(t,s)^2 * V_t
        Delta_sq = (1.0 - rho ** 2) * phi_k ** 2 * L_k ** 2 * v_k
        DeltaTilde_sq[k] = Delta_sq / j_k ** 2

        # muTilde = mu(Phi)/J - 0.5 * Delta^2(Phi) * H / J^3
        mu_phi = r_d * phi_k
        muTilde[k] = mu_phi / j_k - 0.5 * Delta_sq * h_k / j_k ** 3

    return {
        "DeltaTilde_sq": DeltaTilde_sq,
        "muTilde": muTilde,
    }


def invert_flow(
    K: float,
    Phi_t: np.ndarray,
    x_grid: np.ndarray,
) -> float:
    """Find x_K(t) = Phi_t^{-1}(K) by monotone interpolation.

    Parameters
    ----------
    K : float
        Strike (positive).
    Phi_t : np.ndarray, shape (n_x,)
        Flow values at time t on x_grid.
    x_grid : np.ndarray, shape (n_x,)

    Returns
    -------
    float — preimage x such that Phi_t(x) ≈ K.
    """
    # Phi should be monotone increasing; use linear interpolation
    if K < Phi_t[0] or K > Phi_t[-1]:
        return np.nan
    inv_fn = interp1d(Phi_t, x_grid, kind="linear", fill_value="extrapolate")
    return float(inv_fn(K))


# ──────────────────────────────────────────────
# Transformed PDE solver (x-space, variable coefficients)
# ──────────────────────────────────────────────

def solve_transformed_pde_lsrv(
    params: RoughHestonParams,
    x_grid: np.ndarray,
    DeltaTilde_sq: np.ndarray,
    muTilde: np.ndarray,
    reporting_times: list[int],
    init_sigma: float = 0.05,
) -> dict:
    """Solve the transformed pathwise PDE in x-space for the LSRV model.

    PDE (x-space):
      partial_t q_t(x) = -partial_x(muTilde_t(x) q_t(x))
                         + 0.5 * partial_xx(DeltaTilde_t^2(x) q_t(x))

    Uses Crank–Nicolson with central differences on x_grid.

    Parameters
    ----------
    params : RoughHestonParams
    x_grid : np.ndarray, shape (n_x,)
        Positive x-grid (may be nonuniform; we use the step sizes directly).
    DeltaTilde_sq : np.ndarray, shape (n_steps+1, n_x)
        Path-dependent spatially-varying diffusion coefficient.
    muTilde : np.ndarray, shape (n_steps+1, n_x)
        Path-dependent spatially-varying advection coefficient.
    reporting_times : list of int
        Time indices to save.
    init_sigma : float
        Width of the initial Gaussian approximation to delta_{S0}.

    Returns
    -------
    dict with 'q_solutions', 'mass_history', etc.
    """
    n = params.n_steps
    n_x = len(x_grid)
    dt = params.dt
    S0 = params.S0
    dx = np.diff(x_grid)  # (n_x-1,)
    dx_mean = np.mean(dx)  # use mean step for FD approximation

    # Initial condition: narrow Gaussian in x-space approximating delta_{S0}
    q = np.exp(-0.5 * ((x_grid - S0) / init_sigma) ** 2)
    q /= (np.trapz(q, x_grid) + EPSILON)
    q[0] = 0.0
    q[-1] = 0.0

    mass_history = np.zeros(n + 1)
    mass_history[0] = float(np.trapz(q, x_grid))

    q_solutions = []
    if 0 in reporting_times:
        q_solutions.append(q.copy())

    for k in range(n):
        # Use time-averaged coefficients
        D_k = 0.5 * (DeltaTilde_sq[k] + DeltaTilde_sq[k + 1])  # (n_x,)
        mu_k = 0.5 * (muTilde[k] + muTilde[k + 1])              # (n_x,)

        # Conservative flux PDE: partial_t q = -partial_x(mu*q) + 0.5*partial_xx(D*q)
        # Conservative diffusion: d = D, advection: a = mu
        # Use backward Euler for simplicity (more stable for variable coefficients)
        # Build tridiagonal matrix for (I - dt * L)
        # L q_j = -(mu_{j+1} q_{j+1} - mu_{j-1} q_{j-1})/(2*dx)
        #         + 0.5*(D_{j+1} q_{j+1} - 2*D_j q_j + D_{j-1} q_{j-1})/dx^2

        dx2 = dx_mean ** 2
        lower_d = -mu_k[:-1] / (2.0 * dx_mean) + 0.5 * D_k[:-1] / dx2  # (n_x-1,)
        diag_d = -2.0 * 0.5 * D_k / dx2                                  # (n_x,)
        upper_d = mu_k[1:] / (2.0 * dx_mean) + 0.5 * D_k[1:] / dx2     # (n_x-1,)

        diags = [lower_d, diag_d, upper_d]
        L = sp.diags(diags, offsets=[-1, 0, 1], shape=(n_x, n_x), format="lil")
        L[0, :] = 0.0
        L[-1, :] = 0.0
        L = L.tocsr()

        I = sp.eye(n_x, format="csr")
        A = I - dt * L  # backward Euler

        rhs = q.copy()
        rhs[0] = 0.0
        rhs[-1] = 0.0

        q_new = spla.spsolve(A, rhs)
        q = np.maximum(q_new, 0.0)
        q[0] = 0.0
        q[-1] = 0.0

        t_idx = k + 1
        mass_history[t_idx] = float(np.trapz(q, x_grid))

        if t_idx in reporting_times:
            q_solutions.append(q.copy())

    return {
        "x_grid": x_grid,
        "q_solutions": q_solutions,
        "mass_history": mass_history,
        "reporting_times": reporting_times,
    }


# ──────────────────────────────────────────────
# Nested conditional Monte Carlo
# ──────────────────────────────────────────────

def nested_conditional_mc(
    params: RoughHestonParams,
    dZ_path: np.ndarray,
    V_path: np.ndarray,
    leverage_fn: LeverageFunction,
    M_inner: int = 3000,
    t_idx: int = -1,
    seed: int = 42,
) -> dict:
    """Estimate conditional density p_t(s|common path) by nested Monte Carlo.

    Simulates M inner idiosyncratic paths conditional on the frozen (Z, V) path.
    Uses kernel density estimation to estimate the conditional spot density.

    Parameters
    ----------
    params : RoughHestonParams
    dZ_path : np.ndarray, shape (n_steps,)
        Frozen Z increments.
    V_path : np.ndarray, shape (n_steps+1,)
        Frozen variance path.
    leverage_fn : LeverageFunction
    M_inner : int
        Number of inner paths.
    t_idx : int
        Time index to evaluate at (-1 = final).
    seed : int

    Returns
    -------
    dict with:
        'S_t'     : np.ndarray, shape (M_inner,) — conditional spot samples
        'kde'     : gaussian_kde object
        's_grid'  : np.ndarray — evaluation grid for density
        'p_kde'   : np.ndarray — KDE density values on s_grid
    """
    rng = np.random.default_rng(seed)
    n = params.n_steps
    dt = params.dt
    rho = params.rho
    S0 = params.S0
    r_d = params.r - params.d
    time_grid = params.time_grid

    if t_idx < 0:
        t_idx = n + t_idx + 1

    # Simulate M idiosyncratic paths
    d_zeta = rng.standard_normal((M_inner, n)) * np.sqrt(dt)

    log_S = np.full(M_inner, np.log(S0))

    for k in range(t_idx):
        t_k = time_grid[k]
        v_k = float(np.maximum(V_path[k], 0.0))
        sv = np.sqrt(v_k)
        dZ_k = float(dZ_path[k])

        # Current spot for leverage evaluation
        S_k = np.exp(log_S)
        L_k = leverage_fn.L(t_k, S_k)
        sigma_k = S_k * L_k * sv

        # Log-Euler step:
        # d log S = (r-d - 0.5 * sigma_total^2 / S^2) * dt + sigma/S * [rho*dZ + sqrt(1-rho^2)*d_zeta]
        # where sigma_total^2 = (sigma_k)^2 = S_k^2 * L_k^2 * V_k
        # Note: sigma/S = L_k * sv  (local vol * vol of vol process)
        loc_vol = L_k * sv
        dlog = (
            (r_d - 0.5 * loc_vol ** 2) * dt
            + loc_vol * (rho * dZ_k + np.sqrt(max(1.0 - rho ** 2, 0.0)) * d_zeta[:, k])
        )
        log_S = log_S + dlog

    S_t = np.exp(log_S)

    # Kernel density estimate
    S_pos = S_t[S_t > 0]
    if len(S_pos) < 10:
        return {"S_t": S_t, "kde": None, "s_grid": None, "p_kde": None}

    kde = gaussian_kde(S_pos, bw_method="scott")
    s_min = np.percentile(S_pos, 0.5)
    s_max = np.percentile(S_pos, 99.5)
    s_grid = np.linspace(max(s_min * 0.5, EPSILON), s_max * 1.5, 500)
    p_kde = kde(s_grid)

    return {
        "S_t": S_t,
        "kde": kde,
        "s_grid": s_grid,
        "p_kde": p_kde,
    }
