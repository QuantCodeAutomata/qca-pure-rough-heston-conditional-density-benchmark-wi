"""
Fokker–Planck PDE solver for the transformed conditional density q_t.

After conditioning on a common-environment path (Z^i, V^i) in pure rough Heston,
the transformed density qbar_t(y) = e^y q_t(e^y) satisfies a deterministic
advection-diffusion PDE in log-space (y = log x):

  partial_t qbar_t(y) = -partial_y [(b_t - 0.5*c_t^i) qbar_t(y)]
                         + 0.5*c_t^i * partial_yy qbar_t(y)

where:
  b_t   = r - d               (drift; constant in this implementation)
  c_t^i = (1 - rho^2) * V_t^i (frozen path diffusion coefficient)

Because c_t^i is spatially constant (depends only on t, not y), the equation is
a linear advection-diffusion PDE with time-varying coefficients, and the exact
solution is Gaussian in y:
  qbar_t(y) = N(y; mu_y(t), A_t)
  mu_y(t) = log(S0) + R_t - 0.5*A_t,  A_t = int_0^t c_s^i ds

We solve numerically via Crank–Nicolson finite differences on a y-grid and
compare against the exact solution.

Custom — Context7 found no library equivalent for pathwise rough Heston log-space PDE.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import RoughHestonParams
from .density import eval_q_t, EPSILON


def _build_cn_matrix(
    n_y: int,
    a: float,
    D: float,
    dy: float,
    dt: float,
) -> tuple[sp.csr_array, sp.csr_array]:
    """Build (I - 0.5*dt*L) and (I + 0.5*dt*L) for Crank-Nicolson.

    Interior operator L for advection-diffusion:
      (L u)_j = a*(u_{j+1}-u_{j-1})/(2 dy) + D*(u_{j+1}-2u_j+u_{j-1})/dy^2

    Boundary nodes are excluded (Dirichlet zero BC implemented by keeping them fixed at 0).
    Only interior rows 1..n_y-2 are active.

    Parameters
    ----------
    n_y : int
        Total grid size including boundaries.
    a : float
        Advection coefficient b_t - 0.5*c_t.
    D : float
        Diffusion coefficient 0.5*c_t.
    dy : float
        Spatial step.
    dt : float
        Time step.

    Returns
    -------
    A_lhs : csr_array, shape (n_y, n_y)   — (I - 0.5*dt*L)
    A_rhs : csr_array, shape (n_y, n_y)   — (I + 0.5*dt*L)
    """
    m = n_y  # grid size
    # Tridiagonal entries for interior rows.
    # PDE: ∂_t u = -a ∂_y u + D ∂_yy u  (conservative form with a = F = b - 0.5*c)
    # Discretise: (Lu)_j = -a*(u_{j+1}-u_{j-1})/(2dy) + D*(u_{j+1}-2u_j+u_{j-1})/dy^2
    lower = a / (2.0 * dy) + D / dy ** 2    # sub-diagonal  (+a/(2dy)+D/dy^2)
    diag = -2.0 * D / dy ** 2               # main diagonal
    upper = -a / (2.0 * dy) + D / dy ** 2  # super-diagonal (-a/(2dy)+D/dy^2)

    # Build sparse tridiagonal L (only interior rows contribute)
    diags = [
        np.full(m - 1, lower),   # sub-diagonal
        np.full(m, diag),        # main diagonal
        np.full(m - 1, upper),   # super-diagonal
    ]
    L = sp.diags(diags, offsets=[-1, 0, 1], shape=(m, m), format="csr")

    # Enforce Dirichlet at boundaries: rows 0 and m-1 become identity rows
    L = L.tolil()
    L[0, :] = 0.0
    L[m - 1, :] = 0.0
    L = L.tocsr()

    I = sp.eye(m, format="csr")
    A_lhs = I - 0.5 * dt * L
    A_rhs = I + 0.5 * dt * L
    return A_lhs, A_rhs


def solve_pde_log_space(
    params: RoughHestonParams,
    V_path: np.ndarray,
    reporting_times: list[int],
    n_y: int = 500,
    y_half_width: float = 8.0,
    init_sigma: float = 0.01,
) -> dict:
    """Solve log-space Fokker–Planck PDE for a single frozen common-environment path.

    Parameters
    ----------
    params : RoughHestonParams
        Model parameters.
    V_path : np.ndarray, shape (n_steps+1,)
        Frozen variance path (one outer path).
    reporting_times : list of int
        Time-step indices at which to save solutions.
    n_y : int
        Number of interior y-grid nodes (total grid = n_y + 2 including boundaries).
    y_half_width : float
        Half-width of y-domain in units of standard deviations of the expected
        density spread.  Actual domain: [log(S0) - y_half_width*std, log(S0) + y_half_width*std].
    init_sigma : float
        Standard deviation of the regularized delta initial condition (narrow Gaussian).

    Returns
    -------
    dict with keys:
        'y_grid'          : np.ndarray, shape (n_y+2,)
        'x_grid'          : np.ndarray, shape (n_y+2,)  — exp(y_grid)
        'qbar_solutions'  : list of np.ndarray at reporting times
        'q_solutions'     : list of np.ndarray at reporting times
        'mass_history'    : np.ndarray of mass integral at every time step
        'A_t_path'        : np.ndarray, shape (n_steps+1,)
        'R_t_path'        : np.ndarray, shape (n_steps+1,)
        'A_t_reporting'   : np.ndarray at reporting times
        'R_t_reporting'   : np.ndarray at reporting times
        'reporting_times' : list of int
        'params'          : RoughHestonParams
    """
    n = params.n_steps
    dt = params.dt
    S0 = params.S0
    rho = params.rho
    r, d = params.r, params.d
    b = r - d

    # Path-dependent diffusion coefficient: c_t^i = (1-rho^2)*V_t^i
    c_path = (1.0 - rho ** 2) * np.maximum(V_path, 0.0)  # shape (n+1,)

    # Cumulative integrals for A_t and R_t
    # Left-point Riemann: A_t[k] = sum_{j=0}^{k-1} c_{t_j} * dt
    A_t_path = np.zeros(n + 1)
    A_t_path[1:] = np.cumsum(c_path[:-1]) * dt

    R_t_path = (r - d) * params.time_grid

    # Determine y-grid bounds adaptively
    max_std = max(np.sqrt(np.maximum(A_t_path[-1], EPSILON)), init_sigma * 3)
    y_center = np.log(S0)
    y_min = y_center - y_half_width * max_std
    y_max = y_center + y_half_width * max_std

    # Total grid: n_y+2 points (first and last are Dirichlet boundaries)
    n_total = n_y + 2
    y_grid = np.linspace(y_min, y_max, n_total)
    dy = y_grid[1] - y_grid[0]
    x_grid = np.exp(y_grid)

    # Regularized delta initial condition: narrow Gaussian in qbar-space
    # qbar_0(y) ~ e^y * q_0(e^y) where q_0 → delta_{S0}
    # Approximate as: qbar_0(y) = Normal(y; log(S0), init_sigma^2) (unnormalized in y-space)
    # We use a Gaussian in y centred at log(S0).
    log_S0 = np.log(S0)
    qbar = np.exp(-0.5 * ((y_grid - log_S0) / init_sigma) ** 2)
    qbar /= (np.trapz(qbar, y_grid) + EPSILON)  # normalize to unit mass
    qbar[0] = 0.0  # Dirichlet BC
    qbar[-1] = 0.0

    mass_history = np.zeros(n + 1)
    mass_history[0] = float(np.trapz(qbar, y_grid))

    qbar_solutions = []
    q_solutions = []
    A_t_reporting = []
    R_t_reporting = []

    if 0 in reporting_times:
        qbar_solutions.append(qbar.copy())
        q_solutions.append(qbar / np.maximum(x_grid, EPSILON))
        A_t_reporting.append(A_t_path[0])
        R_t_reporting.append(R_t_path[0])

    # Time march with Crank-Nicolson
    for k in range(n):
        # Use time-average coefficient for Crank-Nicolson half-step
        c_mid = 0.5 * (c_path[k] + c_path[k + 1])
        a_mid = b - 0.5 * c_mid
        D_mid = 0.5 * c_mid

        # Build CN matrices
        A_lhs, A_rhs = _build_cn_matrix(n_total, a_mid, D_mid, dy, dt)

        # RHS = A_rhs @ qbar
        rhs = A_rhs.dot(qbar)

        # Enforce Dirichlet BCs in RHS
        rhs[0] = 0.0
        rhs[-1] = 0.0

        # Solve linear system
        qbar_new = spla.spsolve(A_lhs, rhs)

        # Enforce non-negativity and Dirichlet BCs
        qbar_new = np.maximum(qbar_new, 0.0)
        qbar_new[0] = 0.0
        qbar_new[-1] = 0.0

        qbar = qbar_new
        t_idx = k + 1
        mass_history[t_idx] = float(np.trapz(qbar, y_grid))

        if t_idx in reporting_times:
            qbar_solutions.append(qbar.copy())
            q = qbar / np.maximum(x_grid, EPSILON)
            q_solutions.append(q)
            A_t_reporting.append(A_t_path[t_idx])
            R_t_reporting.append(R_t_path[t_idx])

    return {
        "y_grid": y_grid,
        "x_grid": x_grid,
        "qbar_solutions": qbar_solutions,
        "q_solutions": q_solutions,
        "mass_history": mass_history,
        "A_t_path": A_t_path,
        "R_t_path": R_t_path,
        "A_t_reporting": np.array(A_t_reporting),
        "R_t_reporting": np.array(R_t_reporting),
        "reporting_times": reporting_times,
        "params": params,
    }


def compute_pde_error_vs_exact(
    pde_result: dict,
    params: RoughHestonParams,
    I_t_values: np.ndarray,
    Vbar_t_values: np.ndarray,
    M_t_values: np.ndarray,
) -> dict:
    """Compare numerical PDE solution against exact transformed density q_t.

    Parameters
    ----------
    pde_result : dict
        Output of solve_pde_log_space.
    params : RoughHestonParams
    I_t_values : np.ndarray, shape (n_reporting,)
        I_t at reporting times (for reference only; q_t does not depend on I_t).
    Vbar_t_values : np.ndarray, shape (n_reporting,)
        Vbar_t at reporting times.
    M_t_values : np.ndarray, shape (n_reporting,)
        M_t at reporting times.

    Returns
    -------
    dict with per-reporting-time error metrics.
    """
    x_grid = pde_result["x_grid"]
    S0 = params.S0
    r, d = params.r, params.d
    rho = params.rho

    errors = {
        "L1": [],
        "L2": [],
        "Linf": [],
        "mass_error": [],
        "normalization_pde": [],
        "normalization_exact": [],
    }

    for idx, (q_num, A_t, R_t) in enumerate(zip(
        pde_result["q_solutions"],
        pde_result["A_t_reporting"],
        pde_result["R_t_reporting"],
    )):
        # Exact q_t
        q_exact = eval_q_t(x_grid, A_t, R_t, S0)

        dx_arr = np.diff(x_grid)
        denom = np.maximum(np.abs(q_exact), EPSILON)

        L1 = float(np.trapz(np.abs(q_num - q_exact), x_grid))
        L2 = float(np.sqrt(np.trapz((q_num - q_exact) ** 2, x_grid)))
        Linf = float(np.max(np.abs(q_num - q_exact)))

        errors["L1"].append(L1)
        errors["L2"].append(L2)
        errors["Linf"].append(Linf)
        errors["mass_error"].append(float(np.trapz(np.abs(q_num - q_exact), x_grid)))
        errors["normalization_pde"].append(float(np.trapz(q_num, x_grid)))
        errors["normalization_exact"].append(float(np.trapz(q_exact, x_grid)))

    return errors
