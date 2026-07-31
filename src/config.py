"""
Configuration and parameter objects for the Pure Rough Heston model.

Pure Rough Heston specialization:
  L(t,s) = 1,  f(V) = sqrt(V)

Spot dynamics:
  dS_t = (r_t - d_t) S_t dt + S_t sqrt(V_t) (rho dZ_t + sqrt(1-rho^2) d_zeta_t)

Variance process (stochastic Volterra equation):
  V_t = V_0 + int_0^t K(t-s) [kappa*(theta - V_s) ds + nu*sqrt(V_s) dZ_s]
  K(u) = u^(H - 1/2) / Gamma(H + 1/2),  H in (0, 1/2)

Restrictions: H in (0, 0.5), V_0>0, kappa>0, theta>0, nu>0, |rho|<1, S_0>0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class RoughHestonParams:
    """Parameters for the pure rough Heston model.

    Attributes
    ----------
    S0 : float
        Initial spot price (must be > 0).
    V0 : float
        Initial variance (must be > 0).
    H : float
        Hurst exponent, in (0, 0.5).
    kappa : float
        Mean-reversion speed (must be > 0).
    theta : float
        Long-run variance (must be > 0).
    nu : float
        Volatility-of-volatility (must be > 0).
    rho : float
        Correlation between spot and variance Brownian motions (|rho| < 1).
    r : float
        Constant risk-free rate.
    d : float
        Constant continuous dividend yield.
    T : float
        Final horizon (must be > 0).
    n_steps : int
        Number of uniform time steps over [0, T].
    seed : int
        Default random seed for reproducibility.
    """

    S0: float = 1.0
    V0: float = 0.04
    H: float = 0.1
    kappa: float = 1.0
    theta: float = 0.04
    nu: float = 0.5
    rho: float = -0.7
    r: float = 0.05
    d: float = 0.0
    T: float = 1.0
    n_steps: int = 1000
    seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Assert all parameter restrictions hold."""
        if not (0.0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5), got H={self.H}")
        if self.V0 <= 0.0:
            raise ValueError(f"V0 must be > 0, got V0={self.V0}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be > 0, got kappa={self.kappa}")
        if self.theta <= 0.0:
            raise ValueError(f"theta must be > 0, got theta={self.theta}")
        if self.nu <= 0.0:
            raise ValueError(f"nu must be > 0, got nu={self.nu}")
        if self.S0 <= 0.0:
            raise ValueError(f"S0 must be > 0, got S0={self.S0}")
        if not (abs(self.rho) < 1.0):
            raise ValueError(f"|rho| must be < 1, got rho={self.rho}")
        if self.T <= 0.0:
            raise ValueError(f"T must be > 0, got T={self.T}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got n_steps={self.n_steps}")

    @property
    def dt(self) -> float:
        """Uniform time step size."""
        return self.T / self.n_steps

    @property
    def time_grid(self) -> np.ndarray:
        """Uniform time grid 0 = t_0 < t_1 < ... < t_{n_steps} = T."""
        return np.linspace(0.0, self.T, self.n_steps + 1)

    def r_func(self, t: float) -> float:
        """Risk-free rate at time t (constant)."""
        return self.r

    def d_func(self, t: float) -> float:
        """Dividend yield at time t (constant)."""
        return self.d

    def __repr__(self) -> str:
        return (
            f"RoughHestonParams(S0={self.S0}, V0={self.V0}, H={self.H}, "
            f"kappa={self.kappa}, theta={self.theta}, nu={self.nu}, "
            f"rho={self.rho}, r={self.r}, d={self.d}, T={self.T}, "
            f"n_steps={self.n_steps})"
        )


# ──────────────────────────────────────────────
# Convenience factory for robustness grid
# ──────────────────────────────────────────────

def make_param_grid(
    H_values: list[float] | None = None,
    rho_values: list[float] | None = None,
    base: RoughHestonParams | None = None,
) -> list[RoughHestonParams]:
    """Return a list of parameter sets spanning (H, rho) combinations.

    Parameters
    ----------
    H_values : list of float, optional
        Hurst exponents to vary.  Defaults to [0.05, 0.1, 0.2, 0.3].
    rho_values : list of float, optional
        Correlation values.  Defaults to [-0.9, -0.5, 0.0, 0.5].
    base : RoughHestonParams, optional
        Base parameter set whose H and rho are overridden.

    Returns
    -------
    list of RoughHestonParams
    """
    if H_values is None:
        H_values = [0.05, 0.1, 0.2, 0.3]
    if rho_values is None:
        rho_values = [-0.9, -0.5, 0.0, 0.5]
    if base is None:
        base = RoughHestonParams()

    grid: list[RoughHestonParams] = []
    for H in H_values:
        for rho in rho_values:
            p = RoughHestonParams(
                S0=base.S0, V0=base.V0, H=H, kappa=base.kappa,
                theta=base.theta, nu=base.nu, rho=rho,
                r=base.r, d=base.d, T=base.T, n_steps=base.n_steps,
                seed=base.seed,
            )
            grid.append(p)
    return grid
