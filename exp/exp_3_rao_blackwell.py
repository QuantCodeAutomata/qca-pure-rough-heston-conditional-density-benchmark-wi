"""
Experiment 3: Rao–Blackwellized Calibration Ratio Benchmark in Pure Rough Heston.

Validates the equivalence of equations (30) and (31) for Theta(t,K) = E[V_t | S_t=K],
measures convergence as N increases, assesses denominator stability across
strikes and maturities, and optionally computes a synthetic leverage update.
"""
from __future__ import annotations

import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RoughHestonParams
from src.simulator import simulate_rough_heston
from src.path_functionals import compute_path_functionals, get_functionals_at_time
from src.density import (
    eval_p_t_at_strikes,
    eval_q_t_at_inverse_strikes,
    EPSILON,
)
from src.rao_blackwell import (
    verify_estimator_equivalence,
    estimate_theta_surface,
    convergence_study,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_estimator_equivalence(
    params: RoughHestonParams,
    N: int = 1000,
    t_frac: float = 0.5,
    K_values: np.ndarray | None = None,
    tag: str = "baseline",
) -> dict:
    """Verify Theta_hat_trans == Theta_hat_orig and measure denominator stability.

    Returns
    -------
    dict with equivalence metrics and stability diagnostics.
    """
    if K_values is None:
        K_values = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]) * params.S0

    t_idx = max(1, int(t_frac * params.n_steps))

    sim = simulate_rough_heston(params, N, seed=params.seed)
    fcts = compute_path_functionals(sim["V"], sim["dZ"], params)
    ft = get_functionals_at_time(fcts, t_idx)

    I_t = ft["I_t"]
    Vbar_t = ft["Vbar_t"]
    A_t = ft["A_t"]
    M_t = ft["M_t"]
    R_t = ft["R_t"]
    V_t = sim["V"][:, t_idx]

    # p_t^i(K_l) — shape (N, n_K)
    p_dens = eval_p_t_at_strikes(K_values, I_t, Vbar_t, A_t, R_t, params.S0, params.rho)
    # q_t^i(K/M_t^i) — shape (N, n_K)
    q_dens = eval_q_t_at_inverse_strikes(K_values, M_t, A_t, R_t, params.S0)

    equiv = verify_estimator_equivalence(V_t, q_dens, M_t, p_dens)

    # Denominator analysis
    denom_sum = np.sum(p_dens, axis=0)  # (n_K,)
    stable_mask = equiv["is_stable"]

    return {
        "tag": tag,
        "t_frac": t_frac,
        "K_values": K_values.tolist(),
        "theta_trans": equiv["theta_trans"].tolist(),
        "theta_orig": equiv["theta_orig"].tolist(),
        "max_abs_diff": equiv["max_abs_diff"],
        "max_rel_diff": equiv["max_rel_diff"],
        "denom_sum": denom_sum.tolist(),
        "n_stable": int(np.sum(stable_mask)),
        "n_total": len(K_values),
        "V_t_mean": float(np.mean(V_t)),
    }


def run_convergence_exp(
    params: RoughHestonParams,
    N_values: list[int] | None = None,
    t_frac: float = 0.5,
    K_values: np.ndarray | None = None,
    N_ref: int = 15000,
    n_runs: int = 4,
    tag: str = "convergence",
) -> dict:
    """Convergence study: estimator error vs N."""
    if N_values is None:
        N_values = [100, 300, 1000, 3000]
    if K_values is None:
        K_values = np.array([0.8, 1.0, 1.2]) * params.S0

    t_idx = max(1, int(t_frac * params.n_steps))

    result = convergence_study(
        params, N_values, t_idx, K_values, N_ref=N_ref, n_runs=n_runs, seed=params.seed
    )

    return {
        "tag": tag,
        "N_values": N_values,
        "K_values": K_values.tolist(),
        "theta_ref": result["theta_ref"].tolist(),
        "rmse": result["rmse"].tolist(),
        "stable_fraction": result["stable_fraction"].tolist(),
    }


def run_maturity_strike_surface(
    params: RoughHestonParams,
    N: int = 2000,
    t_fracs: list[float] | None = None,
    K_values: np.ndarray | None = None,
    tag: str = "surface",
) -> dict:
    """Compute Theta_hat(t,K) surface and assess stability."""
    if t_fracs is None:
        t_fracs = [0.1, 0.25, 0.5, 1.0]
    if K_values is None:
        K_values = np.linspace(0.4, 2.0, 20) * params.S0

    t_idxs = [max(1, int(f * params.n_steps)) for f in t_fracs]

    sim = simulate_rough_heston(params, N, seed=params.seed)
    fcts = compute_path_functionals(sim["V"], sim["dZ"], params)

    theta_surface = np.full((len(t_fracs), len(K_values)), np.nan)
    stable_surface = np.zeros((len(t_fracs), len(K_values)), dtype=bool)

    for i_t, (f, t_idx) in enumerate(zip(t_fracs, t_idxs)):
        ft = get_functionals_at_time(fcts, t_idx)
        V_t = sim["V"][:, t_idx]
        p_dens = eval_p_t_at_strikes(
            K_values, ft["I_t"], ft["Vbar_t"], ft["A_t"],
            ft["R_t"], params.S0, params.rho
        )
        theta_hat, denom, is_stable = estimate_theta_surface(V_t, p_dens)
        theta_surface[i_t] = theta_hat
        stable_surface[i_t] = is_stable

    return {
        "tag": tag,
        "t_fracs": t_fracs,
        "K_values": K_values.tolist(),
        "theta_surface": theta_surface.tolist(),
        "stable_surface": stable_surface.tolist(),
    }


def main() -> dict:
    all_metrics = {}

    params = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.5, rho=-0.7, r=0.05, d=0.0, T=1.0, n_steps=300, seed=42,
    )

    # Step 1: Equivalence test
    print("[Exp3] Equivalence test...", flush=True)
    eq = run_estimator_equivalence(params, N=1000, t_frac=0.5)
    all_metrics["equivalence"] = eq
    print(f"  max_abs_diff={eq['max_abs_diff']:.3e}, max_rel_diff={eq['max_rel_diff']:.3e}", flush=True)
    print(f"  stable nodes: {eq['n_stable']}/{eq['n_total']}", flush=True)

    # Step 2: Convergence study
    print("[Exp3] Convergence study...", flush=True)
    conv = run_convergence_exp(params, N_values=[100, 300, 1000, 3000], N_ref=10000, n_runs=3)
    all_metrics["convergence"] = conv
    _rmse_atm = [row[0] for row in conv["rmse"]]   # first K column (ATM)
    print(f"  RMSE at K=S0 vs N: {[f'{r:.3e}' for r in _rmse_atm]}", flush=True)

    # Step 3: Maturity-strike surface
    print("[Exp3] Computing theta surface...", flush=True)
    surf = run_maturity_strike_surface(
        params, N=2000,
        K_values=np.linspace(0.5, 2.0, 15) * params.S0,
    )
    all_metrics["surface"] = surf

    # ── Robustness: vary H and rho ──
    for H_val, rho_val in [(0.05, -0.5), (0.3, -0.9)]:
        p = RoughHestonParams(H=H_val, rho=rho_val, n_steps=300, seed=99)
        tag = f"H{H_val:.2f}_rho{rho_val:.1f}"
        print(f"[Exp3] {tag}...", flush=True)
        eq_r = run_estimator_equivalence(p, N=500, t_frac=0.5, tag=tag)
        all_metrics[tag] = {"max_abs_diff": eq_r["max_abs_diff"], "max_rel_diff": eq_r["max_rel_diff"]}

    # ── Plots ──
    # Plot 1: Convergence RMSE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Exp 3: Rao–Blackwellized Estimator Convergence")

    K_vals = np.array(conv["K_values"])
    N_arr = conv["N_values"]
    rmse_arr = np.array(conv["rmse"])  # (n_N, n_K)

    for i_k in range(len(K_vals)):
        axes[0].loglog(N_arr, rmse_arr[:, i_k], marker="o", label=f"K={K_vals[i_k]:.2f}")
    # Reference N^{-0.5} line
    ref_line = rmse_arr[0, 0] * np.sqrt(N_arr[0]) / np.sqrt(np.array(N_arr))
    axes[0].loglog(N_arr, ref_line, "k--", alpha=0.5, label="N^{-0.5}")
    axes[0].set_xlabel("N (outer paths)")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("RMSE convergence")
    axes[0].legend()

    # Plot 2: Theta surface heatmap
    t_fracs_arr = np.array(surf["t_fracs"])
    K_surf = np.array(surf["K_values"])
    theta_arr = np.array(surf["theta_surface"])

    im = axes[1].contourf(
        K_surf / params.S0, t_fracs_arr, theta_arr,
        levels=20, cmap="viridis"
    )
    plt.colorbar(im, ax=axes[1])
    axes[1].set_xlabel("K / S0")
    axes[1].set_ylabel("t / T")
    axes[1].set_title("Theta(t,K) = E[V_t | S_t=K]")

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "exp3_rao_blackwell.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    all_metrics["figures"] = [fname]

    # Save
    path = os.path.join(RESULTS_DIR, "exp3_metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"[Exp3] Saved to {path}", flush=True)
    return all_metrics


if __name__ == "__main__":
    main()
