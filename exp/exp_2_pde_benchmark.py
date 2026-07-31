"""
Experiment 2: Pathwise Fokker–Planck PDE Benchmark in Pure Rough Heston.

Validates that, after freezing a common-environment path, the transformed
conditional density q_t solves a deterministic log-space advection-diffusion PDE:

  partial_t qbar_t(y) = -partial_y [(b_t - 0.5*c_t) qbar_t(y)]
                         + 0.5*c_t * partial_yy qbar_t(y)

where b_t = r-d and c_t = (1-rho^2)*V_t^i.

Compares the numerical PDE solution to the exact transformed density q_t
from equation (38) and checks L1/L2/Linf errors, mass conservation,
and inverse-flow recovery of p_t.
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
from src.density import eval_q_t, eval_p_t, EPSILON
from src.pde_solver import solve_pde_log_space, compute_pde_error_vs_exact

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pde_benchmark(
    params: RoughHestonParams,
    path_idx: int = 0,
    reporting_t_fracs: list[float] | None = None,
    n_y_values: list[int] | None = None,
    tag: str = "baseline",
) -> dict:
    """Run the PDE benchmark for a single parameter set.

    Parameters
    ----------
    params : RoughHestonParams
    path_idx : int
        Which outer path to use as the frozen environment.
    reporting_t_fracs : list of float
        Fractions of T at which to report solutions.
    n_y_values : list of int
        Spatial grid sizes for convergence study.
    tag : str

    Returns
    -------
    dict with all metrics.
    """
    t0 = time.time()
    if reporting_t_fracs is None:
        reporting_t_fracs = [0.1, 0.25, 0.5, 1.0]
    if n_y_values is None:
        n_y_values = [200, 400]

    reporting_times = [max(1, int(f * params.n_steps)) for f in reporting_t_fracs]

    # Simulate outer paths and pick one
    sim = simulate_rough_heston(params, N=max(path_idx + 1, 5), seed=params.seed)
    V_path = sim["V"][path_idx]      # shape (n_steps+1,)
    dZ_path = sim["dZ"][path_idx]    # shape (n_steps,)

    # Compute path functionals for this frozen path
    V_one = V_path[np.newaxis, :]
    dZ_one = dZ_path[np.newaxis, :]
    fcts = compute_path_functionals(V_one, dZ_one, params)

    I_t_report = fcts["I_t"][0, reporting_times]      # (n_report,)
    Vbar_t_report = fcts["Vbar_t"][0, reporting_times]
    M_t_report = fcts["M_t"][0, reporting_times]

    metrics = {
        "tag": tag,
        "params": repr(params),
        "path_idx": path_idx,
        "neg_V_fraction": sim["negative_V_fraction"],
        "pde_errors": {},
        "mass_conservation": {},
        "inverse_flow_errors": {},
        "convergence": {},
    }
    figures = []

    # ── Run PDE for primary n_y ──
    n_y_primary = n_y_values[-1] if n_y_values else 400
    pde_result = solve_pde_log_space(
        params, V_path, reporting_times, n_y=n_y_primary, y_half_width=8.0
    )

    # Compute errors vs exact
    pde_errors = compute_pde_error_vs_exact(
        pde_result, params, I_t_report, Vbar_t_report, M_t_report
    )
    for i_r, tf in enumerate(reporting_t_fracs):
        key = f"t{tf:.2f}"
        metrics["pde_errors"][key] = {
            "L1": pde_errors["L1"][i_r],
            "L2": pde_errors["L2"][i_r],
            "Linf": pde_errors["Linf"][i_r],
        }
        metrics["mass_conservation"][key] = {
            "mass_at_t": float(pde_result["mass_history"][reporting_times[i_r]]),
        }

    print(f"  [PDE] n_y={n_y_primary}: Linf errors = {[f'{e:.3e}' for e in pde_errors['Linf']]}", flush=True)

    # ── Inverse-flow recovery from PDE output ──
    x_grid = pde_result["x_grid"]
    for i_r, tf in enumerate(reporting_t_fracs):
        M_t_val = float(M_t_report[i_r])
        A_t_val = float(pde_result["A_t_reporting"][i_r])
        R_t_val = float(pde_result["R_t_reporting"][i_r])
        I_t_val = float(I_t_report[i_r])
        Vbar_t_val = float(Vbar_t_report[i_r])

        q_num = pde_result["q_solutions"][i_r]
        q_exact_arr = eval_q_t(x_grid, A_t_val, R_t_val, params.S0)

        # K-grid for p_t recovery
        K_test = np.linspace(0.1 * params.S0, 3.0 * params.S0, 300)
        x_K = K_test / max(M_t_val, EPSILON)

        # Interpolate q_num at x_K
        from scipy.interpolate import interp1d
        q_interp = interp1d(x_grid, q_num, kind="linear", fill_value=0.0, bounds_error=False)
        p_from_pde = q_interp(x_K) / max(M_t_val, EPSILON)

        # Exact p_t
        p_exact = eval_p_t(K_test, I_t_val, Vbar_t_val, A_t_val, R_t_val, params.S0, params.rho)

        abs_err_p = np.abs(p_from_pde - p_exact)
        rel_err_p = abs_err_p / np.maximum(np.abs(p_exact), EPSILON)
        metrics["inverse_flow_errors"][f"t{tf:.2f}"] = {
            "max_abs": float(np.max(abs_err_p)),
            "mean_abs": float(np.mean(abs_err_p)),
            "max_rel": float(np.nanmax(rel_err_p)),
        }

    # ── Spatial convergence study ──
    conv_errors = {}
    for n_y in n_y_values:
        res_c = solve_pde_log_space(
            params, V_path, [reporting_times[-1]], n_y=n_y, y_half_width=8.0
        )
        errs_c = compute_pde_error_vs_exact(
            res_c, params,
            I_t_report[-1:], Vbar_t_report[-1:], M_t_report[-1:]
        )
        conv_errors[n_y] = {"L1": errs_c["L1"][0], "Linf": errs_c["Linf"][0]}
    metrics["convergence"] = conv_errors
    print(f"  [PDE] Convergence: {conv_errors}", flush=True)

    # ── Plot ──
    t_frac = reporting_t_fracs[-1]
    t_idx_plot = reporting_times[-1]
    i_r = len(reporting_t_fracs) - 1

    A_t_p = float(pde_result["A_t_reporting"][-1])
    R_t_p = float(pde_result["R_t_reporting"][-1])
    q_num_plot = pde_result["q_solutions"][-1]
    q_exact_plot = eval_q_t(x_grid, A_t_p, R_t_p, params.S0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Exp 2 PDE Benchmark | t={t_frac:.2f}, H={params.H}, rho={params.rho} | {tag}")

    # Panel 1: q_t comparison
    mask = (x_grid > 0.1 * params.S0) & (x_grid < 4.0 * params.S0)
    axes[0].plot(x_grid[mask] / params.S0, q_num_plot[mask] * params.S0, "b-", lw=1.5, label="PDE")
    axes[0].plot(x_grid[mask] / params.S0, q_exact_plot[mask] * params.S0, "r--", lw=1.5, label="Exact")
    axes[0].set_xlabel("x / S0")
    axes[0].set_ylabel("q_t(x) * S0")
    axes[0].set_title("q_t: PDE vs Exact")
    axes[0].legend()

    # Panel 2: pointwise error
    abs_err_plot = np.abs(q_num_plot - q_exact_plot)
    axes[1].semilogy(x_grid[mask] / params.S0, abs_err_plot[mask] + EPSILON, "g-")
    axes[1].set_xlabel("x / S0")
    axes[1].set_ylabel("|q_PDE - q_exact|")
    axes[1].set_title(f"Pointwise error (Linf={pde_errors['Linf'][-1]:.2e})")

    # Panel 3: mass conservation
    t_arr = params.time_grid
    axes[2].plot(t_arr, pde_result["mass_history"], "m-")
    axes[2].axhline(1.0, color="k", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("mass")
    axes[2].set_title("Mass conservation")

    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"exp2_{tag}.png")
    plt.savefig(fname, dpi=100, bbox_inches="tight")
    plt.close(fig)
    figures.append(fname)
    metrics["figures"] = figures
    metrics["elapsed_sec"] = time.time() - t0
    return metrics


def main() -> dict:
    all_metrics = {}

    params = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.5, rho=-0.7, r=0.05, d=0.0, T=1.0, n_steps=500, seed=42,
    )

    print("[Exp2] Baseline PDE benchmark...", flush=True)
    m = run_pde_benchmark(params, path_idx=0, n_y_values=[200, 300], tag="baseline")
    all_metrics["baseline"] = m
    print(f"  Linf at T: {m['pde_errors'].get('t1.00', {}).get('Linf', 'N/A')}", flush=True)

    # Robustness: vary H
    for H_val in [0.05, 0.3]:
        p = RoughHestonParams(H=H_val, rho=-0.5, n_steps=400, seed=42)
        tag = f"H{H_val:.2f}"
        print(f"[Exp2] {tag}...", flush=True)
        m_h = run_pde_benchmark(p, n_y_values=[200], tag=tag)
        all_metrics[tag] = m_h

    # Save
    path = os.path.join(RESULTS_DIR, "exp2_metrics.json")
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"[Exp2] Metrics saved to {path}", flush=True)
    return all_metrics


if __name__ == "__main__":
    main()
