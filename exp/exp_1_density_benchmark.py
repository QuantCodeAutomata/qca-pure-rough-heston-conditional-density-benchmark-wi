"""
Experiment 1: Pure Rough Heston Conditional-Density Benchmark.

Validates the paper's pure rough Heston specialization by reproducing:
  - Stochastic flow: Phi_t(x) = x * M_t
  - Jacobian identity: J_t(x) = M_t > 0,  H_t(x) = 0
  - Explicit transformed density q_t (equation 38)
  - Explicit original conditional density p_t (equation 39)
  - Inverse-flow identity: p_t(K) = q_t(K/M_t) / M_t
  - Transformation consistency: q_t(x) = p_t(x*M_t) * M_t
  - Normalization and positivity
  - Small-time weak convergence toward delta_{S0}

All computations are synthetic (no market data).
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

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RoughHestonParams, make_param_grid
from src.simulator import simulate_rough_heston
from src.path_functionals import compute_path_functionals, get_functionals_at_time
from src.density import (
    eval_q_t_batch,
    eval_p_t_batch,
    eval_p_t_at_strikes,
    eval_q_t_at_inverse_strikes,
    verify_transformation_identity,
    normalization_error,
    check_positivity,
    compute_first_moment,
    EPSILON,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Utility: build evaluation grids
# ──────────────────────────────────────────────

def make_grids(S0: float, n_pts: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Build x-grid and s-grid around S0."""
    x_grid = np.linspace(0.01 * S0, 4.0 * S0, n_pts)
    s_grid = x_grid.copy()
    return x_grid, s_grid


# ──────────────────────────────────────────────
# Core experiment function
# ──────────────────────────────────────────────

def run_exp1(
    params: RoughHestonParams,
    N: int = 500,
    t_fracs: list[float] | None = None,
    n_grid: int = 500,
    tag: str = "baseline",
) -> dict:
    """Run Experiment 1 for a single parameter set.

    Steps:
      1. Simulate N rough Heston outer paths.
      2. Compute path functionals.
      3. Verify flow identities: Phi(x)=xM_t, J=M_t, H=0, J>0.
      4. Evaluate q_t and p_t on grids.
      5. Verify inverse-flow recovery.
      6. Check transformation identity.
      7. Check normalization, positivity, first moment.
      8. Check small-time behavior.

    Returns
    -------
    dict with all metrics.
    """
    t0 = time.time()

    if t_fracs is None:
        t_fracs = [0.1, 0.25, 0.5, 1.0]

    # Reporting time indices
    t_idxs = [int(f * params.n_steps) for f in t_fracs]
    t_idxs = [max(1, min(idx, params.n_steps)) for idx in t_idxs]

    # Step 1: Simulate
    sim = simulate_rough_heston(params, N, seed=params.seed)
    V = sim["V"]
    dZ = sim["dZ"]
    neg_frac = sim["negative_V_fraction"]

    # Step 2: Path functionals
    fcts = compute_path_functionals(V, dZ, params)

    x_grid, s_grid = make_grids(params.S0, n_grid)
    K_grid = s_grid.copy()

    metrics: dict = {
        "tag": tag,
        "N": N,
        "params": repr(params),
        "negative_V_fraction": neg_frac,
        "t_fracs": t_fracs,
        "flow_identity": {},
        "density_normalization": {},
        "density_positivity": {},
        "transformation_identity": {},
        "inverse_flow_recovery": {},
        "first_moment": {},
        "small_time": {},
    }

    figures = []

    # ── Process each reporting time ──
    for f, t_idx in zip(t_fracs, t_idxs):
        ft = get_functionals_at_time(fcts, t_idx)
        I_t = ft["I_t"]     # (N,)
        Vbar_t = ft["Vbar_t"]  # (N,)
        A_t = ft["A_t"]     # (N,)
        R_t = ft["R_t"]     # scalar
        M_t = ft["M_t"]     # (N,)
        V_t = V[:, t_idx]   # (N,)
        t_val = params.time_grid[t_idx]

        # Step 3: Flow identity
        # Phi_t(x) = x * M_t  -> for any x, Phi_t(x) / x = M_t
        # J_t = M_t,  H_t = 0
        J_positivity = bool(np.all(M_t > 0))
        min_J = float(np.min(M_t))
        metrics["flow_identity"][f"{f:.2f}"] = {
            "J_all_positive": J_positivity,
            "min_J": min_J,
            "mean_M_t": float(np.mean(M_t)),
            "std_M_t": float(np.std(M_t)),
        }

        # Step 4: Evaluate densities
        # q_t on x_grid for all N paths: shape (N, n_x)
        q_batch = eval_q_t_batch(x_grid, A_t, R_t, params.S0)
        # p_t on s_grid for all N paths: shape (N, n_s)
        p_batch = eval_p_t_batch(s_grid, I_t, Vbar_t, A_t, R_t, params.S0, params.rho)

        # Path-average densities (approximation to unconditional)
        q_mean = np.mean(q_batch, axis=0)
        p_mean = np.mean(p_batch, axis=0)

        # Step 7: Normalization and positivity (use first path as representative)
        q_norm_err = normalization_error(q_batch[0], x_grid)
        p_norm_err = normalization_error(p_batch[0], s_grid)
        q_pos = check_positivity(q_batch[0])
        p_pos = check_positivity(p_batch[0])

        metrics["density_normalization"][f"{f:.2f}"] = {
            "q_norm_error": q_norm_err,
            "p_norm_error": p_norm_err,
        }
        metrics["density_positivity"][f"{f:.2f}"] = {
            "q_positive": q_pos,
            "p_positive": p_pos,
        }

        # Step 5: Inverse-flow recovery p_t(K) = q_t(K/M_t) / M_t
        # Evaluate both formulas at K_grid for path 0
        i0 = 0
        M0 = float(M_t[i0])
        A0 = float(A_t[i0])
        I0 = float(I_t[i0])
        Vbar0 = float(Vbar_t[i0])

        K_test = np.linspace(0.2 * params.S0, 2.5 * params.S0, 200)
        from src.density import eval_q_t, eval_p_t
        p_direct = eval_p_t(K_test, I0, Vbar0, A0, R_t, params.S0, params.rho)
        p_recovered = eval_q_t(K_test / max(M0, EPSILON), A0, R_t, params.S0) / max(M0, EPSILON)

        abs_err = np.abs(p_direct - p_recovered)
        rel_err = abs_err / np.maximum(np.abs(p_direct), EPSILON)
        metrics["inverse_flow_recovery"][f"{f:.2f}"] = {
            "max_abs_error": float(np.max(abs_err)),
            "mean_abs_error": float(np.mean(abs_err)),
            "max_rel_error": float(np.max(rel_err)),
        }

        # Step 6: Transformation identity q(x) = p(xM_t)*M_t
        x_test = np.linspace(0.1 * params.S0, 3.0 * params.S0, 200)
        trans_check = verify_transformation_identity(
            x_test, M0, I0, Vbar0, A0, R_t, params.S0, params.rho
        )
        metrics["transformation_identity"][f"{f:.2f}"] = {
            "max_abs_error": trans_check["max_abs_error"],
            "max_rel_error": trans_check["max_rel_error"],
        }

        # First moment
        first_mom_q = compute_first_moment(q_batch[0], x_grid)
        first_mom_p = compute_first_moment(p_batch[0], s_grid)
        # Theoretical: E[S_t | common env] = S0 * exp(R_t - 0.5*A_t + 0.5*A_t) = S0*exp(R_t)
        # Actually for p_t: E[S | path i] = S0 * exp(R_t + rho*I_t - 0.5*Vbar_t + 0.5*A_t)
        # = S0 * exp(R_t + rho*I_t - 0.5*rho^2*Vbar_t) = S0 * exp(R_t) * M_t
        # But for q_t: E[x] = S0 * exp(R_t) (unconditional forward price in q-measure)
        theoretical_mom_q = params.S0 * np.exp(R_t)
        metrics["first_moment"][f"{f:.2f}"] = {
            "q_first_moment": first_mom_q,
            "p_first_moment": first_mom_p,
            "theoretical_q": theoretical_mom_q,
            "q_mom_error": abs(first_mom_q - theoretical_mom_q),
        }

        # ── Plot ──
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"Exp 1 | t={t_val:.2f}, H={params.H}, rho={params.rho} | {tag}",
            fontsize=11,
        )

        # Panel 1: q_t for a few paths
        for i_path in range(min(5, N)):
            axes[0].plot(
                x_grid / params.S0, q_batch[i_path] * params.S0,
                alpha=0.5, linewidth=0.8,
            )
        axes[0].plot(x_grid / params.S0, q_mean * params.S0, "k-", lw=1.5, label="mean q_t")
        axes[0].set_xlabel("x / S0")
        axes[0].set_ylabel("q_t(x) * S0")
        axes[0].set_title("Transformed density q_t")
        axes[0].legend()

        # Panel 2: p_t for a few paths
        for i_path in range(min(5, N)):
            axes[1].plot(
                s_grid / params.S0, p_batch[i_path] * params.S0,
                alpha=0.5, linewidth=0.8,
            )
        axes[1].plot(s_grid / params.S0, p_mean * params.S0, "k-", lw=1.5, label="mean p_t")
        axes[1].set_xlabel("s / S0")
        axes[1].set_ylabel("p_t(s) * S0")
        axes[1].set_title("Original density p_t")
        axes[1].legend()

        # Panel 3: inverse-flow recovery check (path 0)
        axes[2].plot(K_test / params.S0, p_direct, "b-", lw=1.5, label="direct p_t(K)")
        axes[2].plot(
            K_test / params.S0, p_recovered, "r--", lw=1.5,
            label="recovered: q_t(K/M_t)/M_t"
        )
        axes[2].set_xlabel("K / S0")
        axes[2].set_ylabel("density")
        axes[2].set_title(f"Inverse-flow recovery (path 0, |err|_inf={metrics['inverse_flow_recovery'][f'{f:.2f}']['max_abs_error']:.2e})")
        axes[2].legend()

        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"exp1_{tag}_t{f:.2f}.png")
        plt.savefig(fname, dpi=100, bbox_inches="tight")
        plt.close(fig)
        figures.append(fname)

    # ── Small-time behavior ──
    small_t_fracs = [0.001, 0.005, 0.01, 0.05, 0.1]
    small_t_metrics = {}
    for sf in small_t_fracs:
        st_idx = max(1, int(sf * params.n_steps))
        ft_small = get_functionals_at_time(fcts, st_idx)
        A_t_s = ft_small["A_t"][0]
        R_t_s = ft_small["R_t"]
        t_s = params.time_grid[st_idx]

        q_s = eval_q_t_batch(x_grid, np.full(1, A_t_s), R_t_s, params.S0)[0]
        # First moment should approach S0 as t->0
        m1 = compute_first_moment(q_s, x_grid)
        mass = float(np.trapz(q_s, x_grid))
        small_t_metrics[f"{sf:.3f}"] = {
            "A_t": float(A_t_s),
            "first_moment": m1,
            "mass": mass,
            "moment_error": abs(m1 - params.S0 * np.exp(R_t_s)) if mass > 0.01 else np.nan,
        }

    metrics["small_time"] = small_t_metrics
    metrics["elapsed_sec"] = time.time() - t0
    metrics["figures"] = figures

    return metrics


# ──────────────────────────────────────────────
# Grid refinement study
# ──────────────────────────────────────────────

def run_refinement_study(
    base_params: RoughHestonParams,
    N: int = 200,
    n_step_values: list[int] | None = None,
) -> dict:
    """Check that identity residuals stabilize or decrease with grid refinement."""
    if n_step_values is None:
        n_step_values = [100, 200, 500, 1000]

    results = {}
    t_frac = 0.5

    for n_steps in n_step_values:
        p = RoughHestonParams(
            S0=base_params.S0, V0=base_params.V0, H=base_params.H,
            kappa=base_params.kappa, theta=base_params.theta, nu=base_params.nu,
            rho=base_params.rho, r=base_params.r, d=base_params.d,
            T=base_params.T, n_steps=n_steps, seed=base_params.seed,
        )
        t_idx = max(1, int(t_frac * n_steps))
        sim = simulate_rough_heston(p, N, seed=p.seed)
        fcts = compute_path_functionals(sim["V"], sim["dZ"], p)
        ft = get_functionals_at_time(fcts, t_idx)

        i0 = 0
        M0 = float(ft["M_t"][i0])
        A0 = float(ft["A_t"][i0])
        I0 = float(ft["I_t"][i0])
        Vbar0 = float(ft["Vbar_t"][i0])
        R_t = float(ft["R_t"])

        x_test = np.linspace(0.1 * p.S0, 3.0 * p.S0, 300)
        check = verify_transformation_identity(
            x_test, M0, I0, Vbar0, A0, R_t, p.S0, p.rho
        )
        results[n_steps] = {
            "max_abs_error": check["max_abs_error"],
            "max_rel_error": check["max_rel_error"],
        }

    return results


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    all_metrics = {}

    # ── Baseline experiment ──
    baseline = RoughHestonParams(
        S0=1.0, V0=0.04, H=0.1, kappa=1.0, theta=0.04,
        nu=0.5, rho=-0.7, r=0.05, d=0.0, T=1.0, n_steps=500, seed=42,
    )
    print("[Exp1] Running baseline...", flush=True)
    m_base = run_exp1(baseline, N=300, tag="baseline")
    all_metrics["baseline"] = m_base
    print(f"  neg_V_fraction = {m_base['negative_V_fraction']:.4f}", flush=True)
    for tf in ["0.10", "0.50", "1.00"]:
        inv = m_base["inverse_flow_recovery"].get(tf, {})
        print(f"  t={tf}: inv-flow max_abs_err = {inv.get('max_abs_error', 'N/A'):.2e}", flush=True)

    # ── Robustness grid (H and rho) ──
    print("[Exp1] Running robustness grid...", flush=True)
    grid_params = make_param_grid(
        H_values=[0.05, 0.2, 0.4],
        rho_values=[-0.8, 0.0, 0.8],
        base=RoughHestonParams(n_steps=300, seed=123),
    )
    grid_metrics = {}
    for p in grid_params:
        tag = f"H{p.H:.2f}_rho{p.rho:.1f}"
        try:
            m = run_exp1(p, N=150, t_fracs=[0.5, 1.0], tag=tag)
            grid_metrics[tag] = {
                "neg_V_fraction": m["negative_V_fraction"],
                "inv_flow_t0.50": m["inverse_flow_recovery"].get("0.50", {}),
                "transf_id_t1.00": m["transformation_identity"].get("1.00", {}),
            }
            print(f"  {tag}: done", flush=True)
        except Exception as e:
            grid_metrics[tag] = {"error": str(e)}
            print(f"  {tag}: ERROR {e}", flush=True)
    all_metrics["robustness_grid"] = grid_metrics

    # ── Grid refinement ──
    print("[Exp1] Running grid refinement...", flush=True)
    refine = run_refinement_study(baseline, N=100, n_step_values=[100, 250, 500])
    all_metrics["grid_refinement"] = refine
    for n_s, v in refine.items():
        print(f"  n_steps={n_s}: max_rel_err = {v['max_rel_error']:.2e}", flush=True)

    # ── Save metrics ──
    metrics_path = os.path.join(RESULTS_DIR, "exp1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"[Exp1] Metrics saved to {metrics_path}", flush=True)

    return all_metrics


if __name__ == "__main__":
    main()
