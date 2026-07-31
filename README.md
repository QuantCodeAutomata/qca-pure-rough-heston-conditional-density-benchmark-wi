# Pure Rough Heston Conditional-Density Benchmark

A fully synthetic Python repository that validates the paper's explicit stochastic-flow
and conditional-density formulas for the pure rough Heston model, solves the pathwise
Fokker–Planck PDE numerically, benchmarks the Rao–Blackwellized calibration-ratio
estimator, and stress-tests transport cancellation beyond the analytically solvable case.

All experiments are synthetic (no historical market data).

---

## Model

The spot price follows

```
dS_t = (r_t − d_t) S_t dt + S_t √V_t (ρ dZ_t + √(1−ρ²) dζ_t)
```

where V_t is a stochastic Volterra process satisfying the rough Heston equation with
Hurst exponent H ∈ (0, 1/2):

```
V_t = V_0 + ∫₀ᵗ K(t−s)[κ(θ − V_s) ds + ν √V_s dZ_s]
```

with K(u) = u^(H−½) / Γ(H+½).

---

## Repository Layout

```
src/                  Core library modules
  config.py           RoughHestonParams dataclass and parameter validation
  simulator.py        Euler–Maruyama rough Heston simulator (Z and V paths)
  path_functionals.py Computes I_t, V̄_t, A_t, R_t, M_t pathwise
  density.py          Explicit q_t (eq. 38) and p_t (eq. 39) density evaluators
  pde_solver.py       Crank–Nicolson log-space Fokker–Planck PDE solver
  rao_blackwell.py    Rao–Blackwellized estimators (eq. 30 and eq. 31)
  lsrv.py             Generic LSRV stochastic-flow solver (Φ_t, J_t, H_t)

exp/                  Experiment entry points
  exp_1_density_benchmark.py   Exp 1 — explicit density identities
  exp_2_pde_benchmark.py       Exp 2 — pathwise PDE vs explicit density
  exp_3_rao_blackwell.py       Exp 3 — Rao–Blackwellized ratio estimator
  exp_4_lsrv_transport.py      Exp 4 — generic LSRV transport cancellation

tests/                pytest test suite (89 tests)
data/                 Reserved for synthetic data (currently empty)
results/              Plots (PNG) and metrics (JSON + RESULTS.md)
```

---

## Experiments

| # | Title | Key Result |
|---|-------|-----------|
| 1 | Pure rough Heston density benchmark | Inverse-flow identity holds to ≤ 10⁻¹⁴; J_t > 0 always; density norm errors ≤ 2×10⁻¹⁶ |
| 2 | Pathwise Fokker–Planck PDE benchmark | Crank–Nicolson L∞ error ≈ 0.03 at T=1 (~0.6% relative); mass conserved to ≤ 10⁻⁶ |
| 3 | Rao–Blackwellized calibration ratio | Trans/orig estimators agree to 10⁻¹⁶; RMSE decays as N⁻⁰·⁵ |
| 4 | Generic LSRV transport cancellation | J_t > 0 and mass ≈ 1 for all nontrivial L(t,s); KDE L1 ≈ 0.25 |

---

## Installation

```bash
pip install numpy scipy matplotlib pandas arch
```

All other listed dependencies (skfolio, vectorbt, etc.) are available in the
execution environment but are not required for this purely synthetic project.

---

## Usage

Run individual experiments:

```bash
python exp/exp_1_density_benchmark.py
python exp/exp_2_pde_benchmark.py
python exp/exp_3_rao_blackwell.py
python exp/exp_4_lsrv_transport.py
```

Run the test suite:

```bash
pytest tests/ -q
```

Results (plots and JSON metrics) are written to `results/`.

---

## Key Formulas Validated

- **Stochastic flow** Φ_t(x) = x M_t, M_t = exp(ρ I_t − ½ρ² V̄_t)
- **Jacobian** J_t(x) = M_t > 0, H_t(x) = 0
- **Transformed density** (eq. 38): q_t(x) = lognormal with variance A_t = (1−ρ²) V̄_t
- **Original density** (eq. 39): p_t(s) = q_t(s/M_t)/M_t
- **Rao–Blackwellized estimator** (eq. 30/31): Θ̂(t,K) = E[V_t | S_t = K]
- **Pathwise PDE** (eq. 40): ∂_t q̄ = −(b−½c_t) ∂_y q̄ + ½c_t ∂_yy q̄

---

## Citation

Results are based on the paper's pure rough Heston specialization (equations 33, 34,
38, 39, 40) and the general LSRV framework (equations 30, 31, 32).
