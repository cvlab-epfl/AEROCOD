# Nested Trajectory Optimisation and Gradient Flow

**Implementation files:**
- `glider_optimization/blocks/ocp.py` — inner OCP, sensitivity extraction
- `glider_optimization/blocks/evaluation.py` — objective, outer gradient seed
- `glider_optimization/utils/go_safe_pdp.py` — COC-PDP system (Safe-PDP framework)
- `glider_optimization/utils/idoc_ineq.py` — IDOC sensitivity identities
- `glider_optimization/utils/glider_jinenv.py` — glider dynamics (CasADi)
- `glider_optimization/runner.py` — outer optimisation loop

---

## 1. Problem Structure

The overall optimisation is a **nested (bilevel) program**:

```
min_{ψ}  J(x*(p(ψ)), p(ψ))

where:
  ψ        Kulfan shape parameters (18–36 reals)
  p(ψ)     Chebyshev surrogate coefficients (φ_CL, φ_CD, φ_CM) — affine in Y(ψ)
  x*(p)    Optimal trajectory (inner OCP solution)
  J        Outer objective (terminal cost or flight time)
```

The outer loop is a gradient-descent step on `ψ`; the inner solve is a nonlinear OCP
solved by IPOPT at every outer iteration.

---

## 2. Inner OCP (Trajectory Optimisation)

### Dynamics

The glider state `X = [x, z, θ, φ, ẋ, ż, θ̇, (CL_f,) dt]` evolves under RK4:

```
X_{k+1} = X_k + (dt/6)(k1 + 2k2 + 2k3 + k4)
```

where `f(X, U, p)` is the CasADi glider dynamics (`glider_jinenv.py`).
See `unsteady_corrections.md` for the full state vector; `dt = X[-1]` always.

**Control:** `U = [elevator deflection rate, (other)]`

**Auxiliary variable** (OCP parameter): `p = vertcat(φ_CL, φ_CD, φ_CM)`.

### Solver

IPOPT via the Safe-PDP `COCsys` wrapper (`go_safe_pdp.py`). Settings:

| Parameter       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `horizon`       | 111     | Number of time steps                 |
| `timeVarying`   | `True`  | Free time-step `dt` (last state)     |
| `warm_start`    | auto    | Warm-start from previous IPOPT solve |

Path inequality constraints enforced at every node:
- Elevator deflection: `φ ∈ [−π/3, π/8]`  (`X[3]`)
- Control (elevator deflection rate): `|U| ≤ 13`

(Called as `env.initConstraints(-π/3, π/8, 13)` in `ocp.py`.)

### Multiple initial conditions

Several initial states are solved in parallel (multiprocessing) to cover different
flight regimes (perching, soft-landing, etc.). The OCP block aggregates results.

---

## 3. Objective

Evaluated by `evaluation.py` after each inner solve.

### Mode: OCP cost (perching / soft-landing)

```
J = (1/N) Σ_i  cost_i
```

where `cost_i` is the IPOPT terminal cost of trajectory `i`:

```
terminal_cost = Σ_k  w_k · x_k(T)²
```

Weights `w` from `ocp.terminal_state_weight` (dim 8, automatically padded with zeros
if the state is extended with `CL_f`).

### Mode: minimum time

```
J = (1/N) Σ_i  Σ_k  dt_k^{(i)}
```

`dt` is always at index `X[:, -1]` in the state trajectory.

---

## 4. Outer Gradient: IDOC / COC-PDP Sensitivity

The total gradient `dJ/dψ` is assembled through a chain of Jacobians:

```
dJ/dψ = (dJ/dp) · (dp/dφ) · (dφ/dY) · (dY/dψ)
         OCP sens   affine    ridge^-1  NeuralFoil
```

### Step 1 — OCP sensitivity: `dJ/dp` (Evaluation → OCP)

Upstream gradient `dJ/dX ∈ ℝ^{T × n_state}` (sparse: non-zero only at terminal node
for cost mode, or uniformly on `dt` column for time mode) is propagated backward
through the optimal control problem using the **Implicit Differentiation of Optimal
Control (IDOC)** identities.

`getAuxSys(opt_sol, threshold)` extracts the KKT-system Jacobians from the IPOPT
solution:

```
Blocks: Lxx, Luu, Lxu (Hessians of Lagrangian)
        dynFx, dynFu   (Jacobians of dynamics)
        GbarHx, GbarHu (active-constraint Jacobians)
```

`build_blocks_idoc(auxsys_COC)` assembles the block-structured matrices `H`, `A`, `B`,
`C` of the IDOC linear system (see `idoc_ineq.py`).

`idoc_full(idoc_ctx)` solves the IDOC system to obtain
`deps/dphi ∈ ℝ^{T × n_state × n_auxvar}` (the sensitivity of each state at each node
w.r.t. each surrogate coefficient).

The OCP gradient is:

```python
dJ_dphi = einsum('ij,ijk->k', dJ_deps, deps_dphi)   # (n_auxvar,)
```

Averaged over `N` trajectories:

```
total_dJ_dphi = (1/N) Σ_i  dJ_dphi^{(i)}
```

### Step 2 — Surrogate gradient: `dJ/dY` (OCP → ReducedModel)

`dJ_dphi` is passed to `ReducedModel.backward()`:

```
dJ_dY = dJ_dphi @ normal_lhs          # (n_samples,) per coefficient row
```

Combined for all three outputs: `dJ_dY ∈ ℝ^{3 × n_samples}`.

### Step 3 — Shape gradient: `dJ/dψ` (ReducedModel → NeuralFoilSampling → Airfoil)

`NeuralFoilSampling.backward()` re-runs the grid evaluation with gradients, then:

```python
grad = autograd.grad(Y, [upper, lower, LE, TE], grad_outputs=dJ_dY)
```

The constraint Lagrangian adds `grad_lagrangian` to enforce confidence and `CL/CD`.

`Airfoil.backward()` receives the Kulfan gradients and calls `optimizer.step()`.

---

## 5. Block Pipeline Summary

```
Forward pass (left → right):
  Airfoil  →  NeuralFoilSampling  →  ReducedModel  →  OCP  →  Evaluation

Backward pass (right → left):
  Evaluation  →  OCP  →  ReducedModel  →  NeuralFoilSampling  →  Airfoil
```

At each backward step, the block receives the upstream gradient dict, transforms it,
and returns the next upstream gradient dict. The Airfoil block applies the final
`optimizer.step()`.

---

## 6. Safe-PDP / COC-PDP Framework

The sensitivity computation uses the **Safe-PDP** framework (Jin et al. 2020), which
differentiates through the Pontryagin Maximum Principle (PMP) conditions of a
Constrained Optimal Control (COC) problem.

Compared to direct differentiation through IPOPT:
- Does not require differentiating through the solver.
- Works for active inequality constraints (complementarity correctly handled via KKT
  multipliers stored in the IPOPT solution).
- Cost: one linear system solve per trajectory per iteration.

---

## 7. Configuration

```yaml
run:
  max_outer_iters: 50

ocp:
  terminal_state_weight: [10., 10., 5., 0.01, 5., 5., 2., 0.01]
  stage_control_weight: 0.02
  initial_states_perching:
    - [-8.5, 0.0, 0.0, 0.0, 6.0, 3.0, 0.0, 0.01]

evaluation:
  mode: Perching   # or SoftLanding, Time
```

---

## 8. References

- Jin, W., Wang, Z., Yang, Z., & Mou, S. (2020). Pontryagin differentiable programming:
  An end-to-end learning and control framework. *NeurIPS 2020*.
  (Safe-PDP / COC-PDP framework implemented in `go_safe_pdp.py`.)

- Andersson, J.A.E. et al. (2019). CasADi — a software framework for nonlinear
  optimization and optimal control. *Mathematical Programming Computation*, **11**, 1-36.
  (CasADi symbolic dynamics and IPOPT interface.)

- Wächter, A. & Biegler, L.T. (2006). On the implementation of an interior-point filter
  line-search algorithm for large-scale nonlinear programming. *Mathematical Programming*,
  **106**(1), 25-57. (IPOPT solver.)

- Biegler, L.T. (2010). *Nonlinear Programming: Concepts, Algorithms, and Applications
  to Chemical Processes*, §10. SIAM. (IDOC sensitivity identities for NLP.)
