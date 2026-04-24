# diffLLT — Differentiable Lifting-Line Shape Optimisation

Standalone module for optimising a wing's Kulfan CST airfoil shape directly through
the 3D nonlinear Lifting-Line Theory (LLT) solver with exact gradients via the
Implicit Function Theorem (IFT).

---

## What this is

The full `glider_optimization` pipeline optimises airfoil shape for **trajectory
performance** (perching, soft landing) using a bilevel structure:

```
ψ → Chebyshev surrogate → OCP (IPOPT) → IDOC sensitivity → ∂J/∂ψ
```

`diffLLT` removes the OCP layer entirely and optimises directly for
**aerodynamic efficiency** at a set of operating points:

```
ψ (Kulfan) → LLT(α, V, wing) → (CL, CD, CM) → J(CL, CD) → ∂J/∂ψ  (IFT)
```

Gradients are **exact** (no surrogate error) and computed in one backward pass
through the IFT adjoint already implemented in `LLTImplicitFn`.

---

## Method summary

For full derivations see:

- [`methods/3d_nonlinear_llt_method.md`](../methods/3d_nonlinear_llt_method.md) — LLT
  panel layout, Picard iteration, force integration, IFT backward.
- [`methods/2d_section_method.md`](../methods/2d_section_method.md) — Kulfan CST
  parameterisation, NeuralFoil evaluation, constraints.
- [`methods/notation.md`](../methods/notation.md) — symbol table.

### Gradient path

At the converged fixed point `Γ*(ψ)` satisfying `F(Γ, ψ) = 0`:

$$
\frac{\partial J}{\partial \psi} =
\underbrace{\frac{\partial C}{\partial \psi}\bigg|_{\Gamma^*}}_{\text{direct}}
- \underbrace{\lambda^\top \frac{\partial F}{\partial \psi}\bigg|_{\Gamma^*}}_{\text{implicit}}
\quad\text{where}\quad
J^\top \lambda = \frac{\partial \tilde{L}}{\partial \Gamma}
$$

This is handled automatically by `LLTImplicitFn.backward`; `core.py` just calls
`loss.backward()`.

### Constraints

Two soft constraints are enforced via the **augmented Lagrangian** method (identical
to the main pipeline's `NeuralFoilSampling` block):

| Constraint | Expression | Default |
|---|---|---|
| Mean lift-to-drag | `mean(CL/CD) ≥ min_avg_clcd` | ≥ 2.0 |
| TE thickness (hard) | `TE ∈ [1e-4, 0.01]` | clamped each step |

The multiplier update follows `λ ← max(0, λ + ρ·relu(g))` after every step.

---

## Public API (`core.py`)

```python
from diffLLT.core import Wing, AirfoilParams, DiffLLTEvaluator, ShapeOptimiser

# 1 — load wing geometry from YAML
wing = Wing.from_yaml("conf/test.yaml")

# 2 — define operating points
alphas    = torch.tensor([0., 2., 4., 6., 8.], device=wing.device)  # degrees
velocities = torch.tensor([10.] * 5, device=wing.device)             # m/s

# 3 — evaluate LLT (differentiable)
evaluator = DiffLLTEvaluator(wing)
airfoil   = AirfoilParams.default(wing.device)
result    = evaluator(alphas, velocities, airfoil)   # → EvalResult(CL, CD, CM)

# 4 — optimise
cost_fn = lambda CL, CD, CM: -CL.mean() / CD.mean()   # maximise CL/CD
opt = ShapeOptimiser(evaluator, cost_fn, alphas, velocities)
opt.run(airfoil, n_iters=50)
```

### `Wing.from_yaml(yaml_path, **kwargs)`

Reads `plane.wing` from the YAML (same keys as the main pipeline) and precomputes
the LLT influence matrices.  Parameters:

| kwarg | default | meaning |
|---|---|---|
| `n_span_stations` | 7 | half-wing stations → 12 full-span panels |
| `device` | auto (cuda/mps/cpu) | torch device |
| `neuralfoil_size` | `xxxlarge` | NeuralFoil model accuracy |
| `llt_beta` | 0.30 | Picard relaxation |
| `llt_tol` | 1e-4 | convergence tolerance |
| `llt_n_iter` / `llt_max_iter` | 20 / 30 | iteration limits |

### `AirfoilParams`

Holds `upper` (8,), `lower` (8,), `LE` (1,), `TE` (1,) tensors with
`requires_grad=True`.  Optionally tip arrays for spanwise variation.

### `DiffLLTEvaluator(wing)(alphas, velocities, airfoil) → EvalResult`

Runs the full 3D LLT for all B operating points in one batched call.
Returns `EvalResult.CL`, `.CD`, `.CM` of shape `(B,)` — all differentiable.

### `ShapeOptimiser.run(airfoil, n_iters) → AirfoilParams`

Adam + ExponentialLR + augmented Lagrangian loop.  Stores per-iteration metrics
in `opt.history` (loss, cl, cd, clcd, penalties).

---

## Environment note (macOS MPS)

`torch.linalg.solve` — used in the IFT backward — is not natively implemented on
MPS.  Set the fallback before starting Jupyter:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
jupyter notebook
```

Or in Python before importing torch:

```python
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
```

---

## Files

```
diffLLT/
├── README.md    ← this file
├── core.py      ← Wing, AirfoilParams, DiffLLTEvaluator, ShapeOptimiser
└── demo.ipynb   ← worked example: CL/CD optimisation from conf/test.yaml
```
```
core.py
 │
 ├── Wing                       ← pure geometry + solver config
 │    └─ Wing.from_yaml()       reads plane.wing + neuralFoilSampling from YAML
 │        ├── build_half_wing_stations_from_cfg()   interpolates y/c/xle/twist
 │        └── build_llt_system()                    computes influence matrices
 │              D_nf (near-field), D_tr (Trefftz), mirror_of, cos_sweep, ...
 │
 ├── AirfoilParams              ← lightweight tensor container, no nn.Module
 │    ├── from_yaml()           reads airfoil: block
 │    ├── default() / default_3d()  fallback initialisations
 │    ├── parameters()          list of leaf tensors → passed to Adam
 │    └── clamp_constraints()   hard-clamps TE ∈ [1e-4, 0.01] and upper ≥ lower+0.05
 │
 ├── DiffLLTEvaluator           ← wraps the existing IFT-backward solver
 │    ├── __call__()            batched forward for B operating points
 │    │     ├── interpolates root→tip Kulfan weights via η for each panel
 │    │     └── calls LLTImplicitFn.apply(...)  [defined in llt.py]
 │    │           ├── forward: Picard iteration → converged Γ* → integrate CL/CD/CM
 │    │           └── backward: IFT — solves one linear system to get ∂L/∂(upper,lower,LE,TE)
 │    └── eval_spanwise()       no-grad version; re-runs Picard, returns panel arrays
 │
 └── ShapeOptimiser             ← gradient loop, nothing new numerically
      ├── run()                 outer loop over n_iters
      └── _step()
            ├── evaluator(alphas, velocities, airfoil)   → CL, CD, CM
            ├── cost_fn(CL, CD, CM).backward()           → gradients on Kulfan weights
            ├── Adam.step() + ExponentialLR.step()
            ├── augmented Lagrangian multiplier update
            └── airfoil.clamp_constraints()              hard geometric repair
```