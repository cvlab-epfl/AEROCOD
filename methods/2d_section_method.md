# 2D Section Method: Airfoil Parameterisation, NeuralFoil Sampling, and Surrogate Construction

**Implementation files:**
- `glider_optimization/blocks/airfoil.py` — airfoil parameter block (2D)
- `glider_optimization/blocks/airfoil3D.py` — per-span-station extension (3D)
- `glider_optimization/blocks/neuralFoilSampling.py` — 2D sampling block
- `glider_optimization/blocks/neuralFoilSampling3D.py` — 3D LLT-based sampling
- `glider_optimization/utils/cu_kulfan_airfoil.py` — NeuralFoil GPU wrapper

---

## 1. Airfoil Parameterisation (Kulfan CST)

The airfoil shape is represented in the Class-Shape Transform (CST) parameterisation
(Kulfan 2008). The surface `y(x)` is expressed as:

```
y(x) = C(x) · S(x)
C(x) = x^N1 · (1-x)^N2           (class function, N1=0.5, N2=1.0)
S(x) = Σ_i w_i · B_{i,n}(x)      (shape function, Bernstein basis)
```

The optimisable parameters for one airfoil are:

| Parameter               | Symbol       | Dim  | Default initialisation        |
|-------------------------|--------------|------|-------------------------------|
| Upper surface weights   | `upper`      | (8,) | `[0.1, 0.15, 0.2, …, 0.01]`  |
| Lower surface weights   | `lower`      | (8,) | `[-0.05, …, 0.0]`             |
| Leading-edge weight     | `LE`         | (1,) | `0.0`                         |
| Trailing-edge thickness | `TE`         | (1,) | `0.0`                         |

Total: 18 real parameters per airfoil. In 3D mode, root and tip airfoils are
parameterised independently (36 parameters), with linear spanwise interpolation
for intermediate panels.

### Optimiser
Adam with exponential LR decay:

```
lr_0 = airfoil.lr          # default 1e-2
γ    = airfoil.gamma       # default 0.99
```

Gradients are computed by the upstream blocks and passed back through the backward
chain; the Airfoil block calls `optimizer.step()` at the end of each iteration.

---

## 2. NeuralFoil Evaluation

NeuralFoil (Sharpe 2024) is a neural-network surrogate for XFOIL trained on the full
Kulfan parameter space. It evaluates `(CL, CD, CM, analysis_confidence)` at arbitrary
`(α, Re)` for any CST airfoil shape.

**GPU wrapper:** `get_aero_from_kulfan_parameters_cuda(...)` in
`utils/cu_kulfan_airfoil.py` — batched CUDA call accepting `(B, 8)` parameter tensors
and `(B,)` `α`/`Re` vectors.

**Model sizes** (accuracy vs. speed): `xxsmall` → `xxxlarge`. Default: `xxxlarge`.

### Sampling grid

A structured Chebyshev grid in `(α, Re)` is used for training-point placement:

```
n_1d = sqrt(n_samples)    # points per axis, e.g. 10 for n_samples=100
α_k  = Cheb(AoA_min, AoA_max, n_1d)   # Chebyshev nodes on [-10°, 25°]
Re_k = Cheb(Re_min,  Re_max,  n_1d)   # Chebyshev nodes on [1e4, 6e5]
```

Chebyshev nodes: $x_k = \tfrac{a+b}{2} + \tfrac{b-a}{2}\cos\!\left(\tfrac{2k+1}{2n}\pi\right)$

The Chebyshev placement clusters points near the domain boundaries, which improves
polynomial surrogate conditioning.

A 20% random validation set (uniform in `(α, Re)`) is also evaluated each iteration
for out-of-distribution error monitoring.

### Forward pass

```python
# Forward (no gradient tape — avoids building a huge graph through xxxlarge)
with torch.no_grad():
    aero = get_aero_from_kulfan_parameters_cuda(kulfan_batch, α_batch, Re_batch, ...)
```

During `backward()`, the forward pass is re-run with gradients enabled.

---

## 3. Constraints (Augmented Lagrangian)

Two constraints are enforced via the augmented Lagrangian method (penalty + multiplier):

### Confidence constraint

```
g_conf = min_confidence − mean(confidence)   ≤ 0
```

`analysis_confidence ∈ [0, 1]` is NeuralFoil's internal measure of extrapolation risk.
The constraint keeps the optimised airfoil inside the surrogate's training domain.

### Lift-to-drag ratio constraint

```
g_Cl_Cd = min_avg_Cl_Cd − mean(CL / CD)   ≤ 0
```

Prevents degenerate shapes with poor aerodynamic efficiency.

### Augmented Lagrangian update

For each constraint `g_i`:

```
L_aug += λ_i · relu(g_i) + (ρ/2) · relu(g_i)²
λ_i   ← λ_i + ρ · relu(g_i)           (multiplier update, each backward call)
```

Defaults: `ρ = 1.0`. Multipliers `λ_conf`, `λ_Cl_Cd` start at 0 and grow to enforce
the constraints without a hard penalty step size.

---

## 4. Backward Pass

During backward the full sampling grid is re-run with `requires_grad=True`:

```
Y = cat([CL, CD, CM])                      # (3B,)
grad = autograd.grad(Y, [upper, lower, LE, TE], grad_outputs=dJ_dY)
grad_lagrangian = autograd.grad(L_aug, [upper, lower, LE, TE])
return grad + grad_lagrangian
```

The upstream `dJ_dY ∈ ℝ^{3B}` is provided by the ReducedModel block's backward.

---

## 5. Configuration

```yaml
airfoil:
  lr: 1e-2
  gamma: 0.99                     # LR decay per iteration
  upper_initial_weights: [0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]
  lower_initial_weights: [-0.05, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0]
  leading_edge_weight: 0.0
  TE_thickness: 0.0

neuralFoilSampling:
  neuralFoil_size: xxxlarge
  AoA_min: -10.0
  AoA_max: 25.0
  Re_min: 1.0e4
  Re_max: 6.0e5
  n_samples: 100                  # total = n_1d² grid points
  min_confidence: 0.7
  min_avg_Cl_Cd: 2.0
  rho: 1.0                        # augmented Lagrangian penalty
```

---

## 6. References

- Kulfan, B.M. (2008). Universal parametric geometry representation method.
  *Journal of Aircraft*, **45**(1), 142-158.
- Sharpe, P.D. (2024). NeuralFoil: An airfoil aerodynamics analysis tool using
  physics-informed machine learning. MIT License.
  [github.com/peterdsharpe/NeuralFoil](https://github.com/peterdsharpe/NeuralFoil)
- Nocedal, J. & Wright, S.J. (2006). *Numerical Optimization*, §17.4 (augmented
  Lagrangian methods). Springer.
