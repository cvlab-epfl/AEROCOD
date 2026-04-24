# Reduced-Order Model: Chebyshev Surrogate for Aerodynamic Coefficients

**Implementation:** `glider_optimization/blocks/reducedModel.py`

---

## Purpose

The reduced model bridges two representations of the airfoil aerodynamics:

1. **NeuralFoil** evaluates `(CL, CD, CM)` at a discrete set of `(α, Re)` points —
   expensive, non-symbolic, GPU-resident.
2. **CasADi OCP** requires an analytic (symbolic) function `CL(α, Re)` for the
   trajectory solver and its sensitivity computation.

A 2D Chebyshev polynomial is fitted to the NeuralFoil data at each optimisation
iteration. The Chebyshev coefficient vectors `φ_CL`, `φ_CD`, `φ_CM` are the
*auxiliary variables* of the OCP.

---

## 1. Chebyshev Basis

The domain `[AoA_min, AoA_max] × [Re_min, Re_max]` is mapped to `[−1, 1]²`:

```
α̃ = 2(α − AoA_min) / (AoA_max − AoA_min) − 1
R̃ = 2(Re − Re_min) / (Re_max − Re_min) − 1
```

The 1D Chebyshev polynomials are built by the three-term recurrence:

```
T_0(x) = 1
T_1(x) = x
T_n(x) = 2x T_{n-1}(x) − T_{n-2}(x)
```

The 2D basis is the outer product:

```
Φ(α̃, R̃) = T(α̃) ⊗ T(R̃)    ∈ ℝ^{(d+1)²}
```

where `d` is the polynomial degree (default: `chebyshev_degree = 17`, giving 324
basis functions).

**Design matrix:** `X ∈ ℝ^{n_samples × (d+1)²}` — rows are `Φ(α̃_i, R̃_i)`.

---

## 2. Ridge Regression

Coefficients are solved once per iteration via ridge (L2-regularised least squares):

```
φ = (X^T X + λ I)^{−1} X^T y
```

where `λ = l2_reg` (default: 0.5). The normal equation matrix `(X^T X + λ I)^{−1} X^T`
is precomputed on the first call using `torch.linalg.solve` and reused for all three
targets `(CL, CD, CM)`:

```python
normal_lhs = linalg.solve(X.T @ X + λ·I,  X.T)  # (n_coeff, n_samples)
φ_CL = normal_lhs @ CL_vec
φ_CD = normal_lhs @ CD_vec
φ_CM = normal_lhs @ CM_vec
```

The Chebyshev grid is fixed for the lifetime of the run, so `normal_lhs` is computed
only once (`_precomputed` flag).

---

## 3. Use Inside the OCP

The OCP dynamics (`glider_jinenv.py`) receive `φ_CL`, `φ_CD`, `φ_CM` as the *auxiliary
parameter vector* `p`. At runtime, the CasADi symbolic expression evaluates:

```
CL_w = w · Φ(α̃, R̃)^T · φ_CL  +  (1−w) · CL_flat(α_w)
```

where:
- `Φ(α̃, R̃)` is the Chebyshev basis vector evaluated at the current `(α_w, Re)` — the same
  basis used during fitting.
- `CL_flat(α) = 2 sin α cos α` is a flat-plate fallback (thin-airfoil theory, α in radians).
- `w = smooth_gate(α_w; AoA_min, AoA_max, k=20)` — a smooth sigmoid gate that is ≈ 1 inside
  the surrogate domain and ≈ 0 outside, ensuring the dynamics remain bounded if the trajectory
  ventures outside the sampling range.

The same blending applies to `CD_w` and `CM_w` (fallbacks: `2 sin²α` and `−0.25 · CL_flat`).

The polynomial part is fully differentiable symbolically by CasADi, enabling:
- Trajectory rollout inside IPOPT.
- Sensitivity of the optimal trajectory w.r.t. `p` (needed for the outer gradient).

**Auxiliary vector dimension:** The config class default is `chebyshev_degree = 17` but
`conf/test.yaml` sets `chebyshev_degree: 25`, giving `(25+1)² = 676` basis functions per
coefficient and a total of `3 × 676 = 2028` auxiliary variables. This matches the hard-coded
`reshape(2028, 1)` in `ocp.py`.

---

## 4. Backward Pass

The backward pass propagates the upstream gradient `dJ/dφ ∈ ℝ^{n_coeff}` (from the OCP
block) back to the NeuralFoil outputs `y ∈ ℝ^{n_samples}`:

```
dφ/dy = (X^T X + λ I)^{−1} X^T   =  normal_lhs   ∈ ℝ^{n_coeff × n_samples}
dJ/dy = (dJ/dφ) · (dφ/dy)         =  dJ_dphi @ normal_lhs
```

This is a single matrix–vector product; no iterative solve is needed here.

---

## 5. Validation

At each iteration a 20% held-out uniform random set (from NeuralFoilSampling) is used
to compute validation MSE:

```
MSE_CL = mean((CL_val − Φ_val @ φ_CL)²)
```

A warning is logged if any validation MSE exceeds `1e-2`.

---

## 6. Configuration

```yaml
reducedModel:
  chebyshev_degree: 17   # polynomial degree per axis; basis size = (degree+1)²
  l2_reg: 0.5            # ridge regularisation λ
```

---

## 7. References

- Mason, J.C. & Handscomb, D.C. (2003). *Chebyshev Polynomials*. Chapman & Hall/CRC.
  (Chebyshev basis, three-term recurrence, approximation theory.)
- Trefethen, L.N. (2013). *Approximation Theory and Approximation Practice*. SIAM.
  (Chebyshev grids, conditioning of polynomial interpolation.)
- Tikhonov, A.N. & Arsenin, V.Y. (1977). *Solutions of Ill-Posed Problems*.
  (Ridge regression / L2 regularisation.)
