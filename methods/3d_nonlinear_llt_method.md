# 3D Nonlinear Lifting-Line Theory

**Implementation:** `glider_optimization/utils/llt.py`  
**Entry point:** `LLTImplicitFn.apply(...)` → returns `(B, 3)` tensor `[CL, CD, CM]`

---

## 1. Wing Discretisation

### Half-wing input stations
Spanwise stations `y_half` (root → tip) define chord `c`, leading-edge x-position `xle`,
geometric twist `twist`, and optionally vertical position `z_half` (dihedral).
If `z_half` is not provided it is computed as `z = y · tan(Γ_dihedral)`.

### Full-span mirroring
`mirror_full(y_half, c_half, xle_half, twist_half)` reflects the half-wing:

```
y_full = [-y_half[::-1], y_half[1:]]   (sorted by y)
```

Dihedral z is mirrored symmetrically: `z(-y) = z(|y|)` (both half-wings go up).

### Panels
`n_pan = n_stations - 1` panels. Each panel runs from station A to station B.

| Symbol      | Code          | Definition                                       |
|-------------|---------------|--------------------------------------------------|
| `y_mid`     | `y_mid`       | Spanwise mid-point of panel                      |
| `c_mid`     | `c_mid`       | Mean chord of panel                              |
| `dy`        | `dy`          | Panel span `|yB − yA|`                           |
| `S`         | `S`           | Wing area `Σ c_mid · dy`                         |
| `c̄`         | `cbar`        | MAC `Σ ½(cA²+cB²)·dy / S` (trapezoidal)          |

### Sweep correction

Per-panel sweep angle Λ is computed from the ¼-chord vortex segment projected onto
the horizontal plane (excludes dihedral):

```
cos Λ_i = |Δy| / sqrt(Δx_{c/4}² + Δy²)
```

This follows the standard swept-wing correction: only the component of flow **normal
to the leading edge** (magnitude `V · cos Λ`) sees the full chord.

---

## 2. Influence Matrices

Built by `build_llt_system(...)`. Control points are placed at `0.25c` (¼-chord line),
slightly below the surface (`z_cp = z_mid − 0.01 c`).

| Matrix  | Size            | Content                                                        |
|---------|-----------------|----------------------------------------------------------------|
| `D_nf`  | `(n_pan, n_pan)`| `−w_z` induced at panel `i` by unit-circulation horseshoe `j` |
| `D_tr`  | `(n_pan, n_pan)`| `−w_z` induced by trailing legs only of panel `j` (Trefftz)   |

**Sign:** `D[i,j] = -v_z` so that `w_induced = Γ @ D.T` gives a positive (downward)
induced velocity for positive circulation.

**Self-influence** (`i == j`): bound segment excluded — only trailing legs are used
(Cauchy principal value, avoids logarithmic singularity of the bound vortex).

**Wake length:** `L_wake = 20 · max(c_max, 1.0)` chord-lengths downstream.

**Core radius:** `rc_nf = 0.25 c_mid` (near-field), `rc_tr = 0.15 c_mid` (Trefftz).

---

## 3. Picard Iteration

### Sweep-corrected quantities (per panel)

```
V_n   = V · cos Λ              # flow speed component normal to LE
c_eff = c · cos Λ              # chord component normal to LE
Re    = ρ · V_n · c_eff / μ    # = ρ V c cos²Λ / μ
```

### Effective angle of attack

Induced normal velocity `w_nf = Γ @ D_nf.T`:

```
α_geo = α + twist                                    (degrees)
α_eff = α_geo − rad2deg( atan2(w_nf, V_n) )         (degrees)
```

All quantities are in **degrees** throughout the LLT: `α` is passed in degrees,
`twist` is stored in degrees, and `atan2(w_nf, V_n)` is converted to degrees via
`torch.rad2deg` before subtraction. NeuralFoil expects degrees.

`α_geo` is the freestream geometric angle of attack (scalar, broadcast to all panels).

### Circulation update (one Picard step `_G`)

NeuralFoil is queried at `(α_eff, Re)` per panel → `CL`:

```
Γ_star = ½ · V_n · c_eff · CL    =  ½ V · c · cos²Λ · CL
Γ_{k+1} = (1 − β) Γ_k + β · Γ_star
```

`β` is the Picard relaxation factor (under-relaxation). With symmetry enforcement:

```
Γ_{k+1} ← ½ (Γ_{k+1} + Γ_{k+1}[:, mirror_of])
```

### Convergence criterion

```
‖Γ_{k+1} − Γ_k‖_∞ / max(1.0, ‖Γ_k‖_∞) < ε_tol
```

Relative ℓ∞ norm with denominator clamped at 1.0 to avoid false convergence near Γ = 0.
A warning is printed if the iteration limit `max_iter` is reached without convergence.

### Initial guess

```
Re_0   = ρ · V · c · cos²Λ / μ        (no induced velocity yet)
CL_0   = NeuralFoil(α + twist, Re_0)   (α_geo, no induced correction)
Γ_0    = ½ · V · c · cos²Λ · CL_0
```

---

## 4. Force and Moment Integration

All integrals over the full span.

### Lift (Kutta-Joukowski)

```
L = ρ V Σ_i Γ_i · dy_i
```

### Induced drag (Trefftz plane)

```
w_tr = Γ @ D_tr.T          # Trefftz induced velocity
D_i  = ρ Σ_i Γ_i · w_tr,i · dy_i
```

Using the Trefftz plane (trailing-legs only) gives better momentum conservation,
consistent with FLOW5 methodology.

### Profile drag

```
D_p  = Σ_i q · c_i · cos²Λ_i · CD_i · dy_i
```

The `cos²Λ` factor: one power from `c_eff = c cos Λ` (section chord normal to LE)
and one from `V_n = V cos Λ` in the dynamic pressure seen by the section.

### Pitching moment (about ¼-chord reference line)

```
M_p  = Σ_i q · c_i² · cos³Λ_i · CM_i · dy_i
```

The extra `cos Λ` (→ `cos³Λ`) comes from the lever arm `c_eff` for the moment.

### Non-dimensionalisation

```
CL = L  / (q · S)
CD = (D_i + D_p) / (q · S)
CM = M_p / (q · S · c̄)
```

---

## 5. Airfoil Parameters (Kulfan CST)

NeuralFoil uses Kulfan CST parameterisation with 8 upper-surface weights, 8 lower-surface
weights, a leading-edge weight, and trailing-edge thickness.

Two modes are supported:

| Mode        | Shape                | Description                               |
|-------------|----------------------|-------------------------------------------|
| Global      | `upper` (8,)         | Same airfoil for every panel              |
| Per-panel   | `upper` (n_pan, 8)   | Linear interpolation root → tip           |

Per-panel parameters are expanded to `(B · n_pan, 8)` via a single batched NeuralFoil call.

---

## 6. Gradient: Implicit Function Theorem (IFT)

`LLTImplicitFn` is a custom `torch.autograd.Function` that decouples the forward fixed-point
iteration from the backward pass.

### Forward
Picard iteration (no gradient tape). Converged `Γ*` stored in `ctx`.

### Backward

**Objective:** given upstream gradient `∂L̃/∂C`, compute `∂L̃/∂ψ` where `ψ` are the
Kulfan shape parameters.

**Residual:** `F(Γ, ψ) = Γ − G(Γ, ψ) = 0` at the fixed point.

**Step 1 — Jacobian:** Build `J = ∂F/∂Γ ∈ ℝ^{B × n_pan × n_pan}` via `n_pan` full-batch
VJP passes (one per row of J):

```
J[b, i, j] = ∂F[b,i] / ∂Γ[b,j]
```

**Step 2 — Solve:**

```
J^T λ = ∂L̃/∂Γ        (batched LU via torch.linalg.solve)
```

**Step 3 — Gradient:**

```
∂L̃/∂ψ = (∂C/∂ψ)|_{Γ*}  −  λ^T · (∂F/∂ψ)|_{Γ*}
           ^^^^direct^^^^      ^^^^implicit^^^^
```

Cost of backward: `n_pan` NeuralFoil calls (independent of batch size `B`).

---

## 7. Key Parameters (YAML / config)

| Parameter            | Config field      | Effective default | Description                             |
|----------------------|-------------------|-------------------|-----------------------------------------|
| `use_3d_llt`         | `use_3d_llt`      | `false`           | Enable 3D LLT (vs. 2D strip theory)    |
| `llt_n_iter`         | `llt_n_iter`      | 20                | Picard iterations per call              |
| `llt_max_iter`       | `llt_max_iter`    | 30                | Hard iteration cap (triggers warning)   |
| `llt_beta`           | `llt_beta`        | 0.30              | Picard relaxation factor                |
| `llt_tol`            | `llt_tol`         | 1e-4              | Convergence tolerance (relative ℓ∞)     |
| `enforce_symmetry`   | (hardcoded)       | `true`            | Average Γ with mirrored panel each step |
| `neuralFoil_size`    | `neuralFoil_size` | `xxxlarge`        | NeuralFoil model size (shared with 2D) |

**Note:** effective defaults are the `getattr` fallbacks in `neuralFoilSampling3D.py`;
`config.py` class fields (`llt_n_iter: 30`, `llt_max_iter: 200`, `llt_beta: 0.5`,
`llt_tol: 1e-5`) are the YAML-overridable values, not the code fallbacks.

---

## 8. References

- Prandtl, L. (1918). *Tragflügeltheorie*. Nachrichten der Gesellschaft der Wissenschaften zu
  Göttingen. (Original LLT formulation.)
- Anderson, J.D. (2001). *Fundamentals of Aerodynamics*, §5.3. McGraw-Hill.
  (Classical Prandtl LLT and Trefftz-plane drag.)
- Katz, J. & Plotkin, A. (2001). *Low-Speed Aerodynamics*, §12. Cambridge University Press.
  (Horseshoe vortex influence matrices, Biot-Savart kernel.)
- Sugar-Gabor, O. (2018). A general numerical unsteady non-linear lifting line theory.
  *The Aeronautical Journal*, 122(1254), 1199-1228.  
  (Sweep corrections and unsteady extensions; see also `unsteady_corrections.md`.)
