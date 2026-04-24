# Quasi-Unsteady Aerodynamic Corrections

**Implementation:** `glider_optimization/utils/glider_jinenv.py`  
**Reference:** Sugar-Gabor, O. (2018). A general numerical unsteady non-linear lifting line
theory. *The Aeronautical Journal*, 122(1254), 1199-1228. **Eq. 13.**  
**Config flags:** `unsteady`, `cl_lag_enabled`, `cl_lag_Tf` in `conf/test.yaml`

---

## 0. Notation and Sign Convention

The glider is modelled in 2D (longitudinal plane, x forward, z upward). The wing aerodynamic
centre (AC) is located at distance `l_w_ac` from the body reference along the pitch axis.

| Symbol         | Meaning                                           |
|----------------|---------------------------------------------------|
| `θ`            | Pitch angle (positive nose-up)                    |
| `θ̇`           | Pitch rate                                        |
| `ẋ, ż`         | Velocity of body reference point                  |
| `ẋ_w, ż_w`     | Velocity of wing AC (including pitch contribution)|
| `v_w`          | Speed of wing AC through air                      |
| `α_w`          | Angle of attack of wing (`= θ − atan2(ż_w, ẋ_w)`)|
| `CL_w`         | Quasi-steady lift coefficient at `α_w`            |
| `CL_f`         | Lagged (filtered) lift coefficient (BL lag state) |
| `CL_eff`       | Effective CL used in lift force (`CL_f` or `CL_w`)|
| `CD_w`         | Drag coefficient (always quasi-steady)            |
| `c, S_w`       | Mean chord and wing planform area                 |
| `ρ`            | Air density                                       |

### Wing AC velocity

The wing AC moves at the body velocity plus a contribution from pitching:

```
ẋ_w = ẋ + l_w_ac · θ̇ · sin θ
ż_w = ż − l_w_ac · θ̇ · cos θ
v_w = sqrt(ẋ_w² + ż_w² + ε)     (ε = 1e-8 avoids divide-by-zero)
```

### Lift and drag directions

In 2D the lift direction is perpendicular to `v_w` (rotated 90° counter-clockwise):

```
n̂_L  ∝  (−ż_w,  ẋ_w)      magnitude = v_w
n̂_D  ∝  (−ẋ_w, −ż_w)      magnitude = v_w   (drag, opposite to motion)
```

The steady force is:

```
F_w = ½ρ v_w S_w [ CL_eff · (−ż_w, ẋ_w)  +  CD_w · (−ẋ_w, −ż_w) ]
```

---

## 1. Sugar-Gabor Unsteady Corrections

### Source equation

Sugar-Gabor (2018) Eq. 13 gives the total aerodynamic force on a vortex panel:

```
dF = ρΓ (V × dl)  +  ρc (∂Γ/∂t) n  +  ρcΓ (dn/dt)
      ^^^Term 1^^^    ^^^Term 2^^^      ^^^Term 3^^^
```

- **Term 1:** Steady Kutta-Joukowski force (already in `F_Lw + F_Dw`).
- **Term 2:** Force from rate of change of circulation (lift direction).
- **Term 3:** Added-mass force from rotation of the bound-vortex normal (drag direction).

### Term 2 derivation

For a pitching wing, `Γ = ½ v_w c CL(α_w)`. Differentiating quasi-steadily:

```
∂Γ/∂t ≈ ½ v_w c (∂CL/∂α) · θ̇
```

Integrating `ρc (∂Γ/∂t) n` over the span (with `dS = c dy`, `n̂ = n̂_L`):

```
F_w2 = ½ ρ c S_w (∂CL/∂α) θ̇ · n̂_L · v_w
     = ½ ρ c S_w (∂CL/∂α) θ̇ · (−ż_w, ẋ_w)
```

Note: the direction vector `(−ż_w, ẋ_w)` already has magnitude `v_w`, so no explicit
`v_w` factor appears in the code.

### Term 3 derivation

The bound-vortex normal rotates at `θ̇`, so `dn/dt = θ̇ · ĉ` where `ĉ` points in the chord
direction (= drag direction `n̂_D`). Integrating `ρcΓ (dn/dt)` over the span:

```
F_w3 = ρ c · (½ v_w c CL_w) · θ̇ · ĉ · (S/c)
     = ½ ρ c S_w CL_w θ̇ · n̂_D · v_w
     = ½ ρ c S_w CL_w θ̇ · (−ẋ_w, −ż_w)
```

Same remark: the direction vector carries the `v_w` factor implicitly.

### Code (exact)

```python
# Fresh symbolic alpha so CasADi can differentiate; Re is treated as constant here.
alpha_sym = SX.sym('alpha_sym')
alpha_scaled_clamped_sym = fmax(-1.0, fmin(1.0, self.scale(alpha_sym, a0_min, a0_max)))
w_sym     = self.smooth_gate(alpha_sym, a0_min, a0_max, sharpness)
X_sym     = self.cheb_basis_2d(alpha_scaled_clamped_sym, Re_scaled_clamped, chebyshev_deg)
# Full blended surrogate (same expression used for CL_w):
CL_sym    = w_sym * dot(X_sym, phi_CL) + (1 - w_sym) * self.C_L(alpha_sym)
dCL_dalpha = substitute(gradient(CL_sym, alpha_sym), alpha_sym, alpha_w)

F_w2 = 0.5 * rho * chord * S_w * dCL_dalpha * thetadot * vertcat(-z_wdot,  x_wdot)
F_w3 = 0.5 * rho * chord * S_w * CL_w        * thetadot * vertcat(-x_wdot, -z_wdot)
```

Both terms are **O(v_w⁰)**: as `v_w → 0`, the direction vectors also → 0, so there is no
singularity at low speed. Both use `CL_w` (quasi-steady), not `CL_f`.

Note: `Re_scaled_clamped` is kept fixed during this symbolic differentiation — only `α`
is varied. This is physically correct since the instantaneous `∂CL/∂α` at the current
operating point does not depend on the Reynolds number sensitivity.

### YAML flag

```yaml
neuralFoilSampling:
  unsteady: true    # enables F_w2 and F_w3; false → both set to zero
```

---

## 2. Beddoes-Leishman Attached-Flow Lift Lag

### Physical motivation

The circulatory lift of a rapidly pitching airfoil lags behind the quasi-steady value.
The Beddoes-Leishman (BL) model approximates this as a first-order ODE (Leishman 1989,
attached-flow regime):

```
dCL_f/dt = (CL_w − CL_f) · 2 v_w / (Tf · c)
```

where `Tf ≈ 6` (non-dimensional lag time constant, expressed in chord-lengths per
half-chord convected time). The time constant `τ = Tf c / (2 v_w)` decreases with
airspeed (faster response at higher speed).

### Code (exact)

```python
_Tf    = self.config.neuralFoilSampling.cl_lag_Tf   # default 6.0
v_safe = fmax(v_w, 0.1)                              # floor to avoid τ → ∞
dCL_f_dt = (CL_w - CL_f) * 2 * v_safe / (_Tf * chord)
CL_eff   = CL_f   # lagged CL used in lift force only
```

The `fmax(v_w, 0.1)` floor keeps the lag time constant finite at hover/low speed
without introducing a singularity.

### Which CL goes where

| Quantity               | Uses `CL_w` (quasi-steady) | Uses `CL_f` (lagged) |
|------------------------|:--------------------------:|:--------------------:|
| Lift force `F_Lw`      |                            | ✓                    |
| Drag force `F_Dw`      | ✓                          |                      |
| Sugar-Gabor Term 2     | ✓ (`dCL/dα` at `α_w`)      |                      |
| Sugar-Gabor Term 3     | ✓                          |                      |
| Pitching moment `M_w`  | ✓                          |                      |

### YAML flags

```yaml
neuralFoilSampling:
  cl_lag_enabled: true    # add CL_f as OCP state; false → quasi-steady only
  cl_lag_Tf: 6.0          # non-dimensional lag time constant
```

---

## 3. State Vector and OCP Integration

### State vector

| `cl_lag_enabled` | State vector `X` (indices 0-based)                              | Dim |
|------------------|------------------------------------------------------------------|-----|
| `false`          | `[x, z, θ, φ, ẋ, ż, θ̇, t]`                                   | 8   |
| `true`           | `[x, z, θ, φ, ẋ, ż, θ̇, CL_f, t]`                             | 9   |

**`t = X[-1]` always** — the final state is the (free) time step `dt`, required by the
`timeVarying` logic in `go_safe_pdp.py`. Never access `dt` by a fixed index.

### Dynamics vector `f`

```python
# lag enabled
self.f = vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot, dCL_f_dt, 0)
# lag disabled
self.f = vertcat(xdot, zdot, thetadot, phidot, xddot, zddot, thetaddot, 0)
```

The trailing `0` is the time derivative of `dt` (dt is a parameter, not a true state).

### Initial condition for `CL_f`

`ocp.py` auto-adjusts the initial condition dimension when the state length changes:

```python
cl_lag = getattr(self.config.neuralFoilSampling, "cl_lag_enabled", False)
if cl_lag and len(s) == 8:
    s = s[:7] + [0.5] + s[7:]   # insert CL_f(0) = 0.5 at index 7
elif not cl_lag and len(s) == 9:
    s = s[:7] + s[8:]            # drop CL_f(0)
```

`CL_f(0) = 0.5` is a reasonable neutral initial guess; the lag ODE rapidly relaxes it to
the correct quasi-steady value within a few chord-lengths of travel.

### Weights and terminal cost

`evaluation.py` auto-pads the state-weight vector so that `CL_f` gets weight 0 by default:

```python
if len(eps_terminal) > len(w):
    w_eff = np.array(w + [0.] * (len(eps_terminal) - len(w)))
```

---

## 4. Compatibility with 3D LLT

Both unsteady corrections operate on the **2D sectional coefficients** (`CL_w`, `CD_w`,
`CM_w`) obtained either from the 2D NeuralFoil surrogate or from the 3D LLT output.  
The Sugar-Gabor and BL-lag blocks in `glider_jinenv.py` are independent of the
`use_3d_llt` flag.

---

## 5. References

- Sugar-Gabor, O. (2018). A general numerical unsteady non-linear lifting line theory.
  *The Aeronautical Journal*, **122**(1254), 1199-1228. DOI: 10.1017/aer.2018.12
  → Equation 13 is the direct source for Terms 2 and 3.

- Leishman, J.G. & Beddoes, T.S. (1989). A semi-empirical model for dynamic stall.
  *Journal of the American Helicopter Society*, **34**(3), 3-17.
  → Section on attached-flow indicial response; `Tf ≈ 6` from Table 1 (NACA 0012).

- Leishman, J.G. (2006). *Principles of Helicopter Aerodynamics*, §8.10. Cambridge University Press.
  → State-space BL model, first-order lag for attached flow.
