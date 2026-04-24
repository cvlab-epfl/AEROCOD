# Notation

Common symbols used across all methods documentation.

---

## Airfoil / Wing Geometry

| Symbol          | Meaning                                                    | Units   |
|-----------------|------------------------------------------------------------|---------|
| `c`             | Local chord length                                         | m       |
| `c̄`             | Mean aerodynamic chord (MAC)                               | m       |
| `S`             | Wing planform area                                         | m²      |
| `b`             | Full wingspan                                              | m       |
| `y`             | Spanwise coordinate (positive to the right)                | m       |
| `xle`           | Leading-edge x-position (local)                            | m       |
| `twist`         | Geometric twist angle (positive = nose-up at tip)          | deg     |
| `Λ`             | Leading-edge sweep angle (horizontal-plane projection)     | rad     |
| `Γ_dihedral`    | Dihedral angle                                             | deg     |
| `x_ref`         | x-position of reference moment centre                      | m       |

## Kulfan CST Airfoil Parameterisation

| Symbol                | Meaning                                       | Dim       |
|-----------------------|-----------------------------------------------|-----------|
| `upper`               | Upper-surface Bernstein weights               | (8,)      |
| `lower`               | Lower-surface Bernstein weights               | (8,)      |
| `LE`                  | Leading-edge weight (bluntness)               | scalar    |
| `TE`                  | Trailing-edge thickness                       | scalar    |
| `N1, N2`              | CST class-shape exponents (default 0.5, 1.0)  | —         |

The full parameter vector for one airfoil is `ψ = (upper, lower, LE, TE)` ∈ ℝ¹⁸.

## Aerodynamic Coefficients

| Symbol   | Meaning                               | Sign convention                    |
|----------|---------------------------------------|------------------------------------|
| `CL`     | Lift coefficient                      | Positive upward                    |
| `CD`     | Drag coefficient                      | Positive opposite to motion        |
| `CM`     | Pitching-moment coefficient (c/4)     | Positive nose-up                   |
| `α`      | Angle of attack                       | Positive nose-up, deg              |
| `Re`     | Reynolds number                       | `ρ V c / μ`                        |

## Flow

| Symbol   | Meaning                     | Units  |
|----------|-----------------------------|--------|
| `ρ`      | Air density                 | kg/m³  |
| `μ`      | Dynamic viscosity           | Pa·s   |
| `V`      | Freestream speed            | m/s    |
| `q`      | Dynamic pressure `½ρV²`    | Pa     |

## Lifting-Line Theory (LLT)

| Symbol       | Meaning                                                              |
|--------------|----------------------------------------------------------------------|
| `Γ`          | Bound circulation (per panel)                                        | 
| `Γ*`         | Converged fixed-point circulation                                    |
| `w_nf`       | Induced normal velocity at near-field control point `Γ @ D_nf.T`   |
| `w_tr`       | Induced velocity at Trefftz plane `Γ @ D_tr.T`                     |
| `D_nf`       | Near-field influence matrix (horseshoe, full)                        |
| `D_tr`       | Trefftz influence matrix (trailing legs only)                        |
| `cos Λ`      | `\|Δy\| / sqrt(Δx_{c/4}² + Δy²)` per panel                         |
| `V_n`        | `V cos Λ` — flow speed normal to leading edge                       |
| `c_eff`      | `c cos Λ` — chord projected normal to leading edge                  |
| `α_eff`      | `(α + twist) − rad2deg(atan2(w_nf, V_n))` — effective local AoA; all in degrees; `α + twist` is what the code calls `alpha_geo` |
| `β`          | Picard relaxation factor                                             |
| `ε_tol`      | Convergence tolerance (relative ℓ∞)                                 |

## Glider Dynamics (2D Longitudinal)

| Symbol      | Meaning                                                            | Units  |
|-------------|--------------------------------------------------------------------|--------|
| `x, z`      | Position (x forward, z upward)                                     | m      |
| `θ`         | Pitch angle (positive nose-up)                                     | rad    |
| `φ`         | Elevator deflection angle                                          | rad    |
| `ẋ, ż`      | Body-reference velocity                                            | m/s    |
| `θ̇`         | Pitch rate                                                         | rad/s  |
| `l_w_ac`    | Distance from body reference to wing AC (along pitch axis)         | m      |
| `ẋ_w, ż_w`  | Wing-AC velocity (includes pitch contribution)                     | m/s    |
| `v_w`       | Wing-AC airspeed `sqrt(ẋ_w² + ż_w²)`                              | m/s    |
| `α_w`       | Wing angle of attack `θ − atan2(ż_w, ẋ_w)`                        | rad    |
| `CL_w`      | Quasi-steady lift coefficient at `α_w`                             | —      |
| `CL_f`      | Beddoes-Leishman lagged lift coefficient (state when lag enabled)  | —      |
| `CL_eff`    | Effective CL in lift force (`CL_f` or `CL_w`)                     | —      |
| `dt`        | Time step (always the last OCP state `X[-1]`)                      | s      |

## Optimisation Pipeline

| Symbol         | Meaning                                                         |
|----------------|-----------------------------------------------------------------|
| `φ_CL`         | Chebyshev surrogate coefficients for `CL(α, Re)`               |
| `φ_CD`         | Chebyshev surrogate coefficients for `CD(α, Re)`               |
| `φ_CM`         | Chebyshev surrogate coefficients for `CM(α, Re)`               |
| `p`            | Auxiliary (parameter) variable in OCP = `(φ_CL, φ_CD, φ_CM)` |
| `J`            | Outer objective (terminal cost or flight time)                  |
| `∂J/∂p`        | OCP sensitivity — gradient of objective w.r.t. surrogate       |
| `∂J/∂φ`        | Gradient of objective w.r.t. Chebyshev coefficients            |
| `∂J/∂Y`        | Gradient w.r.t. NeuralFoil outputs `Y = (CL, CD, CM)`         |
| `∂J/∂ψ`        | Gradient w.r.t. Kulfan shape parameters                        |
| `λ`            | Augmented-Lagrangian multiplier for constraints                 |
| `ρ`            | Augmented-Lagrangian penalty weight                             |
