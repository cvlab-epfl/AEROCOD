# Methods

This folder contains the background documentation for the models used in this repository.

## Contents

- `notation.md`  
  Common notation used across the documentation.

- `2d_section_method.md`  
  2D airfoil parameterization, NeuralFoil sampling, and surrogate construction.

- `reduced_model.md`  
  Reduced-order representation used inside the optimization pipeline.

- `nested_ocp_and_gradients.md`  
  Nested trajectory and shape optimization, including gradient flow through the coupled system.

- `3d_nonlinear_llt_method.md`  
  3D wing formulation: sweep-corrected nonlinear lifting-line theory with Picard iteration and
  implicit-function-theorem (IFT) gradients. Covers panel layout, influence matrices, convergence
  criterion, force/moment integration, and the backward pass.

- `unsteady_corrections.md`  
  Quasi-unsteady aerodynamic corrections for pitching motion: Sugar-Gabor (2018) Terms 2 & 3
  (rate-of-circulation and added-mass forces) and the Beddoes-Leishman attached-flow lift lag.
  Covers derivation from first principles, exact code mapping, sign conventions, and YAML flags.

## Scope

The methodological core of the repository is the 2D pipeline. The 3D formulation extends the same
sectional modeling logic to finite wings through spanwise coupling.

Unsteady corrections are implemented in the trajectory OCP (`glider_jinenv.py`) and are optional:
controlled by `unsteady` and `cl_lag_enabled` flags in `conf/test.yaml`.

## Relation to the repository

- `README.md` gives an overview of the repository
- `tutorials/` contains runnable examples
- `methods/` explains the underlying models
