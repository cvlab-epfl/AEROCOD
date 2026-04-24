# Aerodynamics Co-Design

<img width="2160" height="600" alt="perching_trajectory_evolution" src="https://github.com/user-attachments/assets/e7794cb6-54f9-4705-9d6c-ebc846d6c1e9" />

Gradient-based nested co-design of aerodynamic shape and flight trajectory for agile glider manoeuvres such as perching and soft landing.

This repository implements a differentiable bilevel optimization pipeline in which the outer loop optimizes the airfoil or wing shape, while the inner loop solves a constrained optimal control problem (OCP) for the vehicle dynamics. The dynamics use a physically informed neural network surrogate to predict aerodynamic coefficients.

The repository currently contains two related aerodynamic formulations:

- a **2D section-based formulation**, which is the methodology documented and validated in our paper, and
- a **3D lifting-line-coupled extension**, implemented in the codebase as a subsequent development.

## Repository structure

The repository is organized around three complementary layers:

```text
glider_optimization/
├── glider_optimization/   # core package
│   ├── blocks/            # differentiable pipeline blocks
│   ├── utils/             # lifting-line, geometry, and utility functions
│   ├── config.py
│   ├── runner.py
│   └── main.py
├── conf/                  # run configurations
└── methods/               # aerodynamics background and modeling notes
```

The structure of the pipeline is designed to be modular. It is possible to change the control problem by editing `glider-optimization/blocks/ocp.py`, or even skip it entirely, resulting in a fixed-point shape optimization. In that case, the evaluation function defined in `glider-optimization/blocks/evaluation.py` should be updated accordingly.


## Installation

Clone the repository and install it in editable mode:

```bash
pip install -e .
```

## Quick start

Run a configuration:

```bash
glider-opt --config conf/test.yaml --run-name demo_run
```

## Key configuration knobs

A few settings determine most of the behavior:

- `neuralFoilSampling.use_3d_llt`
  - `false`: use the 2D sectional formulation
  - `true`: use the 3D lifting-line-coupled formulation

- `neuralFoilSampling.n_samples`
  - number of aerodynamic samples used to build the local reduced-order model

- `neuralFoilSampling.neuralFoil_size`
  - NeuralFoil model size used for sectional predictions

- `reducedModel.chebyshev_degree`
  - degree of the Chebyshev surrogate used inside the OCP

- `evaluation.mode`
  - task objective, for example `Perching` or `SoftLanding`

- `plane.wing.*`
  - finite-wing geometry definition, including span stations, chord, sweep, twist, and dihedral

## Citation

If you use this repository in academic work, please cite the accompanying paper:

```
@article{affinita2026gradient,
  title={Gradient-based Nested Co-Design of Aerodynamic Shape and Control for Winged Robots},
  author={Affinita, Daniele and Xu, Mingda and Gherardi, Beno{\^\i}t Valentin and Fua, Pascal},
  journal={arXiv preprint arXiv:2603.06760},
  year={2026}
}
```

## License

This repository is licensed under Apache 2
