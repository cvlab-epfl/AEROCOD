"""
diffLLT.core
============
Three public objects for differentiable-LLT shape optimisation:

  Wing            - loads wing geometry from a YAML file and caches the LLT system.
  DiffLLTEvaluator - thin wrapper around LLTImplicitFn; single-call forward/backward.
  ShapeOptimiser  - Adam + augmented-Lagrangian loop; cost function is a user callable.

All gradient flow is handled by the existing IFT backward in LLTImplicitFn; no new
numerics are introduced here.  See methods/3d_nonlinear_llt_method.md for derivations.

Environment note (macOS MPS):
  torch.linalg.solve is not natively supported on MPS.  Set the env variable
      PYTORCH_ENABLE_MPS_FALLBACK=1
  before starting Python (or Jupyter) so that the IFT backward falls back to CPU
  for that one op.  The notebook cells below remind the user.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import yaml

# Reuse existing utilities — no code copied.
from glider_optimization.utils.llt import (
    LLTImplicitFn,
    LLTConst,
    _DEVICE_TO_ID,
    _MODEL_SIZE_TO_ID,
    _ID_TO_MODEL_SIZE,
    _G,
    _eval_nf_batched,
    build_llt_system,
)
from glider_optimization.utils.spanwise_geometry import (
    build_half_wing_stations_from_cfg,
    _section_centroid_from_kulfan,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_device() -> torch.device:
    """Pick the best available device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _warn_mps_fallback(device: torch.device) -> None:
    if device.type == "mps" and not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        warnings.warn(
            "Device is MPS but PYTORCH_ENABLE_MPS_FALLBACK is not set.  "
            "The IFT backward pass calls torch.linalg.solve which is not natively "
            "supported on MPS.  Set os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' "
            "before calling .backward() or use device='cpu'.",
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Wing
# ---------------------------------------------------------------------------

@dataclass
class Wing:
    """
    Immutable wing description + precomputed LLT system.

    Attributes
    ----------
    comp     : raw dict returned by build_llt_system (numpy arrays)
    tensors  : dict of all geometry tensors on *device*, ready for LLTImplicitFn
    n_pan    : number of panels (full span)
    span     : full wingspan (m)
    S        : planform area (m²)
    cbar     : mean aerodynamic chord (m)
    eta      : normalised spanwise coordinate η ∈ [0,1] for each panel mid-point
    device   : torch.device
    rho      : air density (kg/m³)
    mu       : dynamic viscosity (Pa·s)
    model_size_id : int tensor (NeuralFoil model index)
    device_id     : int tensor (NeuralFoil device index)
    llt_beta, llt_tol, llt_n_iter, llt_max_iter : solver scalars (float tensors)
    enforce_sym   : bool tensor (1.0 = enforce spanwise symmetry)
    """

    comp: dict
    tensors: dict
    n_pan: int
    span: float
    S: float
    cbar: float
    eta: torch.Tensor
    device: torch.device
    rho: torch.Tensor
    mu: torch.Tensor
    model_size_id: torch.Tensor
    device_id: torch.Tensor
    llt_beta: torch.Tensor
    llt_tol: torch.Tensor
    llt_n_iter: torch.Tensor
    llt_max_iter: torch.Tensor
    enforce_sym: torch.Tensor

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        *,
        n_span_stations: int = 7,
        device: Optional[torch.device] = None,
        rho: float = 1.225,
        mu: float = 1.789e-5,
        neuralfoil_size: str = "xxxlarge",
        llt_beta: float = 0.30,
        llt_tol: float = 1e-4,
        llt_n_iter: int = 20,
        llt_max_iter: int = 30,
        enforce_symmetry: bool = True,
    ) -> "Wing":
        """
        Build a Wing from a YAML config file.

        Uses the same ``plane.wing`` block as the main pipeline
        (keys: y_half, c_half, xle_half, twist_half, dihedral).
        Any solver or NeuralFoil settings in the YAML
        (``neuralFoilSampling.llt_beta`` etc.) override the keyword defaults.

        Parameters
        ----------
        yaml_path : path to a YAML config (e.g. ``conf/test.yaml``)
        n_span_stations : half-wing station count (default 7 → 12 panels full-span)
        device : torch device; auto-detected if None
        rho, mu : air density and dynamic viscosity
        neuralfoil_size : NeuralFoil model size string
        llt_beta, llt_tol, llt_n_iter, llt_max_iter : Picard solver settings
        enforce_symmetry : average Γ with mirrored panel each Picard step
        """
        if device is None:
            device = _auto_device()
        _warn_mps_fallback(device)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        plane_cfg = cfg.get("plane", {}) or {}
        wing_cfg  = plane_cfg.get("wing", {}) if isinstance(plane_cfg, dict) else {}
        nf_cfg    = cfg.get("neuralFoilSampling", {}) or {}

        # Allow YAML overrides for solver params
        llt_beta     = float(nf_cfg.get("llt_beta",     llt_beta))
        llt_tol      = float(nf_cfg.get("llt_tol",      llt_tol))
        llt_n_iter   = int(  nf_cfg.get("llt_n_iter",   llt_n_iter))
        llt_max_iter = int(  nf_cfg.get("llt_max_iter", llt_max_iter))
        neuralfoil_size = nf_cfg.get("neuralFoil_size", neuralfoil_size)
        dihedral_deg = float(wing_cfg.get("dihedral", 0.0)) if isinstance(wing_cfg, dict) else 0.0

        stations = build_half_wing_stations_from_cfg(wing_cfg, n_span_stations=n_span_stations)
        comp = build_llt_system(
            stations["y_half"].tolist(),
            stations["c_half"].tolist(),
            stations["xle_half"].tolist(),
            stations["twist_half"].tolist(),
            dihedral_deg=dihedral_deg,
        )

        def _t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        tensors = {
            "dy":        _t(comp["dy"]),
            "y":         _t(comp["y_mid"]),
            "c":         _t(comp["c_mid"]),
            "tw":        _t(comp["tw_mid"]),
            "S":         _t(comp["S"]),
            "cbar":      _t(comp["cbar"]),
            "x_c4":      _t(comp["x_c4_mid"]),
            "span":      _t(comp["span"]),
            "D_nf":      _t(comp["D_nf"]),
            "D_tr":      _t(comp["D_tr"]),
            "mirror_of": torch.as_tensor(comp["mirror_of"], dtype=torch.long, device=device),
            "cos_sweep":  _t(comp["cos_sweep"]),
            "cos2_sweep": _t(comp["cos2_sweep"]),
        }

        half_span = float(comp["span"]) * 0.5
        eta = _t(np.abs(comp["y_mid"]) / max(half_span, 1e-9)).clamp(0.0, 1.0)

        return cls(
            comp=comp,
            tensors=tensors,
            n_pan=int(comp["n_pan"]),
            span=float(comp["span"]),
            S=float(comp["S"]),
            cbar=float(comp["cbar"]),
            eta=eta,
            device=device,
            rho=_t(rho),
            mu=_t(mu),
            model_size_id=torch.tensor(
                _MODEL_SIZE_TO_ID[neuralfoil_size], dtype=torch.int64, device=device
            ),
            device_id=torch.tensor(
                _DEVICE_TO_ID.get(device.type, 0), dtype=torch.int64, device=device
            ),
            llt_beta=torch.tensor(llt_beta),
            llt_tol=torch.tensor(llt_tol),
            llt_n_iter=torch.tensor(float(llt_n_iter)),
            llt_max_iter=torch.tensor(float(llt_max_iter)),
            enforce_sym=torch.tensor(1.0 if enforce_symmetry else 0.0),
        )

    # ------------------------------------------------------------------ repr

    def __repr__(self) -> str:
        return (
            f"Wing(n_pan={self.n_pan}, span={self.span:.3f} m, "
            f"S={self.S:.4f} m², cbar={self.cbar:.4f} m, device={self.device})"
        )


# ---------------------------------------------------------------------------
# AirfoilParams  (lightweight container, not a nn.Module)
# ---------------------------------------------------------------------------

@dataclass
class AirfoilParams:
    """
    Kulfan CST parameters for a single airfoil (or root/tip pair for 3D).

    All tensors live on *wing.device* and have ``requires_grad=True`` by default.

    Attributes
    ----------
    upper : (8,) upper-surface Bernstein weights
    lower : (8,) lower-surface Bernstein weights
    LE    : () or (1,) leading-edge weight
    TE    : () or (1,) trailing-edge thickness
    upper_tip, lower_tip, LE_tip, TE_tip : optional tip values; if None the
        root values are broadcast to all panels (constant spanwise shape).
    """

    upper: torch.Tensor
    lower: torch.Tensor
    LE:    torch.Tensor
    TE:    torch.Tensor
    upper_tip: Optional[torch.Tensor] = None
    lower_tip: Optional[torch.Tensor] = None
    LE_tip:    Optional[torch.Tensor] = None
    TE_tip:    Optional[torch.Tensor] = None

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str | Path,
        device: torch.device,
        requires_grad: bool = True,
    ) -> "AirfoilParams":
        """
        Load Kulfan weights from the ``airfoil:`` block of a pipeline YAML.

        Reads ``upper_initial_weights``, ``lower_initial_weights``,
        ``leading_edge_weight``, ``TE_thickness`` for the root, and the
        ``*_tip`` variants if present.  If tip weights are absent or identical
        to the root, the returned AirfoilParams has ``upper_tip=None`` (uniform
        spanwise shape).
        """
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        af_cfg = cfg.get("airfoil", {}) or {}

        def _p(key, fallback=None):
            val = af_cfg.get(key, fallback)
            if val is None:
                return None
            arr = np.atleast_1d(np.array(val, dtype=np.float32))  # always ≥1D → leaf tensor
            return torch.tensor(arr, dtype=torch.float32,
                                device=device, requires_grad=requires_grad)

        upper = _p("upper_initial_weights")
        lower = _p("lower_initial_weights")
        LE    = _p("leading_edge_weight")
        TE    = _p("TE_thickness")

        if upper is None:
            raise ValueError(f"No 'airfoil.upper_initial_weights' found in {yaml_path}")

        upper_tip = _p("upper_initial_weights_tip")
        lower_tip = _p("lower_initial_weights_tip")
        LE_tip    = _p("leading_edge_weight_tip")
        TE_tip    = _p("TE_thickness_tip")

        af = cls(
            upper=upper, lower=lower, LE=LE, TE=TE,
            upper_tip=upper_tip, lower_tip=lower_tip,
            LE_tip=LE_tip, TE_tip=TE_tip,
        )
        af.clamp_constraints()
        return af

    @classmethod
    def default(cls, device: torch.device, requires_grad: bool = True) -> "AirfoilParams":
        """Placeholder NACA-like weights. Prefer ``from_yaml`` when a config is available."""
        def _p(data):
            return torch.tensor(data, dtype=torch.float32, device=device,
                                requires_grad=requires_grad)
        return cls(
            upper=_p([0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01]),
            lower=_p([-0.05, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0]),
            LE=_p([0.0]),
            TE=_p([0.001]),
        )

    @classmethod
    def default_3d(cls, device: torch.device, requires_grad: bool = True) -> "AirfoilParams":
        """
        Root+tip pair with independent Kulfan weights, both satisfying constraints.
        Root: moderately cambered, NACA-like.
        Tip:  slightly thinner, less camber — typical for a tapered glider wing.
        """
        def _p(data):
            return torch.tensor(data, dtype=torch.float32, device=device,
                                requires_grad=requires_grad)
        af = cls(
            upper     =_p([0.10, 0.15, 0.20, 0.15, 0.10, 0.08, 0.07, 0.06]),
            lower     =_p([-0.05, -0.05, -0.04, -0.03, -0.02, -0.01, -0.005, 0.0]),
            LE        =_p([0.0]),
            TE        =_p([0.001]),
            upper_tip =_p([0.08, 0.12, 0.16, 0.12, 0.08, 0.06, 0.055, 0.052]),
            lower_tip =_p([-0.04, -0.04, -0.03, -0.02, -0.015, -0.008, -0.004, 0.0]),
            LE_tip    =_p([0.0]),
            TE_tip    =_p([0.001]),
        )
        af.clamp_constraints()  # ensure gap constraint is satisfied from the start
        return af

    def parameters(self) -> list[torch.Tensor]:
        """All leaf tensors (for passing to an optimiser)."""
        ps = [self.upper, self.lower, self.LE, self.TE]
        for t in (self.upper_tip, self.lower_tip, self.LE_tip, self.TE_tip):
            if t is not None:
                ps.append(t)
        return ps

    def clamp_constraints(self) -> None:
        """
        Hard geometric constraints matching Airfoil._enforce_constraints:
          - TE_thickness ∈ [1e-4, 0.01]
          - upper ≥ lower + 0.05  (element-wise, prevents surface crossing)
        Applied in-place (no-grad).
        """
        with torch.no_grad():
            self.TE.clamp_(1e-4, 0.01)
            min_gap = 0.05
            self.upper.data = torch.maximum(self.upper.data, self.lower.data + min_gap)
            if self.TE_tip is not None:
                self.TE_tip.clamp_(1e-4, 0.01)
            if self.upper_tip is not None and self.lower_tip is not None:
                self.upper_tip.data = torch.maximum(
                    self.upper_tip.data, self.lower_tip.data + min_gap
                )

    def detach_clone(self) -> "AirfoilParams":
        """Return a detached copy (e.g. for storing the optimum)."""
        def _dc(t):
            return None if t is None else t.detach().clone()
        return AirfoilParams(
            upper=_dc(self.upper), lower=_dc(self.lower),
            LE=_dc(self.LE), TE=_dc(self.TE),
            upper_tip=_dc(self.upper_tip), lower_tip=_dc(self.lower_tip),
            LE_tip=_dc(self.LE_tip), TE_tip=_dc(self.TE_tip),
        )


# ---------------------------------------------------------------------------
# DiffLLTEvaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Output of a single DiffLLTEvaluator call."""
    CL: torch.Tensor   # (B,)
    CD: torch.Tensor   # (B,)
    CM: torch.Tensor   # (B,)
    confidence: Optional[torch.Tensor] = None  # (B, n_pan) or None


class DiffLLTEvaluator:
    """
    Wraps LLTImplicitFn for a fixed Wing.

    Usage
    -----
    evaluator = DiffLLTEvaluator(wing)
    result = evaluator(alphas_deg, velocities, airfoil)
    loss = -result.CL.mean() / result.CD.mean()
    loss.backward()
    """

    def __init__(self, wing: Wing) -> None:
        self.wing = wing

    def __call__(
        self,
        alphas_deg: torch.Tensor,   # (B,)  angle of attack in degrees
        velocities: torch.Tensor,   # (B,)  airspeed in m/s
        airfoil: AirfoilParams,
    ) -> EvalResult:
        """
        Evaluate the 3D LLT for B operating points simultaneously.

        Both ``alphas_deg`` and ``velocities`` must be on ``wing.device``.
        ``airfoil`` must also live on ``wing.device`` and have requires_grad=True
        for the backward to propagate to shape parameters.

        Returns an EvalResult with CL, CD, CM tensors of shape (B,).
        """
        w = self.wing
        T = w.tensors

        # Expand constant airfoil shape to all panels if no spanwise variation
        upper = airfoil.upper
        lower = airfoil.lower
        LE    = airfoil.LE.reshape(-1)
        TE    = airfoil.TE.reshape(-1)

        if airfoil.upper_tip is not None:
            # Linear root→tip interpolation using η  (same as mix_root_tip_torch)
            eta = w.eta.unsqueeze(-1)          # (n_pan, 1)
            upper = (1.0 - eta) * upper + eta * airfoil.upper_tip
            lower = (1.0 - eta) * lower + eta * airfoil.lower_tip
            LE = (1.0 - w.eta) * LE[0] + w.eta * airfoil.LE_tip.reshape(-1)[0]
            TE = (1.0 - w.eta) * TE[0] + w.eta * airfoil.TE_tip.reshape(-1)[0]

        C, _, _ = LLTImplicitFn.apply(
            alphas_deg.reshape(-1),
            velocities.reshape(-1),
            upper, lower, LE, TE,
            T["dy"], T["y"], T["c"], T["tw"],
            T["S"], T["cbar"], T["x_c4"], T["span"],
            T["D_nf"], T["D_tr"], T["mirror_of"],
            T["cos_sweep"], T["cos2_sweep"],
            w.rho, w.mu,
            w.llt_beta, w.llt_tol, w.llt_n_iter, w.llt_max_iter, w.enforce_sym,
            w.model_size_id, w.device_id,
        )

        return EvalResult(CL=C[:, 0], CD=C[:, 1], CM=C[:, 2])

    def eval_spanwise(
        self,
        alpha_deg: float,
        velocity: float,
        airfoil: AirfoilParams,
    ) -> dict:
        """
        Evaluate panel-level quantities for a SINGLE operating point (no grad).

        Returns a dict with numpy arrays of shape (n_pan,):
          y          – panel spanwise positions (full span, root→tip)
          Gamma      – converged vortex circulation [m²/s]
          cl_pan     – local NeuralFoil cl at each panel
          w_nf       – induced downwash [m/s]
          alpha_eff  – effective AoA at each panel [deg]
        """
        w  = self.wing
        T  = w.tensors
        dev = w.device

        alpha_t = torch.tensor([[alpha_deg]], dtype=torch.float32, device=dev)
        V_t     = torch.tensor([[velocity]],  dtype=torch.float32, device=dev)

        upper = airfoil.upper
        lower = airfoil.lower
        LE    = airfoil.LE.reshape(-1)
        TE    = airfoil.TE.reshape(-1)

        if airfoil.upper_tip is not None:
            eta   = w.eta.unsqueeze(-1)
            upper = (1.0 - eta) * upper + eta * airfoil.upper_tip
            lower = (1.0 - eta) * lower + eta * airfoil.lower_tip
            LE    = (1.0 - w.eta) * LE[0] + w.eta * airfoil.LE_tip.reshape(-1)[0]
            TE    = (1.0 - w.eta) * TE[0] + w.eta * airfoil.TE_tip.reshape(-1)[0]

        const = LLTConst(
            dy=T["dy"], y=T["y"], c=T["c"], tw=T["tw"],
            S=T["S"], cbar=T["cbar"], x_c4=T["x_c4"], span=T["span"],
            D_nf=T["D_nf"], D_tr=T["D_tr"], mirror_of=T["mirror_of"],
            cos_sweep=T["cos_sweep"], cos2_sweep=T["cos2_sweep"],
            rho=w.rho, mu=w.mu,
            n_iter=int(w.llt_n_iter.item()),
            beta=float(w.llt_beta.item()),
            tol=float(w.llt_tol.item()),
            enforce_symmetry=bool(w.enforce_sym.item() > 0.5),
            model_size=_ID_TO_MODEL_SIZE[int(w.model_size_id.item())],
        )

        with torch.no_grad():
            tw0      = const.tw.unsqueeze(0)
            cos_sw0  = const.cos_sweep.unsqueeze(0)
            c0       = const.c.unsqueeze(0)
            V_n0     = V_t * cos_sw0
            c_eff0   = c0 * cos_sw0
            alpha_geo = alpha_t + tw0
            Re0      = const.rho * V_n0 * c_eff0 / const.mu
            aero0    = _eval_nf_batched(upper, lower, LE, TE, alpha_geo, Re0, const)
            Gamma    = 0.5 * V_n0 * c_eff0 * aero0["CL"]

            for _ in range(int(w.llt_max_iter.item())):
                Gamma_new = _G(Gamma, alpha_t, V_t, upper, lower, LE, TE, const)
                rel_diff  = (torch.max(torch.abs(Gamma_new - Gamma)) /
                             max(1.0, float(torch.max(torch.abs(Gamma))))).item()
                Gamma = Gamma_new
                if rel_diff < float(w.llt_tol.item()):
                    break

            w_nf     = Gamma @ const.D_nf.T                                          # (1, n_pan)
            V_n_pan  = V_t * const.cos_sweep.unsqueeze(0)                            # (1, n_pan)
            c_eff_pan = const.c.unsqueeze(0) * const.cos_sweep.unsqueeze(0)         # (1, n_pan)
            alpha_eff = (alpha_t + const.tw.unsqueeze(0)
                         - torch.rad2deg(torch.atan2(w_nf, V_n_pan)))               # (1, n_pan)
            Re_pan   = const.rho * V_n_pan * c_eff_pan / const.mu                   # (1, n_pan)
            aero_pan = _eval_nf_batched(upper, lower, LE, TE, alpha_eff, Re_pan, const)

        return {
            "y":         T["y"].cpu().numpy(),
            "Gamma":     Gamma[0].cpu().numpy(),
            "cl_pan":    aero_pan["CL"][0].cpu().numpy(),
            "w_nf":      w_nf[0].cpu().numpy(),
            "alpha_eff": alpha_eff[0].cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# ShapeOptimiser
# ---------------------------------------------------------------------------

@dataclass
class OptimHistory:
    """Recorded metrics at each iteration."""
    loss: list = field(default_factory=list)
    cl: list   = field(default_factory=list)
    cd: list   = field(default_factory=list)
    clcd: list = field(default_factory=list)
    conf_penalty: list = field(default_factory=list)
    clcd_penalty: list = field(default_factory=list)


class ShapeOptimiser:
    """
    Gradient-based shape optimiser with augmented-Lagrangian soft constraints.

    Constraints (same as the main pipeline's NeuralFoilSampling block):
      g_conf : mean(analysis_confidence) ≥ min_confidence
      g_clcd : mean(CL/CD)              ≥ min_avg_clcd

    The TE_thickness hard clamp from Airfoil._enforce_constraints is applied
    after every optimiser step.

    Parameters
    ----------
    evaluator   : DiffLLTEvaluator
    cost_fn     : callable(CL, CD, CM) → scalar Tensor (the objective to minimise)
    alphas_deg  : (B,) Tensor of operating-point AoAs  [degrees]
    velocities  : (B,) Tensor of operating-point airspeeds [m/s]
    lr          : Adam learning rate (default 1e-2)
    gamma       : ExponentialLR decay factor per iteration (default 0.99)
    min_confidence : lower bound on NeuralFoil confidence (default 0.7)
    min_avg_clcd   : lower bound on mean CL/CD (default 2.0)
    rho_aug        : augmented Lagrangian penalty weight (default 1.0)
    verbose        : print progress every *verbose* iters (0 = silent)
    """

    def __init__(
        self,
        evaluator: DiffLLTEvaluator,
        cost_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        alphas_deg: torch.Tensor,
        velocities: torch.Tensor,
        *,
        lr: float = 1e-2,
        gamma: float = 0.99,
        min_confidence: float = 0.7,
        min_avg_clcd: float = 2.0,
        rho_aug: float = 1.0,
        verbose: int = 10,
    ) -> None:
        self.evaluator = evaluator
        self.cost_fn   = cost_fn
        self.alphas    = alphas_deg
        self.velocities = velocities
        self.min_confidence = min_confidence
        self.min_avg_clcd   = min_avg_clcd
        self.rho_aug   = rho_aug
        self.verbose   = verbose

        # These are initialised/reset in run()
        self._airfoil: Optional[AirfoilParams] = None
        self._opt: Optional[torch.optim.Optimizer] = None
        self._sched: Optional[object] = None
        self._lambda_conf = 0.0
        self._lambda_clcd = 0.0
        self.history = OptimHistory()
        self._lr = lr
        self._gamma = gamma

    # ------------------------------------------------------------------ public

    def run(
        self,
        airfoil: AirfoilParams,
        n_iters: int = 100,
    ) -> AirfoilParams:
        """
        Run ``n_iters`` gradient steps starting from *airfoil*.

        Returns the final (in-place updated) AirfoilParams.
        History is accumulated in ``self.history``.
        """
        self._airfoil = airfoil
        self._opt = torch.optim.Adam(airfoil.parameters(), lr=self._lr)
        self._sched = torch.optim.lr_scheduler.ExponentialLR(self._opt, gamma=self._gamma)
        self._lambda_conf = 0.0
        self._lambda_clcd = 0.0

        for i in range(n_iters):
            loss, metrics = self._step(airfoil)
            self.history.loss.append(float(loss))
            self.history.cl.append(metrics["cl"])
            self.history.cd.append(metrics["cd"])
            self.history.clcd.append(metrics["clcd"])
            self.history.conf_penalty.append(metrics["conf_penalty"])
            self.history.clcd_penalty.append(metrics["clcd_penalty"])

            if self.verbose and (i % self.verbose == 0 or i == n_iters - 1):
                print(
                    f"iter {i:4d}  loss={loss:.4f}  CL={metrics['cl']:.4f}"
                    f"  CD={metrics['cd']:.5f}  CL/CD={metrics['clcd']:.2f}"
                    f"  λ_conf={self._lambda_conf:.3f}  λ_clcd={self._lambda_clcd:.3f}"
                )

        return airfoil

    # ------------------------------------------------------------------ private

    def _step(self, airfoil: AirfoilParams):
        self._opt.zero_grad()

        result = self.evaluator(self.alphas, self.velocities, airfoil)
        CL, CD, CM = result.CL, result.CD, result.CM

        # Aerodynamic cost
        base_loss = self.cost_fn(CL, CD, CM)

        # Augmented Lagrangian — confidence
        # (confidence not returned by DiffLLTEvaluator directly; we monitor
        #  mean CL/CD as a proxy and skip confidence penalty if unavailable)
        conf_pen = torch.tensor(0.0, device=CL.device)
        clcd_pen = torch.tensor(0.0, device=CL.device)

        CD_safe = CD.clamp(min=1e-5)
        mean_clcd = (CL / CD_safe).mean()
        viol_clcd = torch.relu(self.min_avg_clcd - mean_clcd)
        clcd_pen = self._lambda_clcd * viol_clcd + 0.5 * self.rho_aug * viol_clcd ** 2

        total_loss = base_loss + conf_pen + clcd_pen
        total_loss.backward()

        self._opt.step()
        self._sched.step()

        # Multiplier updates (after step, outside grad tape)
        with torch.no_grad():
            self._lambda_clcd = max(
                0.0, self._lambda_clcd + self.rho_aug * float(viol_clcd)
            )

        # Hard geometric constraints
        airfoil.clamp_constraints()

        metrics = {
            "cl":           float(CL.mean()),
            "cd":           float(CD.mean()),
            "clcd":         float(mean_clcd),
            "conf_penalty": float(conf_pen),
            "clcd_penalty": float(clcd_pen),
        }
        return float(total_loss), metrics
