"""Compute the largest inscribed box of the 4-DOF wrench polytope.

The achievable wrench polytope (with Tx=Ty=0 enforced) is:

    P = { w ∈ ℝ⁶ : ∃ T ∈ [-25, +25]^8 s.t. M·T = w, w[3] = w[4] = 0 }

For an APRBS sampling box B = [-cx, +cx] × [-cy, +cy] × [-cz, +cz] × [-ctz, +ctz]
in 4-DOF coordinates (Fx, Fy, Fz, Tz), every corner of B must satisfy

    ‖T_pinv · corner‖_∞ ≤ 25

(otherwise the pseudo-inverse allocator + T200 saturation will distort
the realized wrench).

This becomes a set of 8 linear inequalities in (cx, cy, cz, ctz):

    cx |T_pinv[j,Fx]| + cy |T_pinv[j,Fy]| + cz |T_pinv[j,Fz]| + ctz |T_pinv[j,Tz]| ≤ 25
    for each rotor j = 1..8

(The worst case across all 16 corner sign combinations is when all signs
align — the L1 norm of the j-th row weighted by caps.)

This script solves two problems:

  (a) **Max-volume inscribed box**: maximize cx·cy·cz·ctz subject to the
      8 constraints. Solved as a geometric program (log-transform makes
      it convex), using scipy.optimize.minimize with SLSQP.

  (b) **Proportional inscribed box**: find the largest scalar s such that
      (s·65, s·37.5, s·102, s·17) — the current caps scaled by s — is
      inside the polytope.

For each, the script:
  - Reports the inscribed caps.
  - Verifies that every corner of the resulting box has rotor demand ≤ 25 N.
  - Compares against the current APRBS caps.
  - Generates a visualization showing the inscribed box overlaid on the
    saturation heatmap.

Usage::

    python -m EDMDc.find_inscribed_aprbs_box
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize

from .thrusters import ThrusterConfig, ThrustAllocator


_PROJECT = Path(__file__).resolve().parents[1]
_VEHICLE_YAML = _PROJECT / "assets" / "BlueROVHeavy" / "BlueROVHeavy.yaml"
_OUT_DIR = _PROJECT / "EDMDc" / "data" / "plots"

CURRENT_CAPS_POS = np.array([65.0, 37.5, 102.0, 0.0, 0.0, 17.0])
CURRENT_CAPS_NEG = np.array([65.0, 37.5,  80.0, 0.0, 0.0, 17.0])
TASK_AXES = (0, 1, 2, 5)
TASK_NAME = ("Fx", "Fy", "Fz", "Tz")
TASK_UNIT = ("N",  "N",  "N",  "N·m")
MAX_THRUST = 25.0


def _abs_pinv_columns(T_pinv: np.ndarray) -> np.ndarray:
    """Extract |T_pinv[:, task_axes]| → shape (8, 4) — the constraint matrix.

    Each row j is the absolute value of the pseudo-inverse coefficients
    for rotor j as a function of (Fx, Fy, Fz, Tz)."""
    return np.abs(T_pinv[:, list(TASK_AXES)])


def solve_max_volume(A: np.ndarray, cap: float = MAX_THRUST) -> np.ndarray:
    """Maximize cx·cy·cz·ctz subject to A @ [cx,cy,cz,ctz] ≤ cap, c ≥ 0.

    A is (8, 4). Uses log-domain reformulation (geometric program):
    minimize -sum(log(ci)) s.t. constraints + lower bound to avoid log(0).
    """
    n_axes = 4
    # Initialize at a small positive value
    x0 = np.full(n_axes, 1.0)

    def neg_log_vol(c):
        c = np.maximum(c, 1e-9)
        return -np.sum(np.log(c))

    def jac(c):
        c = np.maximum(c, 1e-9)
        return -1.0 / c

    # Inequality constraints: cap - A @ c ≥ 0 (per-rotor)
    constraints = [
        {"type": "ineq",
         "fun": lambda c, j=j: cap - A[j, :] @ c,
         "jac": lambda c, j=j: -A[j, :]}
        for j in range(A.shape[0])
    ]
    bounds = [(1e-3, None)] * n_axes

    result = minimize(neg_log_vol, x0, jac=jac, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-10, "maxiter": 500})
    if not result.success:
        raise RuntimeError(f"max-volume SLSQP failed: {result.message}")
    return result.x


def solve_proportional(A: np.ndarray,
                       current_caps: np.ndarray,
                       cap: float = MAX_THRUST) -> tuple[float, np.ndarray]:
    """Find largest s such that s·current_caps satisfies the per-rotor cap
    on every rotor. Returns (s, s·current_caps)."""
    rotor_demand_at_current = A @ current_caps      # (8,) — demand per rotor
    max_demand = rotor_demand_at_current.max()
    s = cap / max_demand
    return s, s * current_caps


def verify_corner_feasibility(T_pinv: np.ndarray,
                              caps_4dof: np.ndarray) -> tuple[float, np.ndarray]:
    """Check all 16 corners of the box. Returns max per-rotor demand across
    all corners and the corner that achieves it."""
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 4)
    corner_wrenches = np.zeros((16, 6))
    for i, s in enumerate(signs):
        corner_wrenches[i, list(TASK_AXES)] = s * caps_4dof
    rotor_thrust = corner_wrenches @ T_pinv.T   # (16, 8)
    max_per_corner = np.abs(rotor_thrust).max(axis=1)   # (16,)
    worst_idx = max_per_corner.argmax()
    return float(max_per_corner[worst_idx]), corner_wrenches[worst_idx]


def _heatmap_grid(T_pinv: np.ndarray, axis_a: int, axis_b: int,
                  cap_a: float, cap_b: float, n: int = 80):
    grid_a = np.linspace(-cap_a, cap_a, n)
    grid_b = np.linspace(-cap_b, cap_b, n)
    H = np.zeros((n, n))
    w = np.zeros(6)
    for i, a in enumerate(grid_a):
        for j, b in enumerate(grid_b):
            w[:] = 0.0
            w[axis_a] = a; w[axis_b] = b
            H[j, i] = float(np.max(np.abs(T_pinv @ w)))
    return grid_a, grid_b, H


def main() -> int:
    with open(_VEHICLE_YAML) as f:
        vcfg = yaml.safe_load(f)
    tc = ThrusterConfig.from_yaml_dict(vcfg)
    allocator = ThrustAllocator(tc, max_thrust_per_motor=MAX_THRUST)
    T_pinv = allocator.T_pinv
    A = _abs_pinv_columns(T_pinv)   # (8, 4)

    print("=== T_pinv |coefficients| for 4-DOF axes (rotor × axis) ===")
    print(f"{'rotor':>6s}  {'|Fx|':>7s} {'|Fy|':>7s} {'|Fz|':>7s} {'|Tz|':>7s}")
    for j in range(A.shape[0]):
        print(f"{j:>6d}  " + " ".join(f"{A[j, k]:>7.4f}" for k in range(4)))

    # --- Solve (a) max-volume ---
    caps_mv = solve_max_volume(A)
    max_demand_mv, worst_corner_mv = verify_corner_feasibility(T_pinv, caps_mv)
    vol_mv = float(np.prod(caps_mv))
    print(f"\n=== (a) Max-volume inscribed box ===")
    print(f"  caps = ({caps_mv[0]:.2f}, {caps_mv[1]:.2f}, {caps_mv[2]:.2f}, "
          f"{caps_mv[3]:.2f})  [Fx, Fy, Fz, Tz]")
    print(f"  worst-corner rotor demand: {max_demand_mv:.3f} N  "
          f"(cap = {MAX_THRUST} N) — {'FEASIBLE' if max_demand_mv <= MAX_THRUST + 1e-6 else 'INFEASIBLE'}")
    print(f"  volume = {vol_mv:.2f}")

    # --- Solve (b) proportional ---
    current_caps_4dof = CURRENT_CAPS_POS[list(TASK_AXES)]
    s, caps_prop = solve_proportional(A, current_caps_4dof)
    max_demand_prop, _ = verify_corner_feasibility(T_pinv, caps_prop)
    vol_prop = float(np.prod(caps_prop))
    print(f"\n=== (b) Proportional inscribed box ===")
    print(f"  scale s = {s:.4f}")
    print(f"  caps = ({caps_prop[0]:.2f}, {caps_prop[1]:.2f}, {caps_prop[2]:.2f}, "
          f"{caps_prop[3]:.2f})  [Fx, Fy, Fz, Tz]")
    print(f"  worst-corner rotor demand: {max_demand_prop:.3f} N  — "
          f"{'FEASIBLE' if max_demand_prop <= MAX_THRUST + 1e-6 else 'INFEASIBLE'}")
    print(f"  volume = {vol_prop:.2f}")

    # --- Current caps for comparison ---
    max_demand_current, worst_curr = verify_corner_feasibility(T_pinv, current_caps_4dof)
    print(f"\n=== Reference: current APRBS caps ===")
    print(f"  caps = ({current_caps_4dof[0]:.2f}, {current_caps_4dof[1]:.2f}, "
          f"{current_caps_4dof[2]:.2f}, {current_caps_4dof[3]:.2f})")
    print(f"  worst-corner rotor demand: {max_demand_current:.3f} N — INFEASIBLE")
    print(f"  saturation factor: {max_demand_current / MAX_THRUST:.2f}× cap "
          f"at the corner")
    print(f"  volume = {float(np.prod(current_caps_4dof)):.2f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # (Fx, Fy) heatmap
    ax = axes[0]
    gx, gy, H = _heatmap_grid(T_pinv, 0, 1,
                              CURRENT_CAPS_POS[0], CURRENT_CAPS_POS[1])
    extent = (-CURRENT_CAPS_POS[0], CURRENT_CAPS_POS[0],
              -CURRENT_CAPS_POS[1], CURRENT_CAPS_POS[1])
    im = ax.imshow(H, extent=extent, origin="lower", aspect="auto",
                   cmap="RdYlGn_r", vmin=0, vmax=max(MAX_THRUST * 3, H.max()))
    cs = ax.contour(gx, gy, H, levels=[MAX_THRUST], colors="black", linewidths=2.0)
    ax.clabel(cs, fmt={MAX_THRUST: f"{MAX_THRUST:.0f} N cap"}, fontsize=9)
    # Current box (blue), max-volume (orange), proportional (purple)
    def box(c1, c2, **kwargs):
        ax.plot([-c1, c1, c1, -c1, -c1], [-c2, -c2, c2, c2, -c2], **kwargs)
    box(CURRENT_CAPS_POS[0], CURRENT_CAPS_POS[1], color="blue", lw=2.0,
        label=f"current ({CURRENT_CAPS_POS[0]:.0f}, {CURRENT_CAPS_POS[1]:.0f})")
    box(caps_mv[0], caps_mv[1], color="darkorange", lw=2.5,
        label=f"max-volume ({caps_mv[0]:.1f}, {caps_mv[1]:.1f})")
    box(caps_prop[0], caps_prop[1], color="purple", lw=2.5, ls="--",
        label=f"proportional ({caps_prop[0]:.1f}, {caps_prop[1]:.1f})")
    plt.colorbar(im, ax=ax, label="max |rotor demand| [N]")
    ax.set_xlabel("Fx commanded [N]"); ax.set_ylabel("Fy commanded [N]")
    ax.set_title("(Fx, Fy) plane (Fz=0, Tz=0)\nInscribed box options")
    ax.legend(loc="upper left", fontsize=9)

    # (Fz, Tz) heatmap
    ax = axes[1]
    gz, gt, H = _heatmap_grid(T_pinv, 2, 5,
                              CURRENT_CAPS_POS[2], CURRENT_CAPS_POS[5])
    extent = (-CURRENT_CAPS_POS[2], CURRENT_CAPS_POS[2],
              -CURRENT_CAPS_POS[5], CURRENT_CAPS_POS[5])
    im = ax.imshow(H, extent=extent, origin="lower", aspect="auto",
                   cmap="RdYlGn_r", vmin=0, vmax=max(MAX_THRUST * 3, H.max()))
    cs = ax.contour(gz, gt, H, levels=[MAX_THRUST], colors="black", linewidths=2.0)
    ax.clabel(cs, fmt={MAX_THRUST: f"{MAX_THRUST:.0f} N cap"}, fontsize=9)
    def box2(c1, c2, **kwargs):
        ax.plot([-c1, c1, c1, -c1, -c1], [-c2, -c2, c2, c2, -c2], **kwargs)
    box2(CURRENT_CAPS_POS[2], CURRENT_CAPS_POS[5], color="blue", lw=2.0,
         label=f"current ({CURRENT_CAPS_POS[2]:.0f}, {CURRENT_CAPS_POS[5]:.0f})")
    box2(caps_mv[2], caps_mv[3], color="darkorange", lw=2.5,
         label=f"max-volume ({caps_mv[2]:.1f}, {caps_mv[3]:.1f})")
    box2(caps_prop[2], caps_prop[3], color="purple", lw=2.5, ls="--",
         label=f"proportional ({caps_prop[2]:.1f}, {caps_prop[3]:.1f})")
    plt.colorbar(im, ax=ax, label="max |rotor demand| [N]")
    ax.set_xlabel("Fz commanded [N]"); ax.set_ylabel("Tz commanded [N·m]")
    ax.set_title("(Fz, Tz) plane (Fx=0, Fy=0)\nInscribed box options")
    ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        f"Inscribed APRBS boxes — current vs max-volume vs proportional\n"
        f"black contour = {MAX_THRUST} N per-rotor cap "
        f"(any box inside the green region is fully feasible)",
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _OUT_DIR / f"inscribed_box_{ts}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # ---- Final recommendation ----
    print(f"\n=== Recommended caps for APRBS (max-volume inscribed) ===")
    print(f"  WRENCH_CMD_MAX_POS = np.array([{caps_mv[0]:.2f}, {caps_mv[1]:.2f}, "
          f"{caps_mv[2]:.2f}, 0.0, 0.0, {caps_mv[3]:.2f}], dtype=np.float64)")
    print(f"  WRENCH_CMD_MAX_NEG = np.array([{caps_mv[0]:.2f}, {caps_mv[1]:.2f}, "
          f"{caps_mv[2] * (CURRENT_CAPS_NEG[2] / CURRENT_CAPS_POS[2]):.2f}, "
          f"0.0, 0.0, {caps_mv[3]:.2f}], dtype=np.float64)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
