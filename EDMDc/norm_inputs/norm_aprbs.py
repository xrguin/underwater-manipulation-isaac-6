"""Inscribed-box, per-axis normalized [-1,1] control-input conventions.

Design (locked for this folder)
-------------------------------
The EDMDc/ARX control input ``U`` is the 6-DOF body wrench **normalized per axis
to [-1, 1]**, restricted to the 4-DOF task mask (Fx, Fy, Fz, Tz; Tx, Ty = 0).
Excitation samples the **inscribed box** of the achievable wrench polytope — the
same realizable region the shipping pipeline uses — so every sample is achievable
(no allocator saturation). This is the original "normalized [-1,1]" request; the
earlier thruster-cube/zonotope variant was dropped to avoid name confusion with
the payload-cube pickup task.

Single symmetric scale per axis
-------------------------------
EDMDc/ARX are linear in the input, so for ``u_norm -> force`` to be linear (and
``B`` to be just the Newton-B rescaled), each axis must use one scale. The
inscribed caps are asymmetric on heave (Fz +88.17 / -69.15 N); we take the
symmetric scale ``s_i = min(cap_pos_i, cap_neg_i)`` per axis:

    SYMMETRIC_CAPS = [22.57, 23.47, 69.15, 0, 0, 6.59]   (N, N, N, -, -, N*m)
    u_newton = SYMMETRIC_CAPS * u_norm                    (u_norm in [-1, 1])
    B_newton = B_norm / s_active                          (column-wise)

The symmetric box is a subset of the inscribed box (itself inside the achievable
polytope), so with the linear unit-thrust allocator the recorded command equals
the realized wrench — a clean LS fit. Cost: a little heave-up coverage (Fz capped
at 69.15 instead of 88.17 N).
"""
from __future__ import annotations

import numpy as np

from EDMDc.collect_fossen import (
    INSCRIBED_BOX_CAPS_POS,
    INSCRIBED_BOX_CAPS_NEG,
    aprbs_sequence_continuous,
)

# Per-axis symmetric scale s_i = min(cap_pos_i, cap_neg_i). Order: Fx, Fy, Fz,
# Tx, Ty, Tz. Tx, Ty are 0 (masked off, 4-DOF task).
SYMMETRIC_CAPS = np.minimum(INSCRIBED_BOX_CAPS_POS, INSCRIBED_BOX_CAPS_NEG)
# -> [22.57, 23.47, 69.15, 0.0, 0.0, 6.59]

# Normalized command box: every active axis is excited uniformly in [-1, 1].
NORM_CMD_MAX_POS = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
NORM_CMD_MAX_NEG = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)


def denorm_to_newton(u_norm: np.ndarray) -> np.ndarray:
    """Map a normalized command in [-1, 1] to a body wrench in Newtons.

    ``u_norm`` may be (6,) or (N, 6). Elementwise ``u_newton = SYMMETRIC_CAPS * u_norm``.
    """
    return np.asarray(u_norm, dtype=np.float64) * SYMMETRIC_CAPS


def norm_from_newton(u_newton: np.ndarray) -> np.ndarray:
    """Inverse of :func:`denorm_to_newton`. Inactive (zero-scale) axes -> 0."""
    u = np.asarray(u_newton, dtype=np.float64)
    out = np.zeros_like(u)
    nz = SYMMETRIC_CAPS != 0.0
    out[..., nz] = u[..., nz] / SYMMETRIC_CAPS[nz]
    return out


def aprbs_sequence_normalized(
    rng: np.random.Generator,
    n_steps: int,
    axis_mask: np.ndarray,
    hold_max_steps=None,
) -> np.ndarray:
    """APRBS in the normalized [-1, 1] command box (inscribed-box excitation).

    Thin wrapper over ``collect_fossen.aprbs_sequence_continuous`` with caps set
    to the unit box, so each active axis holds a value drawn from Uniform(-1, 1).
    Returns (n_steps, 6) normalized; de-normalize with :func:`denorm_to_newton`
    before feeding the sim.
    """
    return aprbs_sequence_continuous(
        rng, n_steps,
        caps_pos=NORM_CMD_MAX_POS,
        caps_neg=NORM_CMD_MAX_NEG,
        axis_mask=axis_mask,
        hold_max_steps=hold_max_steps,
    )
