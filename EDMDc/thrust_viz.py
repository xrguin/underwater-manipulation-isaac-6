"""Per-rotor 3D thrust-vector arrows in the Isaac Sim viewport.

Draws one coloured line segment per thruster, anchored at the rotor's
body-frame position, pointing along its thrust axis, length proportional to
realized thrust magnitude. Optionally adds a 2-line "V" arrowhead at the tip
since the Isaac debug-draw interface has no native arrowhead primitive.

Usage from bluerov_demo.py:

    from thrust_viz import ThrustVizDrawer
    thrust_viz = ThrustVizDrawer(thruster_cfg)
    # ... main loop:
    thrust_viz.update(pos_w[0], quat_w[0], thrusts_per_rotor)
    # ... shutdown:
    thrust_viz.close()

Color convention:
    positive thrust along axis  -> red    (1.0, 0.2, 0.2, 1.0)
    negative thrust              -> blue   (0.2, 0.4, 1.0, 1.0)
    sub-deadband (|t| < 0.5 N)   -> faded  (0.6, 0.6, 0.6, 0.3)

The debug-draw interface is acquired lazily on the first `update()` call so
that `import thrust_viz` works on a machine without Isaac Sim (matches the
import-safety of `thrusters.py`). If the import fails, the drawer prints one
warning and silently no-ops thereafter.
"""

from __future__ import annotations

import numpy as np


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Isaac Sim quaternion (w, x, y, z) -> 3x3 rotation matrix R such that
    v_world = R @ v_body. Same formula as bluerov_demo.quat_to_rot, copied here
    to keep this module self-contained."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


class ThrustVizDrawer:
    """Per-step debug-draw of per-rotor thrust vectors in the viewport."""

    def __init__(
        self,
        thruster_cfg,
        max_thrust_n: float = 50.0,
        arrow_length_m: float = 0.30,
        line_width_px: int = 4,
        draw_arrowheads: bool = True,
        deadband_n: float = 0.5,
    ):
        """
        Args:
            thruster_cfg: ThrusterConfig (from thrusters.py). We cache its
                `positions` (N, 3, body-frame metres) and `thrust_axes`
                (N, 3, body-frame unit vectors) directly.
            max_thrust_n: Per-rotor thrust at which the arrow reaches its max
                length. Default 50 N matches the per-motor saturation used by
                `ThrustAllocator(..., max_thrust_per_motor=50.0)` in the demo.
            arrow_length_m: Max world-space arrow length at saturation, in
                metres. 0.30 m is roughly the chassis half-width — readable but
                not so big it overlaps with the body.
            line_width_px: Line thickness in screen pixels.
            draw_arrowheads: If True, draw a 2-line "V" at each arrow tip
                (4 extra lines total beyond the shaft per rotor). Cheap visual
                cue for direction. Set False if the viewport feels cluttered.
            deadband_n: Below this thrust magnitude the arrow is rendered as a
                short faded grey line, not red/blue. Avoids visual flicker when
                T200 deadband (sub-1.4 N requests) flips on/off.
        """
        positions = np.asarray(thruster_cfg.positions, dtype=float).reshape(-1, 3)
        axes = np.asarray(thruster_cfg.thrust_axes, dtype=float).reshape(-1, 3)
        if positions.shape[0] != axes.shape[0]:
            raise ValueError(
                f"thruster_cfg.positions has {positions.shape[0]} rows but "
                f"thrust_axes has {axes.shape[0]} — mismatch."
            )
        self.positions = positions
        self.axes = axes
        self.n = positions.shape[0]
        self.max_thrust = float(max_thrust_n)
        self.length = float(arrow_length_m)
        self.lw = int(line_width_px)
        self.draw_heads = bool(draw_arrowheads)
        self.deadband = float(deadband_n)
        self._iface = None
        self._init_failed = False

    # --- debug-draw lazy init ----------------------------------------------

    def _ensure_iface(self) -> None:
        if self._iface is not None or self._init_failed:
            return
        try:
            from isaacsim.util.debug_draw import _debug_draw
            self._iface = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:
            print(f"[thrust_viz] isaacsim.util.debug_draw unavailable ({e}); "
                  f"thrust arrows disabled.")
            self._init_failed = True

    # --- per-step entry point ----------------------------------------------

    def update(
        self,
        pos_world: np.ndarray,
        quat_wxyz_world: np.ndarray,
        thrusts_per_rotor: np.ndarray,
    ) -> None:
        """Redraw all thrust arrows for the current step.

        Args:
            pos_world: (3,) world-frame position of base_link.
            quat_wxyz_world: (4,) world-frame orientation, Isaac Sim wxyz.
            thrusts_per_rotor: (N,) per-rotor thrust forces in Newtons (signed
                — positive means thrust along the rotor's thrust_axis,
                negative means against it).
        """
        if thrusts_per_rotor is None:
            return
        t = np.asarray(thrusts_per_rotor, dtype=float).reshape(-1)
        if t.shape[0] != self.n:
            return
        self._ensure_iface()
        if self._init_failed or self._iface is None:
            return

        R = _quat_wxyz_to_rot(np.asarray(quat_wxyz_world, dtype=float).reshape(4))
        p0 = np.asarray(pos_world, dtype=float).reshape(3)

        # Each rotor's world-frame anchor (start of arrow shaft).
        starts = p0 + (R @ self.positions.T).T               # (N, 3)
        # World-frame thrust axes (unit-length).
        axes_w = (R @ self.axes.T).T                          # (N, 3)
        # Signed scaled vector from start to tip. Sign of `t` flips direction
        # automatically — do NOT also multiply by sign(t).
        scale = (t / self.max_thrust) * self.length          # (N,)
        ends = starts + axes_w * scale[:, None]              # (N, 3)

        starts_list = [tuple(s) for s in starts]
        ends_list = [tuple(e) for e in ends]
        widths = [self.lw] * self.n
        colors = [self._color_for(ti) for ti in t]

        if self.draw_heads:
            h_starts, h_ends, h_colors, h_widths = self._build_arrowheads(
                starts, ends, axes_w, scale, t)
            starts_list += h_starts
            ends_list += h_ends
            colors += h_colors
            widths += h_widths

        try:
            self._iface.clear_lines()
            self._iface.draw_lines(starts_list, ends_list, colors, widths)
        except Exception:
            # Renderer hiccup mid-frame; skip this draw, try again next step.
            pass

    # --- internals ---------------------------------------------------------

    def _color_for(self, thrust_n: float):
        if abs(thrust_n) < self.deadband:
            return (0.6, 0.6, 0.6, 0.3)
        return (1.0, 0.2, 0.2, 1.0) if thrust_n > 0 else (0.2, 0.4, 1.0, 1.0)

    def _build_arrowheads(self, starts, ends, axes_w, scale, t):
        """Build a 2-line "V" at each arrow tip.

        For each rotor with non-trivial thrust, emit two short segments going
        from the tip back toward the shaft, splayed +/- 25 deg about a
        perpendicular axis. Length = 25% of the arrow's signed length.
        Sub-deadband rotors get no arrowhead.
        """
        h_starts, h_ends, h_colors, h_widths = [], [], [], []
        spread = 0.25      # head length as fraction of shaft length
        cos_a = np.cos(np.deg2rad(25.0))
        sin_a = np.sin(np.deg2rad(25.0))
        world_up = np.array([0.0, 0.0, 1.0])

        for i in range(self.n):
            if abs(t[i]) < self.deadband:
                continue
            tip = ends[i]
            # Direction from tip back along the shaft.
            shaft_dir = axes_w[i] * np.sign(scale[i])    # unit vec, points tip-ward
            back = -shaft_dir
            # Perpendicular axis for the V splay. Use world up unless shaft is
            # aligned with it, then pick world +X.
            if abs(np.dot(shaft_dir, world_up)) > 0.95:
                perp = np.cross(shaft_dir, np.array([1.0, 0.0, 0.0]))
            else:
                perp = np.cross(shaft_dir, world_up)
            perp_norm = np.linalg.norm(perp)
            if perp_norm < 1e-9:
                continue
            perp = perp / perp_norm
            head_len = abs(scale[i]) * spread
            d1 = back * (head_len * cos_a) + perp * (head_len * sin_a)
            d2 = back * (head_len * cos_a) - perp * (head_len * sin_a)
            color = self._color_for(t[i])
            h_starts.append(tuple(tip))
            h_ends.append(tuple(tip + d1))
            h_starts.append(tuple(tip))
            h_ends.append(tuple(tip + d2))
            h_colors.append(color)
            h_colors.append(color)
            h_widths.append(self.lw)
            h_widths.append(self.lw)

        return h_starts, h_ends, h_colors, h_widths

    # --- shutdown ----------------------------------------------------------

    def close(self) -> None:
        """Clear any in-flight debug lines. Idempotent + exception-safe."""
        try:
            if self._iface is not None:
                self._iface.clear_lines()
        except Exception:
            pass
        self._iface = None
