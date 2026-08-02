"""Body-frame / COM markers in the Isaac Sim viewport.

Draws coloured point markers each physics step at three locations on the ROV:

    GREEN   — base_link prim origin ("body frame center" reported by RigidPrimView).
    RED     — base_link's PhysX center of mass. This is where PhysX applies the
              forces from `apply_forces_and_torques_at_pos(positions=None, ...)`.
              If the USD's COM is at the prim origin (common for hand-tuned
              vehicles), this coincides with GREEN.
    YELLOW  — combined system COG = mass-weighted average of base_link COM,
              gripper COM, and (optionally) a payload COM, in body frame. Only
              drawn when at least one attachment (gripper / payload) is present;
              otherwise it overlaps RED and is skipped to avoid clutter.
    BLUE    — combined system COB (center of buoyancy) = buoyancy-weighted
              average of all supplied buoyancy sources (ROV hull, gripper,
              payload), in body frame. Only drawn when ``cob_sources`` is given.
              The COG-above/below-COB offset is what produces the passive
              roll/pitch restoring moment, so seeing both markers explains the
              equilibrium attitude (e.g. the forward COG shift from a payload).
    BRIGHT-RED — center of the net applied force (center of pressure),
              passed per-step as ``force_center_world`` to ``update()``. This is
              the point on the applied wrench's line of action nearest the COM,
              i.e. where the combined hydro + buoyancy + thrust force effectively
              pushes. Only drawn when that argument is provided.

Uses the Isaac debug-draw ``draw_points`` call (not ``draw_lines``), so it does not collide
with ThrustVizDrawer's clear_lines() each step — both can run simultaneously.

Usage from bluerov_demo.py:

    body_frame_viz = BodyFrameVizDrawer(
        base_com_body=base_com_offset,    # (3,) m, from base_view.get_coms()
        base_mass_kg=float(masses[0]),
        gripper_mass_kg=0.7 if GRIPPER_ENABLED else 0.0,
        gripper_offset_body=GRIPPER_ATTACH_OFFSET_BODY,
    )
    # ... main loop:
    body_frame_viz.update(pos_w[0], quat_w[0])
    # ... shutdown:
    body_frame_viz.close()
"""

from __future__ import annotations

import numpy as np


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """Isaac Sim quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


class BodyFrameVizDrawer:
    """Per-step debug-draw of body-frame origin + COM markers."""

    # RGBA colours (debug_draw expects 0..1 floats).
    COLOR_ORIGIN = (0.2, 1.0, 0.2, 1.0)     # green
    COLOR_BASE_COM = (1.0, 0.2, 0.2, 1.0)   # red
    COLOR_COMBINED = (1.0, 0.95, 0.1, 1.0)  # yellow
    COLOR_COB = (0.2, 0.6, 1.0, 1.0)        # blue
    COLOR_FORCE = (1.0, 0.0, 0.0, 1.0)      # bright red — center of applied force

    def __init__(
        self,
        base_com_body: np.ndarray,
        base_mass_kg: float,
        gripper_mass_kg: float = 0.0,
        gripper_offset_body: np.ndarray | None = None,
        payload_mass_kg: float = 0.0,
        payload_offset_body: np.ndarray | None = None,
        cob_sources: "list[tuple[float, np.ndarray]] | None" = None,
        show_origin: bool = True,
        show_base_com: bool = True,
        point_size_px: int = 12,
    ):
        """
        Args:
            base_com_body: (3,) base_link COM offset in body frame (metres).
            base_mass_kg: base_link mass in kg (from base_view.get_masses()).
            gripper_mass_kg: gripper mass in kg, or 0.0 if no gripper attached.
            gripper_offset_body: (3,) gripper COM offset in body frame. Used
                only when gripper_mass_kg > 0. If None, treated as zeros.
            payload_mass_kg: payload (e.g. carried cube) mass in kg, or 0.0 if
                none. Folded into the yellow combined-COG marker.
            payload_offset_body: (3,) payload COM offset in body frame. Used
                only when payload_mass_kg > 0. If None, treated as zeros.
            cob_sources: optional list of (buoyancy_N, offset_body) tuples — one
                per buoyancy contribution (ROV hull, gripper, payload). When
                given, their buoyancy-weighted average is drawn as the BLUE COB
                marker. If None / empty, no COB marker is drawn.
            show_origin: draw the GREEN prim-origin marker (default True).
            show_base_com: draw the RED base_link COM marker (default True).
            point_size_px: marker size in screen pixels.
        """
        self.base_com_body = np.asarray(base_com_body, dtype=float).reshape(3)
        self.base_mass = float(base_mass_kg)
        self.gripper_mass = float(gripper_mass_kg)
        if gripper_offset_body is None:
            self.gripper_offset = np.zeros(3)
        else:
            self.gripper_offset = np.asarray(gripper_offset_body, dtype=float).reshape(3)
        self.payload_mass = float(payload_mass_kg)
        if payload_offset_body is None:
            self.payload_offset = np.zeros(3)
        else:
            self.payload_offset = np.asarray(payload_offset_body, dtype=float).reshape(3)
        self._show_origin = bool(show_origin)
        self._show_base_com = bool(show_base_com)
        self.point_size = int(point_size_px)

        # Combined COG in body frame: mass-weighted average over base + any
        # attached bodies (gripper, payload). Only drawn when there is at least
        # one attachment; otherwise it overlaps RED and we skip it.
        masses = [(self.base_mass, self.base_com_body)]
        if self.gripper_mass > 0.0:
            masses.append((self.gripper_mass, self.gripper_offset))
        if self.payload_mass > 0.0:
            masses.append((self.payload_mass, self.payload_offset))
        self._has_combined = len(masses) > 1
        _total_m = sum(m for m, _ in masses)
        self.combined_com_body = sum(
            (m * off for m, off in masses), np.zeros(3)) / _total_m

        # Combined COB in body frame: buoyancy-weighted average of all supplied
        # buoyancy sources. Drawn in BLUE when provided.
        self._has_cob = bool(cob_sources)
        if self._has_cob:
            _total_b = sum(float(n) for n, _ in cob_sources)
            self.combined_cob_body = sum(
                (float(n) * np.asarray(off, dtype=float).reshape(3)
                 for n, off in cob_sources), np.zeros(3)) / _total_b
        else:
            self.combined_cob_body = None

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
            print(f"[body_frame_viz] isaacsim.util.debug_draw unavailable ({e}); "
                  f"markers disabled.")
            self._init_failed = True

    # --- per-step entry point ----------------------------------------------

    def update(self, pos_world: np.ndarray, quat_wxyz_world: np.ndarray,
               force_center_world: np.ndarray | None = None) -> None:
        """Redraw all body-frame markers for the current step.

        Args:
            pos_world: (3,) world-frame position of base_link prim origin.
            quat_wxyz_world: (4,) world-frame orientation, Isaac Sim wxyz.
            force_center_world: optional (3,) world-frame point marking the
                center of the net applied force (center of pressure). Drawn as a
                BRIGHT-RED marker. Already in world frame, so it is drawn
                directly (no body-frame rotation). None = not drawn.
        """
        self._ensure_iface()
        if self._init_failed or self._iface is None:
            return

        R = _quat_wxyz_to_rot(np.asarray(quat_wxyz_world, dtype=float).reshape(4))
        p0 = np.asarray(pos_world, dtype=float).reshape(3)

        points: list = []
        colors: list = []
        sizes: list = []

        if self._show_origin:
            points.append(tuple(p0))
            colors.append(self.COLOR_ORIGIN)
            sizes.append(self.point_size)

        if self._show_base_com:
            base_com_w = p0 + R @ self.base_com_body
            points.append(tuple(base_com_w))
            colors.append(self.COLOR_BASE_COM)
            sizes.append(self.point_size)

        if self._has_combined:
            combined_w = p0 + R @ self.combined_com_body
            points.append(tuple(combined_w))
            colors.append(self.COLOR_COMBINED)
            sizes.append(self.point_size)

        if self._has_cob:
            cob_w = p0 + R @ self.combined_cob_body
            points.append(tuple(cob_w))
            colors.append(self.COLOR_COB)
            sizes.append(self.point_size)

        if force_center_world is not None:
            fc = np.asarray(force_center_world, dtype=float).reshape(3)
            points.append(tuple(fc))
            colors.append(self.COLOR_FORCE)
            sizes.append(self.point_size)   # same size as the COG marker

        try:
            self._iface.clear_points()
            self._iface.draw_points(points, colors, sizes)
        except Exception:
            # Renderer hiccup mid-frame; skip this draw, try again next step.
            pass

    # --- shutdown ----------------------------------------------------------

    def close(self) -> None:
        """Clear any in-flight debug points. Idempotent + exception-safe."""
        try:
            if self._iface is not None:
                self._iface.clear_points()
        except Exception:
            pass
        self._iface = None
