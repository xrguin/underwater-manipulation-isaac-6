"""Small Isaac Sim 6 compatibility boundary for the underwater simulator.

This module intentionally presents the handful of NumPy-oriented operations
used by the existing plant code while implementing them with Isaac Sim 6 Core
Experimental APIs.  Import it only after ``SimulationApp`` has started.

The compatibility surface is deliberately narrow:

* :class:`World` owns stage creation, timeline state, physics stepping, and
  optional rendering.
* :class:`RigidPrimView` preserves the old single/multi-prim NumPy method
  shapes while wrapping ``isaacsim.core.experimental.prims.RigidPrim``.
* :class:`Articulation` adapts the constructor used by the existing scripts.
* :class:`Camera` wraps the renamed Isaac 6 camera API.
* :func:`add_reference_to_stage` retains the old keyword spelling.

Hydrodynamics, thruster allocation, controller interfaces, and dataset code do
not depend on Isaac modules through this file.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import (
    Articulation as _ExperimentalArticulation,
)
from isaacsim.core.experimental.prims import RigidPrim as _ExperimentalRigidPrim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.camera import Camera as _Isaac6Camera


def _numpy(value: Any) -> np.ndarray:
    """Copy an Isaac/Warp/NumPy value to a regular NumPy array."""
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).copy()


def add_reference_to_stage(*, usd_path: str, prim_path: str):
    """Isaac 6 stage-reference helper using the legacy keyword contract."""
    return stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)


def get_current_stage():
    """Return the current USD stage."""
    return stage_utils.get_current_stage(backend="usd")


class _SceneRegistry:
    """Compatibility no-op for the former ``world.scene.add`` registry.

    Core Experimental prim wrappers subscribe to physics lifecycle events
    themselves, so explicit registration is unnecessary.  Returning the
    object keeps call sites and any direct handles intact.
    """

    @staticmethod
    def add(obj):
        return obj


class World:
    """Minimal replacement for the deprecated Core ``World`` class."""

    def __init__(
        self,
        *,
        stage_units_in_meters: float = 1.0,
        physics_dt: float = 1.0 / 60.0,
        backend: str = "numpy",
    ):
        if not np.isclose(float(stage_units_in_meters), 1.0):
            raise ValueError("This project requires a metre-scale USD stage")
        if backend != "numpy":
            raise ValueError("The underwater simulator compatibility layer uses NumPy")
        self._dt = float(physics_dt)
        self.scene = _SceneRegistry()
        stage_utils.create_new_stage()
        # Match the old NumPy World path: CPU PhysX with a fixed physics step.
        SimulationManager.setup_simulation(dt=self._dt, device="cpu")
        RenderingManager.set_dt(self._dt)

    @property
    def is_playing(self) -> bool:
        return bool(app_utils.is_playing())

    @property
    def is_paused(self) -> bool:
        return bool(app_utils.is_paused())

    @property
    def is_stopped(self) -> bool:
        return bool(app_utils.is_stopped())

    def reset(self) -> None:
        """(Re)initialize physics and all Experimental prim tensor views."""
        if not app_utils.is_stopped():
            app_utils.stop()
            app_utils.update_app()
        app_utils.play()
        # Committing one app update triggers SimulationManager warmup and the
        # PHYSICS_READY callbacks registered by RigidPrim/Articulation.
        app_utils.update_app()

    def step(self, *, render: bool = False) -> bool:
        """Step exactly one physics tick.

        When the timeline is paused or stopped, no physics call is made.  A
        render-only update may still be pumped so the GUI remains responsive.
        The boolean return lets callers distinguish a real physics step.
        """
        if not app_utils.is_playing():
            if render:
                RenderingManager.render()
            return False
        if SimulationManager.get_physics_simulation_view() is None:
            SimulationManager.initialize_physics()
        SimulationManager.step()
        if render:
            RenderingManager.render()
        return True

    @staticmethod
    def pump_app() -> None:
        """Process one render/UI update without advancing physics."""
        RenderingManager.render()


class Articulation:
    """Constructor adapter for Core Experimental ``Articulation``."""

    def __init__(
        self,
        *,
        prim_path: str,
        name: str | None = None,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ):
        del name
        positions = None if position is None else np.asarray(position).reshape(1, 3)
        orientations = (
            None if orientation is None else np.asarray(orientation).reshape(1, 4)
        )
        self._impl = _ExperimentalArticulation(
            prim_path,
            positions=positions,
            orientations=orientations,
        )

    @property
    def impl(self):
        return self._impl


class RigidPrimView:
    """NumPy adapter over Core Experimental ``RigidPrim``."""

    def __init__(self, *, prim_paths_expr: str, name: str | None = None):
        del name
        self._impl = _ExperimentalRigidPrim(prim_paths_expr)

    @property
    def impl(self):
        return self._impl

    @property
    def physics_view_valid(self) -> bool:
        return bool(self._impl.is_physics_tensor_entity_valid())

    def _require_physics_view(self) -> None:
        if self.physics_view_valid:
            return
        if app_utils.is_playing():
            SimulationManager.initialize_physics()
        if not self.physics_view_valid:
            raise RuntimeError(
                "Isaac 6 rigid-body physics view is unavailable "
                "(timeline paused/stopped or physics not initialized)"
            )

    def get_world_poses(self):
        self._require_physics_view()
        positions, orientations = self._impl.get_world_poses()
        return _numpy(positions), _numpy(orientations)

    def get_linear_velocities(self):
        self._require_physics_view()
        linear, _ = self._impl.get_velocities()
        return _numpy(linear)

    def get_angular_velocities(self):
        self._require_physics_view()
        _, angular = self._impl.get_velocities()
        return _numpy(angular)

    def set_world_poses(self, *, positions=None, orientations=None) -> None:
        self._require_physics_view()
        self._impl.set_world_poses(
            positions=positions,
            orientations=orientations,
        )

    def set_velocities(
        self,
        *,
        velocities=None,
        linear_velocities=None,
        angular_velocities=None,
    ) -> None:
        self._require_physics_view()
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=float).reshape(-1, 6)
            linear_velocities = velocities[:, :3]
            angular_velocities = velocities[:, 3:]
        self._impl.set_velocities(
            linear_velocities=linear_velocities,
            angular_velocities=angular_velocities,
        )

    def apply_forces_and_torques_at_pos(
        self,
        *,
        forces=None,
        torques=None,
        positions=None,
        is_global: bool = True,
    ) -> None:
        self._require_physics_view()
        self._impl.apply_forces_and_torques_at_pos(
            forces=forces,
            torques=torques,
            positions=positions,
            local_frame=not bool(is_global),
        )

    def get_masses(self):
        masses = _numpy(self._impl.get_masses())
        return masses.reshape(-1)

    def get_coms(self):
        self._require_physics_view()
        positions, orientations = self._impl.get_coms()
        return _numpy(positions), _numpy(orientations)


class Camera:
    """Compatibility wrapper over the renamed Isaac 6 camera API.

    Core Experimental is used for the physics-facing API above.  CameraSensor
    from ``isaacsim.sensors.experimental.rtx`` currently blocks while creating
    a render product after the standalone timeline is already running in the
    Isaac Sim 6.0 release candidate.  The renamed high-level Camera API remains
    available in Isaac 6 and preserves the existing ``initialize/get_rgba``
    contract, so the workaround stays contained here.
    """

    def __init__(self, *, prim_path: str, resolution: tuple[int, int]):
        self._prim_path = str(prim_path)
        self._resolution_wh = (int(resolution[0]), int(resolution[1]))
        self._sensor: _Isaac6Camera | None = None

    def initialize(self) -> None:
        if self._sensor is None:
            self._sensor = _Isaac6Camera(
                self._prim_path,
                resolution=self._resolution_wh,
            )
            self._sensor.initialize()

    def get_rgba(self):
        if self._sensor is None:
            self.initialize()
        data = self._sensor.get_rgba()
        return None if data is None else _numpy(data)
