"""Subprocess proxies for the out-of-process live-plot workers.

Two thin client classes — ``LivePlot2D`` and ``LivePlot3D`` — that spawn
the matching ``live_plot_worker``/``live_plot_worker_3d`` module in a
separate Python process and feed it line-delimited JSON. Running the GUI
in a child process is the workaround for the Qt/event-loop conflict
between matplotlib and Isaac Sim's ``SimulationApp`` (see
``memory/mpc_live_plot_qt_conflict.md`` for the original incident).

Pure stdlib + numpy; no Isaac dependency, so this module can be unit-
tested or used from any Python process. The workers themselves live in
``EDMDc/live_plot_worker.py`` (2D, 6-axis ν vs ref) and
``EDMDc/live_plot_worker_3d.py`` (3D path + waypoints).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


# Default interpreter used to run the live-plot subprocess. The marinegym
# environment ships matplotlib + Qt; the Isaac Sim Python does NOT (and even
# if it did, we'd hit the same Qt conflict that motivated this split). Allow
# override via the ``LIVE_PLOT_PYTHON`` env var.
DEFAULT_LIVE_PLOT_PYTHON = os.environ.get(
    "LIVE_PLOT_PYTHON",
    "/home/xzha/miniconda3/envs/marinegym/bin/python",
)


def _project_root() -> str:
    """Project root so subprocess ``-m EDMDc.live_plot_worker`` resolves."""
    return str(Path(__file__).resolve().parents[1])


class LivePlot2D:
    """Pipe-fed 6-axis velocity plot.

    Spawns ``EDMDc.live_plot_worker`` and streams ``(t, nu_meas, nu_ref)``
    tuples to it. The worker renders a rolling-window plot with one
    subplot per ν component.

    Parameters
    ----------
    dt : float
        Sim time step (seconds). Used by the worker to size the rolling
        window in samples.
    title : str
        Window title, also used as figure suptitle.
    window_seconds : float
        Width of the rolling time window, in seconds.
    update_every : int
        Redraw every Nth sample. Default 3 (≈20 Hz at dt=1/60).
    label_meas, label_ref : str
        Legend labels for the measurement and reference traces.
    save_on_close : str | None
        If set, the worker writes the final figure to this path before
        exiting (Agg-rendered, headless-safe).
    python_exe : str | None
        Override the subprocess interpreter. Falls back to env var
        ``LIVE_PLOT_PYTHON`` or the marinegym default.
    """

    def __init__(self, *, dt: float, title: str,
                 window_seconds: float, update_every: int = 3,
                 label_meas: str = "Isaac",
                 label_ref: str = "ref",
                 save_on_close: Optional[str] = None,
                 python_exe: Optional[str] = None):
        py = python_exe or DEFAULT_LIVE_PLOT_PYTHON
        cmd = [py, "-u", "-m", "EDMDc.live_plot_worker",
               "--dt", f"{dt}",
               "--window-seconds", f"{window_seconds}",
               "--update-every", f"{update_every}",
               "--title", title,
               "--label-meas", label_meas,
               "--label-ref",  label_ref]
        if save_on_close is not None:
            cmd += ["--save-on-close", str(save_on_close)]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, cwd=_project_root())
            print(f"[live-plot-2d] worker spawned (pid={self._proc.pid})")
        except FileNotFoundError:
            print(f"[live-plot-2d] disabled (python {py!r} not found)")
            self._proc = None

    def push(self, t: float, nu_meas: np.ndarray, nu_ref: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            msg = json.dumps({
                "t":   float(t),
                "nu":  [float(x) for x in nu_meas],
                "ref": [float(x) for x in nu_ref],
            })
            self._proc.stdin.write((msg + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.write(b'{"close": true}\n')
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=5.0)   # give the worker time to save the PNG
        except Exception:
            try: self._proc.terminate()
            except Exception: pass
        self._proc = None


class LivePlot3D:
    """Pipe-fed 3D path plot with multi-waypoint support.

    Spawns ``EDMDc.live_plot_worker_3d``. Meta updates carry the full
    waypoint list + current target index; tick updates carry ``(t, pos)``.
    The worker draws the path so far, all waypoints (red stars), the
    currently-active target (gold ring), and a dashed reference polyline
    from start through all waypoints. Backward-compatible with the
    single-target ``{"target": [x, y, z]}`` form used by older scripts.

    Parameters
    ----------
    title : str
        Window title.
    update_every : int
        Redraw every Nth sample. Default 3.
    save_on_close : str | None
        If set, the worker writes the final figure to this path before
        exiting.
    python_exe : str | None
        Override the subprocess interpreter.
    """

    def __init__(self, *, title: str, update_every: int = 3,
                 save_on_close: Optional[str] = None,
                 python_exe: Optional[str] = None):
        py = python_exe or DEFAULT_LIVE_PLOT_PYTHON
        cmd = [py, "-u", "-m", "EDMDc.live_plot_worker_3d",
               "--title", title,
               "--update-every", f"{update_every}"]
        if save_on_close is not None:
            cmd += ["--save-on-close", str(save_on_close)]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, cwd=_project_root())
            print(f"[live-plot-3d] worker spawned (pid={self._proc.pid})")
        except FileNotFoundError:
            print(f"[live-plot-3d] disabled (python {py!r} not found)")
            self._proc = None

    def push_meta(self, *,
                  start: Optional[np.ndarray] = None,
                  waypoints: Optional[np.ndarray] = None,
                  current_idx: Optional[int] = None,
                  target: Optional[np.ndarray] = None) -> None:
        """Send a (partial) meta update.

        Pass any subset of fields; ``None`` fields are omitted. ``target``
        is the legacy single-target field — the worker treats it as a
        1-element waypoint list when ``waypoints`` is not given.
        """
        if self._proc is None or self._proc.stdin is None:
            return
        payload: dict = {}
        if start is not None:
            payload["start"] = [float(x) for x in start]
        if waypoints is not None:
            payload["waypoints"] = [
                [float(x) for x in wp]
                for wp in np.asarray(waypoints).reshape(-1, 3)
            ]
        if current_idx is not None:
            payload["current_idx"] = int(current_idx)
        if target is not None:
            payload["target"] = [float(x) for x in target]
        if not payload:
            return
        try:
            msg = json.dumps({"meta": payload})
            self._proc.stdin.write((msg + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None

    def push(self, t: float, pos_world: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            msg = json.dumps({
                "t":   float(t),
                "pos": [float(x) for x in pos_world],
            })
            self._proc.stdin.write((msg + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.write(b'{"close": true}\n')
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            try: self._proc.terminate()
            except Exception: pass
        self._proc = None


class LivePlotThrusters:
    """Pipe-fed per-thruster plot.

    Spawns ``EDMDc.live_plot_worker_thrusters`` and streams
    ``(t, thrusts)`` tuples to it. Each thruster gets its own subplot with
    an AUTO-SCALED y-axis -- so small commanded thrusts near a waypoint
    stay visible (the Isaac arrow glyphs collapse to invisible there).

    Parameters
    ----------
    dt : float
        Sim time step (seconds).
    title : str
        Window title / figure suptitle.
    window_seconds : float
        Width of the rolling time window, in seconds.
    n_thrusters : int
        Number of thruster channels (subplot count).
    max_thrust : float
        If > 0, the worker draws dotted +/- lines at this magnitude as a
        saturation reference. Default 0 (off, pure auto-scale).
    update_every : int
        Redraw every Nth sample.
    save_on_close : str | None
        Final figure path (full history, not rolling window).
    python_exe : str | None
        Override the subprocess interpreter.
    """

    def __init__(self, *, dt: float, title: str,
                 window_seconds: float, n_thrusters: int,
                 max_thrust: float = 0.0,
                 update_every: int = 3,
                 save_on_close: Optional[str] = None,
                 python_exe: Optional[str] = None):
        py = python_exe or DEFAULT_LIVE_PLOT_PYTHON
        cmd = [py, "-u", "-m", "EDMDc.live_plot_worker_thrusters",
               "--dt", f"{dt}",
               "--window-seconds", f"{window_seconds}",
               "--update-every", f"{update_every}",
               "--n-thrusters", f"{int(n_thrusters)}",
               "--max-thrust", f"{float(max_thrust)}",
               "--title", title]
        if save_on_close is not None:
            cmd += ["--save-on-close", str(save_on_close)]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, cwd=_project_root())
            print(f"[live-plot-thr] worker spawned (pid={self._proc.pid})")
        except FileNotFoundError:
            print(f"[live-plot-thr] disabled (python {py!r} not found)")
            self._proc = None

    def push(self, t: float, thrusts: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            msg = json.dumps({
                "t":       float(t),
                "thrusts": [float(x) for x in thrusts],
            })
            self._proc.stdin.write((msg + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.write(b'{"close": true}\n')
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            try: self._proc.terminate()
            except Exception: pass
        self._proc = None
