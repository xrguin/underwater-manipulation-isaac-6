"""Out-of-process 3D trajectory plot for cascade navigation.

A sibling of ``live_plot_worker.py`` that renders a 3D matplotlib figure
showing:

  * The body's path so far (line of position samples).
  * The current body position (marker).
  * All waypoints (red stars) with the currently active one highlighted
    by a yellow ring. Single-target runs degrade gracefully to one star.
  * The reference polyline through start → wp0 → wp1 → … (dashed).

Runs in its own Python process so its Qt/Tk event loop doesn't clash with
Isaac Sim. Talks to the parent via line-delimited JSON on stdin:

    {"t": <float>, "pos": [x, y, z]}                              ← position update
    {"meta": {"waypoints": [[...], ...], "current_idx": k,        ← multi-waypoint
              "start": [...]}}                                    setup / switch
    {"meta": {"target": [...], "start": [...]}}                   ← single-target
                                                                    (back-compat)
    {"close": true}                                               ← clean shutdown

Optional ``--save-on-close`` writes the final figure to the given path.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

import numpy as np


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", type=str, default="Cascade nav — 3D path (live)")
    ap.add_argument("--update-every", type=int, default=3,
                    help="Redraw every Nth sample. Default 3.")
    ap.add_argument("--save-on-close", type=str, default=None)
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    import matplotlib
    for backend in ("Qt5Agg", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            break
        except Exception:
            continue
    else:
        print("[live-plot-3d] no interactive backend available",
              file=sys.stderr, flush=True)
        return 2
    plt.ion()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(args.title)
    # Distinct window-manager title so users can tell this 3D path plot apart
    # from the 2D velocity plot when both are open.
    try:
        fig.canvas.manager.set_window_title("3D path (live)")
    except Exception:
        pass

    # Pre-create artists; updated as data arrives.
    path_line, = ax.plot([], [], [], color="C0", lw=1.6, label="path")
    body_pt    = ax.scatter([], [], [], s=80, c="C0", marker="o", label="body now")
    waypoints_pt    = ax.scatter([], [], [], s=160, c="r", marker="*",
                                 label="waypoints")
    # current_target_pt = ax.scatter([], [], [], s=380, facecolors="none",
    #                                edgecolors="gold", linewidths=2.2,
    #                                label="current target")
    ref_line,  = ax.plot([], [], [], color="C3", ls="--", lw=1.0,
                         alpha=0.7, label="ref polyline")
    start_pt   = ax.scatter([], [], [], s=80, c="g", marker="s", label="start")
    # Labels for each waypoint — list, recreated each redraw because matplotlib
    # 3D text artists cannot be cleanly re-positioned.
    wp_text_artists: list = []
    ax.legend(loc="best", fontsize=9)
    fig.canvas.draw()
    plt.show(block=False)

    # Buffers
    stop_event   = threading.Event()
    buffer_lock  = threading.Lock()
    full_t:   list[float] = []
    full_xyz: list[np.ndarray] = []
    # waypoints: (N, 3) array or None. current_idx: int. start: (3,) or None.
    meta:     dict = {"waypoints": None, "current_idx": 0, "start": None}

    def _reader():
        for raw in sys.stdin:
            if stop_event.is_set():
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("close"):
                stop_event.set(); break
            if "meta" in msg:
                with buffer_lock:
                    m = msg["meta"]
                    if "start" in m:
                        meta["start"] = list(map(float, m["start"]))
                    if "waypoints" in m:
                        meta["waypoints"] = np.asarray(m["waypoints"], dtype=float)
                    elif "target" in m:
                        # Back-compat: treat single target as a 1-element wp list.
                        meta["waypoints"] = np.asarray([m["target"]], dtype=float)
                    if "current_idx" in m:
                        meta["current_idx"] = int(m["current_idx"])
                continue
            try:
                t = float(msg["t"]); pos = list(map(float, msg["pos"]))
            except (KeyError, ValueError, TypeError):
                continue
            with buffer_lock:
                full_t.append(t)
                full_xyz.append(np.asarray(pos, dtype=float))
        stop_event.set()

    threading.Thread(target=_reader, daemon=True).start()

    def _update_axes_bounds(P: np.ndarray, wps, start):
        # Compose all known points for axis-bound computation.
        pts = [P]
        if wps is not None and len(wps): pts.append(np.asarray(wps).reshape(-1, 3))
        if start is not None:            pts.append(np.asarray(start).reshape(1, 3))
        allP = np.vstack(pts)
        lo = allP.min(axis=0); hi = allP.max(axis=0)
        rng = hi - lo
        pad = np.maximum(0.5, 0.15 * rng)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_zlim(lo[2] - pad[2], hi[2] + pad[2])

    # Cache the last drawn (waypoints, current_idx) so we only rebuild the
    # text labels when something actually changes. Without this cache the
    # labels were removed + recreated every redraw cycle, which on the Qt
    # backend produces a visible flicker (the figure window keeps
    # invalidating + reasserting focus, blocking "minimise").
    _last_meta = {"wps_sig": None, "idx": None}

    def _redraw(P, wps, current_idx, start):
        path_line.set_data(P[:, 0], P[:, 1])
        path_line.set_3d_properties(P[:, 2])
        body_pt._offsets3d = (P[-1:, 0], P[-1:, 1], P[-1:, 2])
        if start is not None:
            s = np.asarray(start)
            start_pt._offsets3d = (s[None, 0], s[None, 1], s[None, 2])
        if wps is not None and len(wps):
            W = np.asarray(wps).reshape(-1, 3)
            waypoints_pt._offsets3d = (W[:, 0], W[:, 1], W[:, 2])
            idx = max(0, min(int(current_idx), W.shape[0] - 1))
            cur = W[idx:idx + 1]
            # current_target_pt._offsets3d = (cur[:, 0], cur[:, 1], cur[:, 2])
            # # Reference polyline: start → wp0 → wp1 → ... → wpN-1
            if start is not None:
                s = np.asarray(start)
                xs = np.concatenate([[s[0]], W[:, 0]])
                ys = np.concatenate([[s[1]], W[:, 1]])
                zs = np.concatenate([[s[2]], W[:, 2]])
            else:
                xs, ys, zs = W[:, 0], W[:, 1], W[:, 2]
            ref_line.set_data(xs, ys)
            ref_line.set_3d_properties(zs)

            # Rebuild waypoint labels ONLY when the waypoint set or the
            # active index changed. ``wps_sig`` is a hashable fingerprint
            # of the float coords; ``idx`` is the active waypoint.
            wps_sig = W.tobytes()
            if (wps_sig != _last_meta["wps_sig"]
                    or idx != _last_meta["idx"]):
                for txt in wp_text_artists:
                    try: txt.remove()
                    except Exception: pass
                wp_text_artists.clear()
                for i, wp in enumerate(W):
                    marker = f"wp{i+1}" + ("*" if i == idx else "")
                    wp_text_artists.append(
                        ax.text(float(wp[0]), float(wp[1]), float(wp[2]) + 0.08,
                                marker, color="r", fontsize=9))
                _last_meta["wps_sig"] = wps_sig
                _last_meta["idx"] = idx

    step = 0
    while not stop_event.is_set():
        with buffer_lock:
            if full_xyz:
                P = np.asarray(full_xyz)
            else:
                P = None
            wps         = meta.get("waypoints")
            current_idx = meta.get("current_idx", 0)
            start       = meta.get("start")
        if P is not None and (step % args.update_every == 0):
            _redraw(P, wps, current_idx, start)
            _update_axes_bounds(P, wps, start)
            fig.canvas.draw_idle()
        # See live_plot_worker.py for the rationale — ``plt.pause`` re-raises
        # the window every iteration; the bare event-loop call avoids it.
        try:
            fig.canvas.start_event_loop(0.03)
        except Exception:
            break
        step += 1

    # Save final figure
    if args.save_on_close:
        try:
            with buffer_lock:
                if full_xyz:
                    P = np.asarray(full_xyz)
                    wps         = meta.get("waypoints")
                    current_idx = meta.get("current_idx", 0)
                    start       = meta.get("start")
                    _redraw(P, wps, current_idx, start)
                    _update_axes_bounds(P, wps, start)
            import os as _os
            _os.makedirs(_os.path.dirname(args.save_on_close) or ".", exist_ok=True)
            fig.savefig(args.save_on_close, dpi=130)
            print(f"[live-plot-3d] saved final figure -> {args.save_on_close}",
                  flush=True)
        except Exception as e:
            print(f"[live-plot-3d] save_on_close failed: {e}",
                  file=sys.stderr, flush=True)

    try: plt.close(fig)
    except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
