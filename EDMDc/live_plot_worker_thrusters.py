"""Out-of-process live thruster plot.

Mirrors ``live_plot_worker.py``: spawned as a subprocess, reads
line-delimited JSON from stdin, renders a rolling-window matplotlib figure
with one subplot per thruster. Each subplot AUTO-SCALES its y-axis
independently -- so a 0.2 N micro-thrust near a waypoint is just as
readable as a 60 N cruise thrust.

This sidesteps the Isaac Sim "thruster arrow" glyph problem: those scale
with magnitude, so they collapse to invisible at low thrust. Here the
y-axis re-fits each frame, so small values stay visible.

Protocol:
    stdin is line-delimited JSON. Each line is either
        {"t": float, "thrusts": [N floats]}
    or
        {"close": true}
    EOF on stdin also terminates cleanly.

The number of thrusters N is inferred from the first message.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

import numpy as np


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--window-seconds", type=float, default=10.0)
    ap.add_argument("--update-every", type=int, default=3)
    ap.add_argument("--title", type=str,
                    default="Thruster commands (live)")
    ap.add_argument("--n-thrusters", type=int, default=8,
                    help="Used to lay out subplots before the first message "
                         "arrives. Will adapt to incoming data length.")
    ap.add_argument("--max-thrust", type=float, default=0.0,
                    help="If > 0, draw +/- horizontal lines at this magnitude "
                         "as a saturation reference. Default off (auto-scale "
                         "only).")
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
        print("[live-plot-thr] no interactive matplotlib backend available",
              file=sys.stderr, flush=True)
        return 2
    plt.ion()

    window_steps = max(2, int(args.window_seconds / args.dt))
    n_thr = int(args.n_thrusters)

    # Layout: 2 rows x ceil(N/2) cols. Each panel auto-scales.
    n_cols = (n_thr + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 5.2),
                             sharex=True)
    axes = np.atleast_1d(axes).flatten()[:n_thr]
    fig.suptitle(args.title, fontsize=11)
    try:
        fig.canvas.manager.set_window_title("Thrusters (live)")
    except Exception:
        pass

    # One colored line per thruster + a title that updates with the current
    # value (the user wanted "thrust magnitude readable even near the
    # waypoint" -- the title gives the live numeric value alongside the
    # plot).
    cmap = plt.cm.get_cmap("tab10", max(10, n_thr))
    lines = []
    base_titles = []
    for i, ax in enumerate(axes):
        ln, = ax.plot([], [], color=cmap(i), lw=1.4)
        ax.set_ylabel("N")
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.5)
        if args.max_thrust > 0:
            ax.axhline( args.max_thrust, color="r", ls=":", lw=0.6, alpha=0.6)
            ax.axhline(-args.max_thrust, color="r", ls=":", lw=0.6, alpha=0.6)
        base_titles.append(f"T{i}")
        ax.set_title(base_titles[i], fontsize=9)
        lines.append(ln)
    for ax in axes[-n_cols:]:
        ax.set_xlabel("t [s]")
    fig.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)

    times: list[float] = []
    thrusts: list[np.ndarray] = []
    full_t: list[float] = []
    full_thr: list[np.ndarray] = []

    stop_event   = threading.Event()
    buffer_lock  = threading.Lock()

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
                stop_event.set()
                break
            try:
                t = float(msg["t"])
                thr = np.asarray(msg["thrusts"], dtype=np.float64)
            except (KeyError, ValueError, TypeError):
                continue
            with buffer_lock:
                times.append(t)
                thrusts.append(thr)
                full_t.append(t)
                full_thr.append(thr)
                if len(times) > window_steps:
                    del times[:len(times) - window_steps]
                    del thrusts[:len(thrusts) - window_steps]
        stop_event.set()

    threading.Thread(target=_reader, daemon=True).start()

    step = 0
    while not stop_event.is_set():
        with buffer_lock:
            if not times:
                t_arr = None
            else:
                t_arr = np.asarray(times, dtype=np.float64)
                T_arr = np.vstack(thrusts)

        if t_arr is not None and (step % args.update_every == 0):
            tmax = float(t_arr.max())
            xlo  = max(0.0, tmax - window_steps * args.dt)
            xhi  = max(0.5, tmax)
            last = T_arr[-1]
            for i in range(min(n_thr, T_arr.shape[1])):
                lines[i].set_data(t_arr, T_arr[:, i])
                ax = axes[i]
                ax.set_xlim(xlo, xhi)
                lo = float(T_arr[:, i].min())
                hi = float(T_arr[:, i].max())
                pad = max(0.5, 0.15 * max(abs(lo), abs(hi)))
                ax.set_ylim(lo - pad, hi + pad)
                # Live numeric readout in the subplot title -- the whole
                # point of this plot is being able to read off small
                # commanded thrusts near the waypoints.
                ax.set_title(f"{base_titles[i]}: {last[i]:+.2f} N",
                             fontsize=9)
            fig.canvas.draw_idle()
        try:
            fig.canvas.start_event_loop(0.03)
        except Exception:
            break
        step += 1

    # Save final figure with FULL history (not rolling window).
    if args.save_on_close:
        try:
            with buffer_lock:
                fT = np.asarray(full_t, dtype=np.float64)
                fX = np.vstack(full_thr) if full_thr else np.zeros((0, n_thr))
            if fT.size > 0:
                ncol = min(n_thr, fX.shape[1])
                for i in range(ncol):
                    lines[i].set_data(fT, fX[:, i])
                    ax = axes[i]
                    ax.set_xlim(0.0, max(0.5, float(fT.max())))
                    lo = float(fX[:, i].min())
                    hi = float(fX[:, i].max())
                    pad = max(0.5, 0.15 * max(abs(lo), abs(hi)))
                    ax.set_ylim(lo - pad, hi + pad)
                fig.canvas.draw_idle()
                import os as _os
                _os.makedirs(_os.path.dirname(args.save_on_close) or ".",
                             exist_ok=True)
                fig.savefig(args.save_on_close, dpi=130)
                print(f"[live-plot-thr] saved final figure -> "
                      f"{args.save_on_close}", flush=True)
        except Exception as e:
            print(f"[live-plot-thr] save_on_close failed: {e}",
                  file=sys.stderr, flush=True)

    try:
        plt.close(fig)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
