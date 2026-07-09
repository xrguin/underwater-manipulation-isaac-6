"""Out-of-process live velocity tracking plot for EDMDc-MPC.

This script is spawned as a subprocess by `mpc_isaac.py` so the matplotlib
GUI lives in a fresh Python process with its own Qt context — sidestepping
the conflict with Isaac Sim's own Qt event loop.

Protocol:
  stdin is line-delimited JSON. Each line is either
    {"t": float, "nu": [6], "ref": [6]}
  or a control message
    {"close": true}

  EOF on stdin also terminates the plotter cleanly.

Run standalone for debugging:
    ~/miniconda3/envs/marinegym/bin/python -m EDMDc.live_plot_worker --dt 0.0167
    (then pipe JSON lines to its stdin)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

import numpy as np


NU_NAMES = ("u", "v", "w", "p", "q", "r")


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--window-seconds", type=float, default=10.0)
    ap.add_argument("--update-every", type=int, default=3)
    ap.add_argument(
        "--title", type=str,
        default="EDMDc-MPC velocity tracking (live)",
    )
    ap.add_argument(
        "--label-meas", type=str, default="achieved",
        help="Legend label for the 'nu' (solid) trace. Default 'achieved'.",
    )
    ap.add_argument(
        "--label-ref", type=str, default="ref",
        help="Legend label for the 'ref' (dashed) trace. Default 'ref'.",
    )
    ap.add_argument(
        "--save-on-close", type=str, default=None,
        help="If set, save the final figure to this path before closing. "
             "Captures the full trajectory recorded over the live session.",
    )
    return ap


def main() -> int:
    args = _build_argparser().parse_args()

    import matplotlib
    # Pick the first interactive backend that loads; matplotlib will raise
    # on use() if the backend's deps are missing, so try in order.
    for backend in ("Qt5Agg", "TkAgg", "QtAgg"):
        try:
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            break
        except Exception:
            continue
    else:
        print("[live-plot] no interactive matplotlib backend available",
              file=sys.stderr, flush=True)
        return 2
    plt.ion()

    window_steps = max(2, int(args.window_seconds / args.dt))
    times: list[float] = []
    meas:  list[np.ndarray] = []
    refs:  list[np.ndarray] = []
    # Full-history buffers (not rolling). Used to save the final figure.
    full_t:    list[float] = []
    full_meas: list[np.ndarray] = []
    full_refs: list[np.ndarray] = []

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    fig.suptitle(args.title, fontsize=11)
    # Distinct window-manager title so users can tell this 2D velocity plot
    # apart from the 3D path plot when both are open.
    try:
        fig.canvas.manager.set_window_title("Velocity tracking (2D)")
    except Exception:
        pass

    line_meas, line_ref = [], []
    for ax, name in zip(axes, NU_NAMES):
        lref,  = ax.plot([], [], "--", linewidth=2,   color="C0",
                         label=args.label_ref)
        lmeas, = ax.plot([], [],       linewidth=1.5, color="C3",
                         label=args.label_meas)
        ax.set_title(name)
        ax.set_xlabel("t [s]")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        line_ref.append(lref)
        line_meas.append(lmeas)
    fig.tight_layout()
    fig.canvas.draw()
    plt.show(block=False)

    # Stdin is read on a background thread so the GUI event loop stays
    # responsive even when no new data is coming in.
    stop_event = threading.Event()
    buffer_lock = threading.Lock()

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
                nu = np.asarray(msg["nu"], dtype=np.float64)
                ref = np.asarray(msg["ref"], dtype=np.float64)
            except (KeyError, ValueError, TypeError):
                continue
            with buffer_lock:
                times.append(t)
                meas.append(nu)
                refs.append(ref)
                full_t.append(t)
                full_meas.append(nu)
                full_refs.append(ref)
                if len(times) > window_steps:
                    del times[:len(times) - window_steps]
                    del meas[:len(meas) - window_steps]
                    del refs[:len(refs) - window_steps]
        stop_event.set()

    threading.Thread(target=_reader, daemon=True).start()

    step = 0
    while not stop_event.is_set():
        with buffer_lock:
            if not times:
                t_arr = None
            else:
                t_arr = np.asarray(times, dtype=np.float64)
                m_arr = np.asarray(meas)
                r_arr = np.asarray(refs)
        if t_arr is not None and (step % args.update_every == 0):
            for i in range(6):
                line_meas[i].set_data(t_arr, m_arr[:, i])
                line_ref[i].set_data(t_arr, r_arr[:, i])
                ax = axes[i]
                tmax = float(t_arr.max())
                ax.set_xlim(max(0.0, tmax - window_steps * args.dt),
                            max(0.5, tmax))
                lo = min(float(m_arr[:, i].min()), float(r_arr[:, i].min())) - 0.05
                hi = max(float(m_arr[:, i].max()), float(r_arr[:, i].max())) + 0.05
                if hi - lo < 0.1:
                    lo, hi = lo - 0.05, hi + 0.05
                ax.set_ylim(lo, hi)
            fig.canvas.draw_idle()
        # ``plt.pause(0.03)`` internally calls ``show(block=False)`` every
        # iteration which raises the window and steals focus. Running the
        # Qt event loop directly via the canvas avoids that — the window
        # stays minimisable and doesn't flicker.
        try:
            fig.canvas.start_event_loop(0.03)
        except Exception:
            break
        step += 1

    # Save a final image showing the FULL trajectory (not just the rolling
    # window) before closing the window.
    if args.save_on_close:
        try:
            with buffer_lock:
                fT = np.asarray(full_t, dtype=np.float64)
                fM = np.asarray(full_meas) if full_meas else np.zeros((0, 6))
                fR = np.asarray(full_refs) if full_refs else np.zeros((0, 6))
            if fT.size > 0:
                # Re-plot with the full history (no rolling window).
                for i, ax in enumerate(axes):
                    line_meas[i].set_data(fT, fM[:, i])
                    line_ref[i].set_data(fT, fR[:, i])
                    ax.set_xlim(0.0, max(0.5, float(fT.max())))
                    lo = min(float(fM[:, i].min()), float(fR[:, i].min())) - 0.05
                    hi = max(float(fM[:, i].max()), float(fR[:, i].max())) + 0.05
                    if hi - lo < 0.1:
                        lo, hi = lo - 0.05, hi + 0.05
                    ax.set_ylim(lo, hi)
                fig.canvas.draw_idle()
                # Save with non-interactive backend redirection.
                import os as _os
                _os.makedirs(_os.path.dirname(args.save_on_close) or ".", exist_ok=True)
                fig.savefig(args.save_on_close, dpi=130)
                print(f"[live-plot] saved final figure -> {args.save_on_close}",
                      flush=True)
        except Exception as e:
            print(f"[live-plot] save_on_close failed: {e}", file=sys.stderr, flush=True)

    # Close window immediately when the data stream ends.
    try:
        plt.close(fig)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
