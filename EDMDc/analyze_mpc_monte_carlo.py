"""Analyze paired MPC Monte Carlo navigation results."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ._plot_style import FIG_W, apply as apply_plot_style


SCENARIO_LABEL = {
    "with_gripper": "gripper only",
    "with_gripper_cube": "gripper + cube",
}
CONTROLLER_LABEL = {
    "edmdc": "EDMDc",
    "memory_h3": "EDMDc + memory h=3",
}


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("returncode") != "0":
                rows.append(row)
                continue
            for key in (
                "trial_idx", "success", "reached", "n_waypoints",
                "completion_time_s", "final_wp_error_m", "mean_wp_error_m",
                "vel_rmse", "cmd_rms", "cmd_sat_frac",
                "max_abs_roll_pitch_deg", "elapsed_wall_s",
            ):
                if key in row and row[key] not in ("", "nan"):
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _binom_two_sided_p(k: int, n: int) -> float:
    if n == 0:
        return math.nan
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        # Exact enough for the small discordant table without requiring scipy.
        probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
        pk = probs[k]
        return float(min(1.0, sum(p for p in probs if p <= pk + 1e-15)))


def _success_summary(rows: list[dict]) -> list[dict]:
    out = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["controller"])].append(row)
    for (scenario, controller), vals in sorted(groups.items()):
        n = len(vals)
        k = sum(int(float(v.get("success", 0))) for v in vals)
        p, lo, hi = _wilson(k, n)
        out.append({
            "scenario": scenario,
            "controller": controller,
            "n": n,
            "successes": k,
            "success_rate": p,
            "ci_low": lo,
            "ci_high": hi,
        })
    return out


def _paired_summary(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for row in rows:
        if row.get("returncode") != "0":
            continue
        key = (row["scenario"], int(float(row["trial_idx"])))
        by_key[key][row["controller"]] = int(float(row.get("success", 0)))

    out = []
    for scenario in sorted({k[0] for k in by_key}):
        both_success = ed_only = mem_only = both_fail = 0
        for (sc, _trial), vals in by_key.items():
            if sc != scenario or "edmdc" not in vals or "memory_h3" not in vals:
                continue
            ed = vals["edmdc"]
            mem = vals["memory_h3"]
            if ed and mem:
                both_success += 1
            elif ed and not mem:
                ed_only += 1
            elif mem and not ed:
                mem_only += 1
            else:
                both_fail += 1
        discordant = ed_only + mem_only
        out.append({
            "scenario": scenario,
            "both_success": both_success,
            "edmdc_only": ed_only,
            "memory_only": mem_only,
            "both_fail": both_fail,
            "memory_minus_edmdc": mem_only - ed_only,
            "mcnemar_exact_p": _binom_two_sided_p(min(ed_only, mem_only), discordant),
        })
    return out


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_success(summary: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = []
    heights = []
    yerr = [[], []]
    colors = []
    for row in summary:
        labels.append(
            f"{SCENARIO_LABEL.get(row['scenario'], row['scenario'])}\n"
            f"{CONTROLLER_LABEL.get(row['controller'], row['controller'])}"
        )
        heights.append(row["success_rate"])
        yerr[0].append(row["success_rate"] - row["ci_low"])
        yerr[1].append(row["ci_high"] - row["success_rate"])
        colors.append("C0" if row["controller"] == "edmdc" else "C1")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
    ax.bar(x, heights, yerr=yerr, capsize=4, color=colors)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("success rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "success_rate_ci.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _plot_ecdf(rows: list[dict], key: str, xlabel: str, out_name: str,
               out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.8), sharey=True)
    scenarios = ["with_gripper", "with_gripper_cube"]
    for ax, scenario in zip(axes, scenarios):
        for controller, color in (("edmdc", "C0"), ("memory_h3", "C1")):
            vals = np.array([
                float(r[key]) for r in rows
                if r.get("returncode") == "0"
                and r["scenario"] == scenario
                and r["controller"] == controller
                and r.get(key, "") not in ("", "nan")
            ], dtype=float)
            x, y = _ecdf(vals)
            if x.size:
                ax.step(x, y, where="post", label=CONTROLLER_LABEL[controller],
                        color=color)
        ax.set_title(SCENARIO_LABEL[scenario])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ECDF")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, loc="lower right", fontsize=8)
    fig.tight_layout()
    out = out_dir / out_name
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_paired(paired: list[dict], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    labels = [SCENARIO_LABEL.get(r["scenario"], r["scenario"]) for r in paired]
    both_success = np.array([r["both_success"] for r in paired], dtype=float)
    memory_only = np.array([r["memory_only"] for r in paired], dtype=float)
    ed_only = np.array([r["edmdc_only"] for r in paired], dtype=float)
    both_fail = np.array([r["both_fail"] for r in paired], dtype=float)
    totals = both_success + memory_only + ed_only + both_fail
    totals[totals == 0] = 1.0
    stacks = [both_success / totals, memory_only / totals,
              ed_only / totals, both_fail / totals]
    names = ["both success", "memory only", "EDMDc only", "both fail"]
    colors = ["#4daf4a", "#ff7f00", "#377eb8", "#999999"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    bottom = np.zeros_like(x, dtype=float)
    for vals, name, color in zip(stacks, names, colors):
        ax.bar(x, vals, bottom=bottom, label=name, color=color)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of paired trials")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper center", ncol=4, frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "paired_success_outcomes.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _write_markdown(out_dir: Path, success: list[dict], paired: list[dict],
                    failure_counts: dict[tuple[str, str], Counter]) -> Path:
    out = out_dir / "summary.md"
    lines = ["# MPC Monte Carlo Summary", ""]
    lines.append("## Success Rate")
    lines.append("")
    lines.append("| Scenario | Controller | n | Success | 95% CI |")
    lines.append("|---|---|---:|---:|---:|")
    for r in success:
        lines.append(
            f"| {SCENARIO_LABEL.get(r['scenario'], r['scenario'])} "
            f"| {CONTROLLER_LABEL.get(r['controller'], r['controller'])} "
            f"| {r['n']} | {100*r['success_rate']:.1f}% "
            f"| [{100*r['ci_low']:.1f}, {100*r['ci_high']:.1f}]% |"
        )
    lines.append("")
    lines.append("## Paired Outcomes")
    lines.append("")
    lines.append("| Scenario | Both success | Memory only | EDMDc only | Both fail | McNemar p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in paired:
        lines.append(
            f"| {SCENARIO_LABEL.get(r['scenario'], r['scenario'])} "
            f"| {r['both_success']} | {r['memory_only']} | {r['edmdc_only']} "
            f"| {r['both_fail']} | {r['mcnemar_exact_p']:.4g} |"
        )
    lines.append("")
    lines.append("## Failure Counts")
    lines.append("")
    lines.append("| Scenario | Controller | Failure reason | Count |")
    lines.append("|---|---|---|---:|")
    for (scenario, controller), counts in sorted(failure_counts.items()):
        for reason, count in sorted(counts.items()):
            lines.append(
                f"| {SCENARIO_LABEL.get(scenario, scenario)} "
                f"| {CONTROLLER_LABEL.get(controller, controller)} "
                f"| {reason} | {count} |"
            )
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.csv.with_suffix("").parent / (args.csv.stem + "_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()
    rows = _read_rows(args.csv)
    success = _success_summary(rows)
    paired = _paired_summary(rows)
    failure_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        failure_counts[(row["scenario"], row["controller"])][row["failure_reason"]] += 1

    _write_table(out_dir / "success_rate_summary.csv", success)
    _write_table(out_dir / "paired_success_summary.csv", paired)
    outputs = [
        _plot_success(success, out_dir),
        _plot_paired(paired, out_dir),
        _plot_ecdf(rows, "final_wp_error_m", "final waypoint error [m]",
                   "final_waypoint_error_ecdf.png", out_dir),
        _plot_ecdf(rows, "completion_time_s", "completion time [s]",
                   "completion_time_ecdf.png", out_dir),
        _plot_ecdf(rows, "vel_rmse", "velocity tracking RMSE",
                   "velocity_rmse_ecdf.png", out_dir),
        _write_markdown(out_dir, success, paired, failure_counts),
    ]
    for out in outputs:
        print(f"[mc-analysis] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
