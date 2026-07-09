"""Shared matplotlib styling for paper figures (IEEE two-column).

All paper figures are saved at the SAME width (`FIG_W`) so that, when each is
placed at the column/text width in LaTeX, the absolute text size is identical
across figures. Call :func:`apply` once at the top of a plotting script.
"""
from __future__ import annotations

import matplotlib as mpl

# IEEE two-column: full text width ≈ 7.16 in (\textwidth), one column ≈ 3.5 in.
# These figures span both columns (wide), so use the full text width.
FIG_W = 7.16


def apply() -> None:
    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
