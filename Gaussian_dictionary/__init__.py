"""Four-state Gaussian-RBF EDMDc tools for the BlueROV simulation.

Exports are loaded lazily so ``python -m Gaussian_dictionary.gaussian_edmdc``
does not import the executable module twice.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONTROL_DIM",
    "CONTROL_NAMES",
    "DICT_DIM",
    "FULL_NU_DIM",
    "MODEL_KIND",
    "STATE_INDICES",
    "STATE_NAMES",
    "GaussianDictionary",
    "GaussianEDMDcModel",
    "fit_dictionary",
    "fit_edmdc",
    "select_controlled_state",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import gaussian_edmdc

    return getattr(gaussian_edmdc, name)
