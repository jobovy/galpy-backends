"""Shared matplotlib style for all paper figures.

Colourblind-safe palette (Wong 2011, Nature Methods) and PDF settings chosen so
saved figures are byte-stable across runs (see figure.py's CreationDate handling).
"""

import matplotlib as mpl

# Wong 2011 colourblind-friendly palette.
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Stable per-backend colours, from palette.py (the one mapping for every figure):
# a device-agnostic "jax"/"torch" line takes the mid (TITAN Xp) shade.
from .palette import _COLOURS as _P  # noqa: E402

BACKEND_COLORS = {
    "numpy": _P[("numpy", "cpu")],
    "C": _P[("c", "cpu")],
    "jax": _P[("jax", "titan")],
    "jax-cpu": _P[("jax", "cpu")],
    "jax-gpu": _P[("jax", "titan")],
    "jax-h100": _P[("jax", "h100")],
    "torch": _P[("torch", "titan")],
    "torch-cpu": _P[("torch", "cpu")],
    "torch-gpu": _P[("torch", "titan")],
    "torch-h100": _P[("torch", "h100")],
}


def apply_style():
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,  # embed TrueType -> portable, reproducible bytes
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
