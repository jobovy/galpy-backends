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

# Stable per-backend colours so every comparison figure uses the same mapping.
BACKEND_COLORS = {
    "numpy": "#000000",
    "C": "#666666",
    "jax": "#0072B2",
    "jax-gpu": "#56B4E9",
    "torch": "#D55E00",
    "torch-gpu": "#E69F00",
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
