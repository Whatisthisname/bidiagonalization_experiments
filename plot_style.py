"""Shared plotting style for all paper figures.

Centralizes the look so every figure matches: tueplots handles fonts and
sizing, and the visual conventions (log y-axis, "-o" markers, light grid,
legend) follow the benchmark "backward time" figure.
"""

import matplotlib.pyplot as plt
from tueplots import axes, figsizes, fontsizes

# Semantic colors shared across the timing figures.
FORWARD_COLOR = "black"
ADJOINT_COLOR = "#1f77b4"
AUTODIFF_COLOR = "#ff7f0e"

MARKER = "o"
MARKERSIZE = 3.0
LINEWIDTH = 1.25
GRID_ALPHA = 0.3
FIGSIZE = (5, 3.2)


def use_paper_style():
    """Apply the tueplots-based rcParams used by every figure."""
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(figsizes.iclr2024(rel_width=0.4, height_to_width_ratio=1.25))
    plt.rcParams.update(fontsizes.iclr2024(default_smaller=2))


def style_axis(ax, xlabel, ylabel, *, logy=True, legend=True):
    """Apply consistent axis labels, log-scale, grid, and legend."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=GRID_ALPHA)
    if legend:
        ax.legend()
