"""Shared style for all paper figures, built on tueplots.

Every figure is drawn at the size at which it appears in latex/document.tex
and included there without scaling, so that font sizes and line widths agree
across figures and with the text. Each method has one color, marker, and
label, used in every figure.
"""

import shutil

import matplotlib.pyplot as plt
from tueplots import axes, fonts, fontsizes

TEXTWIDTH_IN = 460.72 / 72.27
"""\\textwidth of latex/document.tex (a4paper, 12pt, a4wide), in inches."""

MARKERSIZE = 3.0
FILL_ALPHA = 0.15
REFERENCE_LINE = {"color": "0.5", "linestyle": ":", "linewidth": 0.75}

_ADJOINT = "#1f77b4"
_AUTODIFF = "#ff7f0e"
METHODS = {
    "forward": {"label": "Forward pass", "color": "black", "marker": "o", "linestyle": "--"},
    "forward_noreortho": {
        "label": "Forward pass (no reorth.)",
        "color": "0.55",
        "marker": "o",
        "linestyle": ":",
    },
    "bidiag_adjoint": {"label": "Adjoint (reproj.)", "color": _ADJOINT, "marker": "o", "zorder": 4},
    "bidiag_adjoint_noreproj": {
        "label": "Adjoint (no reproj.)",
        "color": _ADJOINT,
        "marker": "o",
        "markerfacecolor": "white",
        "linestyle": "--",
    },
    "bidiag_autodiff": {"label": "Autodiff", "color": _AUTODIFF, "marker": "s"},
    "bidiag_autodiff_noreortho": {
        "label": "Autodiff (no reorth.)",
        "color": _AUTODIFF,
        "marker": "s",
        "markerfacecolor": "white",
        "linestyle": "--",
    },
    "lanczos_gram": {"label": r"Lanczos adjoint on $A^\top A$", "color": "#d62728", "marker": "^"},
    "arnoldi_augmented": {"label": "Arnoldi adjoint, augmented", "color": "#2ca02c", "marker": "v"},
}


def use_paper_style():
    """Apply the rcParams shared by all figures."""
    plt.rcParams.update(fontsizes.jmlr2001(default_smaller=1))
    if shutil.which("latex"):
        plt.rcParams.update(fonts.jmlr2001_tex(family="serif"))
    else:
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    plt.rcParams.update(axes.lines())
    plt.rcParams.update(axes.legend())
    plt.rcParams.update(axes.grid(grid_alpha=0.3))
    plt.rcParams.update(
        {
            "lines.markersize": MARKERSIZE,
            "figure.constrained_layout.use": True,
            "savefig.pad_inches": 0.015,
            "legend.handlelength": 2.5,
        }
    )


def figure(ncols=1, *, rel_width=1.0, height_to_width_ratio=0.75, **kwargs):
    """Figure with `ncols` panels of the given aspect ratio, `rel_width` of the text wide."""
    width = rel_width * TEXTWIDTH_IN
    height = width / ncols * height_to_width_ratio
    return plt.subplots(1, ncols, figsize=(width, height), **kwargs)


def plot(ax, x, y, method, **kwargs):
    """Line plot of `method` in its shared style."""
    return ax.plot(x, y, **(METHODS[method] | kwargs))


def band(ax, x, lower, upper, method):
    """Shaded range around a line of `method`."""
    ax.fill_between(x, lower, upper, color=METHODS[method]["color"], alpha=FILL_ALPHA, linewidth=0)


def figure_legend(fig, ax, ncols):
    """One legend above all panels, built from the handles of `ax`."""
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=ncols)
