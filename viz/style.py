"""Shared chart style for every notebook (see the Charts section of AGENTS.md).

Call `use_style()` once at the top of a notebook, then color marks by role
(MODEL, COMPARISON) or by `CATEGORICAL` in order.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Categorical hues, always assigned in this order (validated for color-vision
# deficiency as adjacent pairs; scatter plots and maps use at most the first 3).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
# Roles: our model, what it is compared with, a third series.
MODEL, COMPARISON, THIRD = CATEGORICAL[:3]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"  # titles, labels, reference lines
SURFACE, GRID = "#fcfcfb", "#e1e0d9"
# Magnitude: one hue, light to dark. Polarity: blue and red around a gray midpoint.
SEQUENTIAL = LinearSegmentedColormap.from_list("team_blue", ["#cde2fb", "#2a78d6", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("team_div", ["#2a78d6", "#f0efec", "#e34948"])

FULL_WIDTH, HALF_WIDTH = 6.5, 3.2  # inches, for a letter page with 1 in margins


def use_style() -> None:
    """Set matplotlib defaults: Arial, recessive axes and grid, 300 dpi export."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "text.color": INK, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "lines.linewidth": 2, "lines.markersize": 4.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "figure.figsize": (FULL_WIDTH, 3.5), "figure.dpi": 110,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
        "image.cmap": "team_blue",
    })
    for cmap in (SEQUENTIAL, DIVERGING):
        if cmap.name not in plt.colormaps():
            plt.colormaps.register(cmap)
