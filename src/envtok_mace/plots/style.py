"""One visual system for every figure in the repository (Nature conventions).

Palette: only blue, grey and deep-red families, low saturation (OKLCH chroma
<= 0.095), and each model on its own lightness step, dark to light:

    this work        navy        L 0.31   the series the figure is about
    authors, DIRECT  deep red    L 0.48   a different selection method
    authors, EnvTok  slate blue  L 0.63   same method as ours, lighter tier of our family
    authors, random  light grey  L 0.80   context: four seeds and their mean

The steps are at least 0.15 apart in lightness, so the models stay apart in
greyscale print and for colour-blind readers (worst pair: OKLab Delta E 18.6 for
normal vision, 15.2 under protan/deutan simulation). MACE-OMAT-0, trained on
all of OMat24, is an upper bound rather than a competitor: it is drawn as a
dashed dark-grey reference line and always labelled where it is drawn.

Every model also has its own marker, so identity never depends on colour alone;
text is set in ink, never in a series colour. Plots sit on white with a closed
frame (bare top and right spines, ticks only bottom and left); SVG text stays
editable.
"""

from __future__ import annotations

from contextlib import contextmanager

INK = {"ink": "#272727", "ink2": "#4d4d4d", "muted": "#767676", "frame": "#4d4d4d",
       "grid": "#e6e6e6", "surface": "#ffffff", "chip": "#ededed"}

# model -> label, colours, marker; the order is the legend order (dark to light)
MODELS: dict[str, dict] = {
    "ours": {"label": "This work (EnvTok)", "color": "#122f5a", "edge": "#ffffff", "marker": "o"},
    "authors_direct": {"label": "Authors, DIRECT", "color": "#8b4643", "edge": "#ffffff", "marker": "^"},
    "authors_envtok": {"label": "Authors, EnvTok", "color": "#6a8cb3", "edge": "#ffffff", "marker": "s"},
    "random": {"label": "Authors, random (4 seeds)", "color": "#bebebe", "edge": "#717171", "marker": "D"},
    "mace_omat0": {"label": "MACE-OMAT-0 (all of OMat24)", "color": "#4d4d4d", "marker": None,
                   "linestyle": (0, (4, 2.5))},
}

# test data ordered by distance from training; shown as text chips and line styles, never as colour
DISTANCE = {"ID": "in distribution", "SHIFT": "dataset shift", "OOD": "out of distribution"}
DISTANCE_LINESTYLE = {"ID": "-", "SHIFT": (0, (5, 2)), "OOD": (0, (1.2, 1.6))}

SINGLE_COLUMN_IN = 3.5    # 89 mm
DOUBLE_COLUMN_IN = 7.2    # 183 mm
LINE_PT = 1.25
MARKER_PT = 5.5
EDGE_PT = 0.7


ONE_HALF_COLUMN_IN = 5.35  # 136 mm
SEED_LINE = "#a8a8a8"     # lines of the random seeds: one step darker than their markers


def color(model: str) -> str:
    return MODELS[model]["color"]


def line_color(model: str) -> str:
    """Colour for a line of this model; the light grey of the random seeds needs a darker step."""
    return SEED_LINE if model == "random" else MODELS[model]["color"]


def rc_params() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"],
        "text.color": INK["ink"], "axes.labelcolor": INK["ink"], "axes.titlecolor": INK["ink"],
        "xtick.color": INK["frame"], "ytick.color": INK["frame"],
        "xtick.labelcolor": INK["ink2"], "ytick.labelcolor": INK["ink2"],
        # closed frame: all four spines, ticks only bottom/left, outward
        "axes.edgecolor": INK["frame"], "axes.linewidth": 0.6,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.top": False, "ytick.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "grid.color": INK["grid"], "grid.linewidth": 0.4, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": LINE_PT, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": MARKER_PT,
        "legend.frameon": False, "legend.handlelength": 1.6,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    }


@contextmanager
def style():
    import matplotlib as mpl

    with mpl.rc_context(rc_params()):
        yield INK
