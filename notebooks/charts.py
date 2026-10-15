"""How the essay's and the appendix's figures look: the colours, the type, and the few
helpers every figure draws its title, labels and ticks with. Importing it sets the style."""

import sys
from datetime import date
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib import font_manager

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, PALE = "#e1e0d9", "#c3c2b7", "#e1e0d9"
# One colour per venue, in every figure. A forecast takes the colour of the venue whose
# book it reads, and the controls are grey.
COINBASE, KRAKEN = "#2a78d6", "#eb6834"

# IBM Plex Sans ships beside this file under the SIL Open Font License (fonts/OFL.txt), so
# the figures come out the same on every machine.
for font in sorted((Path(__file__).parent / "fonts").glob("*.ttf")):
    font_manager.fontManager.addfont(str(font))

mpl.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 100,
    # No left spine and no y ticks: the gridlines carry the scale.
    "axes.spines.left": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 10.5, "axes.titleweight": "medium", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.titlepad": 8,
    "axes.labelsize": 9.5, "axes.labelcolor": MUTED,
    "xtick.color": AXIS, "xtick.labelcolor": INK2, "xtick.labelsize": 9, "xtick.major.size": 3,
    "ytick.color": AXIS, "ytick.labelcolor": INK2, "ytick.labelsize": 9, "ytick.major.size": 0,
    "ytick.major.pad": 2,
    "axes.unicode_minus": True, "legend.frameon": False, "legend.fontsize": 9,
    "lines.solid_capstyle": "round",
})

# Twice the pixels, shown at the same size, so lines and type stay sharp on dense screens.
if (ipython := sys.modules.get("IPython")) and (shell := ipython.get_ipython()):
    shell.run_line_magic("config", "InlineBackend.figure_formats = ['retina']")

# Behind a label that sits on gridlines, so the lines stop short of the text.
LABEL_BOX = {"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}


def headline(fig, title, subtitle=None, top=1.0, **margins):
    """The figure's finding, and under it what is plotted, flush with its left edge. `top`
    inches are kept above the axes for both, and `margins` go to subplots_adjust."""
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.1 / h, title, ha="left", va="top", fontsize=13.5,
             fontweight="semibold", color=INK)
    if subtitle:
        # DejaVu, matplotlib's own, draws what Plex lacks, such as the trade markers ▲ and ▼.
        fig.text(0.012, 1 - 0.4 / h, subtitle, ha="left", va="top", fontsize=10, color=INK2,
                 fontfamily=["IBM Plex Sans", "DejaVu Sans"])
    fig.subplots_adjust(top=1 - top / h, **margins)


def name_series(ax, x, y, text, colour, **kw):
    """A series named in its own colour where it ends, in place of a legend."""
    kw = {"va": "center", "fontsize": 9.5, "fontweight": "semibold", **kw}
    return ax.text(x, y, text, color=colour, **kw)


def categories(ax, labels, colours):
    """Names down the y axis, each in its series' colour, for a chart whose rows are
    categories and whose values run along x."""
    ax.set_yticks(range(len(labels)), labels, fontsize=9.5, fontweight="semibold")
    for label, colour in zip(ax.get_yticklabels(), colours):
        label.set_color(colour)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x")


def note(ax, x, y, text, **kw):
    """A quieter label, for a reference line or one point."""
    kw = {"fontsize": 9, "color": INK2, **kw}
    return ax.text(x, y, text, **kw)


def minus(text):
    """Hyphens to true minus signs, for numbers set as text."""
    return text.replace("-", "\N{MINUS SIGN}")


def signed(unit="", digits=0, prefix=""):
    """Tick labels with a sign, a true minus and the unit, and zero bare: +3 bp, 0, −2 bp,
    or with `prefix="$"`, +$40."""
    space = " " if unit[:1].isalnum() else ""

    def fmt(v, _):
        if abs(v) < 1e-12:
            return "0"
        return minus(f"{'-' if v < 0 else '+'}{prefix}{abs(v):.{digits}f}{space}{unit}")
    return fmt


def apart(ys, gap):
    """Heights for labels at series ending at `ys`, pushed at least `gap` apart and kept
    centred on where the series end."""
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    out = ys[order].copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    out += (ys[order] - out).mean()
    placed = np.empty_like(out)
    placed[order] = out
    return placed


def day_labels(dates):
    """Day labels with the month only where it changes: Jul 5, 6, 10."""
    out, month = [], None
    for d in map(date.fromisoformat, dates):
        out.append(f"{d:%b} {d.day}" if d.month != month else f"{d.day}")
        month = d.month
    return out
