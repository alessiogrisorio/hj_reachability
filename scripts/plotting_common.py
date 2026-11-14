"""Formattazione condivisa dei grafici della tesi; nessun calcolo HJ."""
from fractions import Fraction

import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.layout_engine import PlaceHolderLayoutEngine

METRIC_LABELS = {
    "euclidean": "Euclidean", "dce": "DCE", "btn": "BTN", "stn": "STN",
    "long_lat": "min(BTN, STN)", "ttc": "TTC", "time_eggert": "TBR",
    "rss": "RSS", "eggert": "PRM", "mixed": "Mixed",
}


def angle_label(value):
    ratio = Fraction(float(value / np.pi)).limit_denominator(60)
    if not np.isclose(float(ratio) * np.pi, value, rtol=0, atol=1e-9):
        return f"{value:.3g}"
    numerator, denominator = ratio.numerator, ratio.denominator
    if numerator == 0:
        return "0"
    sign = "-" if numerator < 0 else ""
    top = r"\pi" if abs(numerator) == 1 else rf"{abs(numerator)}\pi"
    return sign + top if denominator == 1 else rf"{sign}\frac{{{top}}}{{{denominator}}}"


def orient_3d_labels(ax):
    """Inverte una sola volta la label Z rispetto all'orientamento automatico."""
    if getattr(ax, "name", None) != "3d" or getattr(ax, "_thesis_z_oriented", False):
        return
    ax.zaxis.set_rotate_label(True)
    ax.figure.canvas.draw()
    rotation = ax.zaxis.label.get_rotation()
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation((rotation + 180) % 360)
    ax._thesis_z_oriented = True


def format_colorbars(fig):
    """Mostra sempre due cifre decimali, anche dopo set_ticklabels locali."""
    for ax in fig.axes:
        colorbar = getattr(ax, "_colorbar", None)
        if colorbar is not None:
            colorbar.formatter = FormatStrFormatter("%.2f")
            colorbar.update_ticks()
    return fig


def finish_figure(fig):
    """Uniforma tutte le colorbar e le label Z dei grafici 3D.

    Richiamare prima di show per l'anteprima; export_figure lo richiama anche
    prima del salvataggio. Non cambia vista, dati, dimensioni o testi fissi.
    """
    # Un layout manuale deve restare manuale anche dopo gli export successivi.
    # Ripristinare None durante savefig riattiverebbe il default constrained.
    if fig.get_layout_engine() is None:
        fig.set_layout_engine(PlaceHolderLayoutEngine(
            adjust_compatible=True, colorbar_gridspec=False,
        ))
    format_colorbars(fig)
    spatial = [ax for ax in fig.axes if getattr(ax, "name", None) == "3d"]
    if not spatial:
        return fig
    for ax in spatial:
        orient_3d_labels(ax)
    for ax in fig.axes:
        colorbar = getattr(ax, "_colorbar", None)
        if colorbar is not None and colorbar.orientation == "vertical":
            ax.yaxis.label.set_rotation(270)
            ax.yaxis.label.set_verticalalignment("top")
            ax.yaxis.label.set_horizontalalignment("center")
            ax.yaxis.labelpad = 10
    return fig