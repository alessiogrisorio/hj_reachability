"""Mosaico 3x3 di V0: righe v_H, colonne v_E.

Dipendenze: numpy, scipy, matplotlib.
Esecuzione: python plot_terminal_slices.py
Percorsi relativi riferiti alla cartella dello script; output nella stessa cartella.
Lo smoothing modifica solo le fette visualizzate, senza modificare il file NPZ.
"""
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

# ---------------- Configurazione ----------------
METRIC = "eggert"  # Nome del file NPZ senza estensione; anche btn, stn, ecc.
DATA_DIRECTORY = Path("../results/brt")  # Se lo script e in scripts/
INPUT_FILE = None  # Percorso esplicito opzionale, prevale su DATA_DIRECTORY e METRIC
FIELD_KEY = "V0"  # Anche V0_long o V0_lat, se presenti nel file
OUTPUT_FILE = "eggert_v0_slices.pdf"  # Estensione libera: .pdf, .png, .svg, ecc.

V_E_VALUES = (2.0, 6.0, 10.0)  # Colonne, da sinistra a destra
V_H_VALUES = (2.0, 6.0, 10.0)  # Righe, dall'alto in basso
THETA_REL = 0.0  # Radianti
DELTA_E = 0.0  # Radianti

X_LIMITS = None  # None: intero dominio; oppure (min, max)
Y_LIMITS = None
REFINEMENT = 6
SMOOTHING_SIGMA = 0.8  # Celle raffinate; 0 disattiva lo smoothing
COLORBAR_TICK_STEP = 1.0
COLORMAP = "RdBu_r"
ZERO_COLOR = "black"
ZERO_LINEWIDTH = 1.0
FIGURE_SIZE = (10.2, 5.5)
DPI = 350
SHOW_FIGURE = True
# ------------------------------------------------

BASE = Path(__file__).resolve().parent
STATE_NAMES = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")


def bracket(axis, value, name, periodic=False):
    if not np.isfinite(value):
        raise ValueError(f"{name} deve essere finito.")
    if periodic:
        value = (value - axis[0]) % (2 * np.pi) + axis[0]
        if value > axis[-1]:
            width = axis[0] + 2 * np.pi - axis[-1]
            return len(axis) - 1, 0, float((value - axis[-1]) / width)
    if value < axis[0] - 1e-7 or value > axis[-1] + 1e-7:
        raise ValueError(f"{name}={value:g} fuori da [{axis[0]:g}, {axis[-1]:g}].")
    value = np.clip(value, axis[0], axis[-1])
    if len(axis) == 1:
        return 0, 0, 0.0
    low = int(np.clip(np.searchsorted(axis, value, side="right") - 1, 0, len(axis) - 2))
    return low, low + 1, float((value - axis[low]) / (axis[low + 1] - axis[low]))


def extract_slice(field, axes, v_h, v_e):
    result = field
    # Ordine decrescente per mantenere invariati gli indici delle dimensioni.
    for dim, value in ((5, v_e), (4, DELTA_E), (3, v_h), (2, THETA_REL)):
        lo, hi, weight = bracket(axes[dim], value, STATE_NAMES[dim], periodic=(dim == 2))
        result = ((1 - weight) * np.take(result, lo, axis=dim)
                  + weight * np.take(result, hi, axis=dim))
    if not np.isfinite(result).all():
        raise ValueError(f"Fetta v_H={v_h:g}, v_E={v_e:g} con NaN o infiniti.")
    return np.asarray(result, dtype=float)


def load_data():
    path = Path(INPUT_FILE).expanduser() if INPUT_FILE else Path(DATA_DIRECTORY).expanduser() / f"{METRIC}.npz"
    if not path.is_absolute():
        path = BASE / path
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}\nImpostare DATA_DIRECTORY oppure INPUT_FILE.")
    with np.load(path, allow_pickle=False) as data:
        axes = [np.asarray(data[name], dtype=float) for name in STATE_NAMES]
        for name, axis in zip(STATE_NAMES, axes):
            if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
                raise ValueError(f"Asse non valido: {name}")
        if "state_names" in data and tuple(data["state_names"].tolist()) != STATE_NAMES:
            raise ValueError("Ordine degli stati incompatibile.")
        if FIELD_KEY not in data:
            raise KeyError(f"Campo {FIELD_KEY} assente. Campi disponibili: {data.files}")
        field = data[FIELD_KEY]
        if field.shape != tuple(len(a) for a in axes):
            raise ValueError(f"Forma del campo incompatibile con gli assi: {field.shape}")
    print(f"Input: {path}\nCampo: {FIELD_KEY}; theta_rel={THETA_REL:g}, delta_E={DELTA_E:g} rad")
    return axes, field


def make_figure():
    if len(V_E_VALUES) != 3 or len(V_H_VALUES) != 3:
        raise ValueError("Specificare tre velocita ego e tre velocita human.")
    if int(REFINEMENT) != REFINEMENT or REFINEMENT < 1 or SMOOTHING_SIGMA < 0:
        raise ValueError("REFINEMENT deve essere intero >= 1; SMOOTHING_SIGMA >= 0.")
    axes, field = load_data()
    x, y = axes[:2]
    if min(len(x), len(y)) < 2:
        raise ValueError("Servono almeno due nodi per ciascun asse spaziale.")
    limits = [X_LIMITS if X_LIMITS is not None else (x[0], x[-1]),
              Y_LIMITS if Y_LIMITS is not None else (y[0], y[-1])]
    for name, axis, (lo, hi) in zip(STATE_NAMES[:2], (x, y), limits):
        if hi <= lo or lo < axis[0] or hi > axis[-1]:
            raise ValueError(f"Limiti non validi per {name}: {(lo, hi)}")
    # Raffinamento e filtro sul dominio completo, poi ritaglio: bordi coerenti.
    xf = np.linspace(x[0], x[-1], (len(x) - 1) * int(REFINEMENT) + 1)
    yf = np.linspace(y[0], y[-1], (len(y) - 1) * int(REFINEMENT) + 1)
    xx, yy = np.meshgrid(xf, yf, indexing="ij")
    points = np.stack((xx, yy), axis=-1)
    xr = np.unique(np.r_[limits[0][0], xf[(xf > limits[0][0]) & (xf < limits[0][1])], limits[0][1]])
    yr = np.unique(np.r_[limits[1][0], yf[(yf > limits[1][0]) & (yf < limits[1][1])], limits[1][1]])
    rx, ry = np.meshgrid(xr, yr, indexing="ij")
    render_points = np.stack((rx, ry), axis=-1)
    slices = []
    for row, v_h in enumerate(V_H_VALUES):
        for column, v_e in enumerate(V_E_VALUES):
            original = extract_slice(field, axes, v_h, v_e)
            fine = RegularGridInterpolator((x, y), original)(points)
            if SMOOTHING_SIGMA:
                fine = gaussian_filter(fine, sigma=SMOOTHING_SIGMA, mode="nearest")
            plotted = RegularGridInterpolator((xf, yf), fine)(render_points)
            slices.append(plotted)
            print(f"Riga {row+1}, colonna {column+1}: v_H={v_h:g}, v_E={v_e:g}, "
                  f"range=[{plotted.min():.6g}, {plotted.max():.6g}]")
    vmin = min(float(s.min()) for s in slices)
    vmax = max(float(s.max()) for s in slices)
    if not vmin < vmax:
        raise ValueError("Le nove fette hanno tutte lo stesso valore: colorbar degenere.")
    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax))
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 13, "axes.labelsize": 14,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, panels = plt.subplots(3, 3, figsize=FIGURE_SIZE, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.84, bottom=0.10, top=0.86, wspace=0.08, hspace=0.10)
    handles = [
        Patch(facecolor="lightcyan", edgecolor="#263746", linewidth=1.4,
              linestyle="--", label="Ego vehicle"),
        Line2D([0], [0], color=ZERO_COLOR, linewidth=ZERO_LINEWIDTH,
               label=r"$V_0=0$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=2, frameon=False, fontsize=13, handlelength=2.5, columnspacing=2.5)
    for index, (ax, values) in enumerate(zip(panels.flat, slices)):
        row, column = divmod(index, 3)
        image = ax.pcolormesh(xr, yr, values.T, shading="gouraud", cmap=COLORMAP,
                              norm=norm, rasterized=True)
        if values.min() < 0 < values.max():
            ax.contour(xr, yr, values.T, levels=[0], colors=ZERO_COLOR,
                       linewidths=ZERO_LINEWIDTH)
        else:
            warnings.warn(f"Nessun attraversamento di zero nella fetta ({row+1}, {column+1}).")
        ego = plt.Rectangle(
            (-4.68 / 2, -2.20 / 2), 4.68, 2.20,
            facecolor="lightcyan", edgecolor="#263746",
            linewidth=1.4, linestyle="--", alpha=1.0, zorder=5,
        )
        ax.add_patch(ego)
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=12, direction="out", length=3)
        if row == 2:
            ax.set_xlabel(r"$x_{\mathrm{rel}}\ [\mathrm{m}]$")
        if column == 0:
            ax.set_ylabel(r"$y_{\mathrm{rel}}\ [\mathrm{m}]$")
        if row == 0:
            ax.annotate(rf"$v_E={V_E_VALUES[column]:g}\,\mathrm{{m/s}}$",
                        xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom")
        if column == 0:
            ax.annotate(rf"$v_H={V_H_VALUES[row]:g}\,\mathrm{{m/s}}$",
                        xy=(0, 0.5), xycoords="axes fraction", xytext=(-76, 0),
                        textcoords="offset points", ha="center", va="center", rotation=90)
    fig.canvas.draw()
    bottom = panels[2, 2].get_position().y0
    top = panels[0, 2].get_position().y1
    cax = fig.add_axes((0.875, bottom, 0.022, top - bottom))
    bar = fig.colorbar(image, cax=cax)
    bar.ax.set_yscale("linear")
    step = COLORBAR_TICK_STEP
    if not np.isfinite(step) or step <= 0:
        raise ValueError("COLORBAR_TICK_STEP deve essere positivo e finito.")
    ticks = np.arange(np.ceil(vmin / step), np.floor(vmax / step) + 1) * step
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{v:g}" for v in ticks])
    bar.set_label(r"$V_0(\mathbf{x}_R)$", labelpad=9)
    bar.ax.tick_params(labelsize=12)
    output = BASE / Path(OUTPUT_FILE).name
    if not output.suffix:
        raise ValueError("OUTPUT_FILE deve includere un'estensione, ad esempio .pdf.")
    fig.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    print(f"Colorbar comune: [{vmin:.6g}, {vmax:.6g}]\nExport: {output}")
    print(f"Smoothing solo grafico: sigma={SMOOTHING_SIGMA:g} celle raffinate.")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return output


if __name__ == "__main__":
    make_figure()
