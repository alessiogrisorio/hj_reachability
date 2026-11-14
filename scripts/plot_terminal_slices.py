"""Mosaico V0/BRT: colonne v_E, righe theta_rel, delta_E o v_H.
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
try:
    from .plotting_common import angle_label, format_colorbars
except ImportError:
    from plotting_common import angle_label, format_colorbars
# ---------------- Configurazione ----------------
METRICS = (
    "euclidean",
    "dce",
    "btn",
    "stn",
    "long_lat",
    "ttc",
    "time_eggert",
    "rss",
    "eggert",
)
DATA_DIRECTORY = Path("../results/brt")  # Se lo script e in scripts/
INPUT_FILE = None  # Percorso opzionale, utilizzabile con una sola metrica
FIELD_KEY = "BRT"  # Anche V0_long o V0_lat, se presenti nel file
OUTPUT_DIRECTORY = Path(".")  # Relativa allo script; PDF: <metrica>_<campo>_slices.pdf
SHOW_V0_CONTOUR = True  # Attivo solo quando FIELD_KEY = "BRT"
V0_COLOR = "#287a32"
V0_LINESTYLE = "--"
V0_LINEWIDTH = 1.4
V_E_VALUES = (4.0, 6.0, 10.0)  # Colonne, da sinistra a destra
ROW_PARAMETER = None  # None: default per metrica; oppure theta_rel, delta_E, v_H
ROW_VALUES = None  # None: valori predefiniti del parametro selezionato
V_H_FIXED = 6.0
FONT_SIZE = 16
V_H_VALUES = (4.0, 6.0, 10.0)  # Righe, dall'alto in basso
THETA_REL = 0.0  # Radianti
DELTA_E = 0.0  # Radianti
X_LIMITS = None  # None: intero dominio; oppure (min, max)
Y_LIMITS = None
REFINEMENT = 6
SMOOTHING_SIGMA = 0.8  # Celle raffinate; 0 disattiva lo smoothing
COLORBAR_TICK_STEP = 1.0
MAX_COLORBAR_TICKS = 9
COLORMAP = "RdBu_r"
ZERO_COLOR = "black"
ZERO_LINEWIDTH = 1.0
FIGURE_SIZE = (10.2, 5.5)
DPI = 350
SHOW_FIGURE = False  # True: mostra tutte le figure alla fine del batch
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

def load_data(metric):
    path = Path(INPUT_FILE).expanduser() if INPUT_FILE else Path(DATA_DIRECTORY).expanduser() / f"{metric}.npz"
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
        terminal = None
        target_time = None
        if FIELD_KEY == "BRT":
            if "target_time" not in data:
                raise KeyError("target_time assente nel file BRT.")
            target_time = float(data["target_time"])
            if not np.isfinite(target_time):
                raise ValueError("target_time non finito.")
            if SHOW_V0_CONTOUR:
                if "V0" not in data:
                    raise KeyError("V0 assente: impossibile sovrapporre il terminal set.")
                terminal = data["V0"]
                if terminal.shape != field.shape:
                    raise ValueError("V0 e BRT devono avere la stessa forma.")
    print(f"Input: {path}\nCampo: {FIELD_KEY}; theta_rel={THETA_REL:g}, delta_E={DELTA_E:g} rad")
    return axes, field, terminal, target_time

def make_mosaic(
    metric, key="V0", *, row_parameter=None, row_values=None,
    v_h=6.0, v_e_values=(4.0, 6.0, 10.0), theta_rel=0.0, delta_e=0.0,
    data_directory=None, input_file=None, figure_size=(10.2, 5.5),
    refinement=6, smoothing_sigma=2.0, cmap="RdBu_r", font_size=16,
    show_v0_contour=True, x_limits=None, y_limits=None,
):
    """Restituisce un mosaico; non salva e non modifica i dati o le impostazioni.

    Colonne: v_E. Righe: theta_rel, delta_E oppure v_H.
    I parametri fissi non sono scritti nella figura. Nei BRT il bordo V0=0
    viene estratto con gli stessi parametri di ogni singolo pannello.
    """
    if row_parameter is None:
        row_parameter = ("theta_rel" if metric in ("dce", "btn", "stn") else
                         "delta_E" if metric in ("ttc", "time_eggert") else "v_H")
    defaults = {
        "theta_rel": (0.0, np.pi/6, np.pi/3),
        "delta_E": (-np.pi/12, 0.0, np.pi/12),
        "v_H": tuple(V_H_VALUES),
    }
    if row_parameter not in defaults:
        raise ValueError("row_parameter: theta_rel, delta_E oppure v_H.")
    row_values = tuple(defaults[row_parameter] if row_values is None else row_values)
    v_e_values = tuple(v_e_values)
    if not row_values or not v_e_values:
        raise ValueError("Specificare almeno una riga e una colonna.")
    if refinement < 1 or int(refinement) != refinement or smoothing_sigma < 0:
        raise ValueError("refinement intero >= 1; smoothing_sigma >= 0.")
    folder = Path(DATA_DIRECTORY if data_directory is None else data_directory).expanduser()
    path = Path(input_file).expanduser() if input_file else folder / f"{metric}.npz"
    if not path.is_absolute():
        path = BASE / path
    with np.load(path, allow_pickle=False) as data:
        axes = [np.asarray(data[name], dtype=float) for name in STATE_NAMES]
        shape = tuple(len(axis) for axis in axes)
        for name, axis in zip(STATE_NAMES, axes):
            if axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError(f"Asse non valido: {name}")
        if "state_names" in data and tuple(data["state_names"].tolist()) != STATE_NAMES:
            raise ValueError("Ordine degli stati incompatibile.")
        field = data[key]
        if field.shape != shape:
            raise ValueError("Forma del campo incompatibile con la griglia.")
        initial = data["V0"] if key == "BRT" and show_v0_contour else None
        if initial is not None and initial.shape != shape:
            raise ValueError("Forma V0 incompatibile con la griglia.")
        time = float(data["target_time"]) if key == "BRT" else None
        if time is not None and not np.isfinite(time):
            raise ValueError("target_time non finito.")
    field_label = (rf"V({time:g},\mathbf{{x}}_R)" if key == "BRT" else
                   r"V_0(\mathbf{x}_R)" if key == "V0" else key.replace("_", r"\_"))
    x, y = axes[:2]
    if min(len(x), len(y)) < 2:
        raise ValueError("Almeno due nodi per ciascun asse spaziale.")
    limits = [x_limits or (x[0], x[-1]), y_limits or (y[0], y[-1])]
    for axis, (lo, hi) in zip((x, y), limits):
        if not axis[0] <= lo < hi <= axis[-1]:
            raise ValueError("Limiti XY fuori griglia.")
    xf, yf = [np.linspace(a[0], a[-1], (len(a)-1)*int(refinement)+1) for a in (x, y)]
    points = np.stack(np.meshgrid(xf, yf, indexing="ij"), axis=-1)
    xr, yr = [np.unique(np.r_[lo, a[(a > lo) & (a < hi)], hi])
              for a, (lo, hi) in zip((xf, yf), limits)]
    crop = np.stack(np.meshgrid(xr, yr, indexing="ij"), axis=-1)

    def section(values, parameters):
        for dim, name in ((5, "v_E"), (4, "delta_E"), (3, "v_H"), (2, "theta_rel")):
            lo, hi, weight = bracket(axes[dim], parameters[name], name, periodic=(dim == 2))
            values = (1-weight)*np.take(values, lo, axis=dim) + weight*np.take(values, hi, axis=dim)
        if not np.isfinite(values).all():
            raise ValueError("Fetta con NaN o infiniti.")
        fine = RegularGridInterpolator((x, y), values)(points)
        if smoothing_sigma:
            fine = gaussian_filter(fine, smoothing_sigma, mode="nearest")
        return RegularGridInterpolator((xf, yf), fine)(crop)

    slices, initial_slices = [], []
    for row_value in row_values:
        for ve in v_e_values:
            parameters = dict(v_E=ve, v_H=v_h, theta_rel=theta_rel, delta_E=delta_e)
            parameters[row_parameter] = row_value
            slices.append(section(field, parameters))
            initial_slices.append(section(initial, parameters) if initial is not None else None)
    vmin = min(float(a.min()) for a in slices)
    vmax = max(float(a.max()) for a in slices)
    if not vmin < vmax:
        raise ValueError("Colorbar degenere.")
    norm = (TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax
            else Normalize(vmin=vmin, vmax=vmax))
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": font_size, "axes.labelsize": font_size,
                         "xtick.labelsize": font_size-1, "ytick.labelsize": font_size-1,
                         "legend.fontsize": font_size-1, "pdf.fonttype": 42, "ps.fonttype": 42})
    nr, nc = len(row_values), len(v_e_values)
    fig = plt.figure(figsize=figure_size, layout="constrained")
    grid = fig.add_gridspec(nr+1, nc+2, width_ratios=(0.16, *([1]*nc), 0.045),
                           height_ratios=(0.25, *([1]*nr)))
    header = fig.add_subplot(grid[0, :])
    header.set_axis_off()
    handles = [Patch(facecolor="lightcyan", edgecolor="#263746", linewidth=1.4,
                     linestyle="--", label="Ego vehicle"),
               Line2D([], [], color=ZERO_COLOR, linewidth=ZERO_LINEWIDTH,
                      label=rf"${field_label}=0$")]
    if initial is not None:
        handles.append(Line2D([], [], color=V0_COLOR, linestyle=V0_LINESTYLE,
                              linewidth=V0_LINEWIDTH, label=r"$V_0(\mathbf{x}_R)=0$"))
    header.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                  handlelength=2.5, columnspacing=2.0)
    panels = []
    for row, value in enumerate(row_values):
        label_ax = fig.add_subplot(grid[row+1, 0])
        label_ax.set_axis_off()
        label = (rf"$v_H={value:g}\,\mathrm{{m/s}}$" if row_parameter == "v_H" else
                 rf"$\theta_{{\mathrm{{rel}}}}={angle_label(value)}\,\mathrm{{rad}}$"
                 if row_parameter == "theta_rel" else
                 rf"$\delta_E={angle_label(value)}\,\mathrm{{rad}}$")
        label_ax.text(0.5, 0.5, label, rotation=90, ha="center", va="center")
        for column, ve in enumerate(v_e_values):
            index = row*nc+column
            ax = fig.add_subplot(grid[row+1, column+1],
                                 sharex=panels[0] if panels else None,
                                 sharey=panels[0] if panels else None)
            panels.append(ax)
            values = slices[index]
            image = ax.pcolormesh(xr, yr, values.T, shading="gouraud", cmap=cmap,
                                  norm=norm, rasterized=True)
            for current, color, linewidth, linestyle in (
                (values, ZERO_COLOR, ZERO_LINEWIDTH, "-"),
                (initial_slices[index], V0_COLOR, V0_LINEWIDTH, V0_LINESTYLE),
            ):
                if current is not None and current.min() < 0 < current.max():
                    ax.contour(xr, yr, current.T, levels=[0], colors=color,
                               linewidths=linewidth, linestyles=linestyle, zorder=4)
            ax.add_patch(plt.Rectangle((-4.68/2, -2.20/2), 4.68, 2.20,
                                       facecolor="lightcyan", edgecolor="#263746",
                                       linewidth=1.4, linestyle="--", zorder=5))
            ax.set(xlim=limits[0], ylim=limits[1])
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(direction="out", length=3, labelleft=column == 0, labelbottom=row == nr-1)
            if row == nr-1:
                ax.set_xlabel(r"$x_{\mathrm{rel}}\ [\mathrm{m}]$")
            if column == 0:
                ax.set_ylabel(r"$y_{\mathrm{rel}}\ [\mathrm{m}]$")
            if row == 0:
                ax.set_title(rf"$v_E={ve:g}\,\mathrm{{m/s}}$", fontsize=font_size, pad=8)
    cax = fig.add_subplot(grid[1:, -1])
    bar = fig.colorbar(image, cax=cax)
    ticks = (np.r_[np.linspace(vmin, 0, 5), np.linspace(0, vmax, 5)[1:]]
             if vmin < 0 < vmax else np.linspace(vmin, vmax, 7))
    bar.set_ticks(ticks, labels=[f"{t:.2f}" for t in ticks])
    bar.set_label(rf"${field_label}$", labelpad=9)
    format_colorbars(fig)
    return fig


def make_figure(metric):
    fig = make_mosaic(
        metric, FIELD_KEY, row_parameter=ROW_PARAMETER, row_values=ROW_VALUES,
        v_h=V_H_FIXED, v_e_values=V_E_VALUES, theta_rel=THETA_REL, delta_e=DELTA_E,
        input_file=INPUT_FILE, figure_size=FIGURE_SIZE, refinement=REFINEMENT,
        smoothing_sigma=SMOOTHING_SIGMA, cmap=COLORMAP, font_size=FONT_SIZE,
        show_v0_contour=SHOW_V0_CONTOUR, x_limits=X_LIMITS, y_limits=Y_LIMITS,
    )
    folder = Path(OUTPUT_DIRECTORY).expanduser()
    if not folder.is_absolute():
        folder = BASE / folder
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{metric}_{FIELD_KEY.lower()}_slices.pdf"
    fig.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    if not SHOW_FIGURE:
        plt.close(fig)
    print(f"Export: {output}")
    return output


def main():
    if isinstance(METRICS, str) or not METRICS:
        raise ValueError("METRICS deve essere una sequenza non vuota.")
    if INPUT_FILE is not None and len(METRICS) != 1:
        raise ValueError("INPUT_FILE richiede una sola metrica.")
    outputs = [make_figure(metric) for metric in METRICS]
    if SHOW_FIGURE:
        plt.show()
    return outputs


if __name__ == "__main__":
    main()