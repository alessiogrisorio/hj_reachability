"""Scatter e superficie zero di V0 o BRT, per una o piu metriche.
Dipendenze: numpy, scipy, matplotlib >= 3.7, scikit-image.
Eseguire: python plot_terminal_value.py
I percorsi relativi sono riferiti alla cartella dello script.
"""
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

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
KEY = "BRT"  # "V0" oppure "BRT"
DATA_DIRECTORY = Path("../results/brt")  # Assoluta oppure relativa allo script
INPUT_FILE = None  # Percorso opzionale, utilizzabile con una sola metrica
OUTPUT_DIRECTORY = Path(".")  # Relativa allo script; PDF: <metrica>_<key>.pdf
V_E = 6.0
V_H = 6.0
DELTA_E = 0.0  # Radianti
THETA_LIMITS = (-np.pi / 2, np.pi / 2)  # None: intero periodo
X_LIMITS = None  # None: dominio completo; oppure (min, max)
Y_LIMITS = None
SURFACE_REFINEMENT = 4
SURFACE_SMOOTHING = 1.2  # Sigma in celle RAFFINATE; 0 disattiva lo smoothing
SCATTER_STRIDE = (1, 1, 2)
SCATTER_SIZE = 5.0
COLOR_LIMITS = None  # Es.: (-3, 10); None: estremi della fetta
COLORMAP = "jet"
SURFACE_COLOR = "#990000"
ELEVATION = 22
AZIMUTH = -120
FIGURE_SIZE = (10.8, 4.7)
DPI = 350
SHOW_FIGURE = False  # True: mostra tutte le figure alla fine del batch
# ------------------------------------------------
BASE = Path(__file__).resolve().parent
STATE_NAMES = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")

def bracket(axis, value, name):
    if not np.isfinite(value) or value < axis[0] - 1e-7 or value > axis[-1] + 1e-7:
        raise ValueError(f"{name}={value} fuori dal dominio [{axis[0]}, {axis[-1]}].")
    value = np.clip(value, axis[0], axis[-1])
    if len(axis) == 1:
        return 0, 0, 0.0
    low = int(np.clip(np.searchsorted(axis, value, side="right") - 1, 0, len(axis) - 2))
    weight = (value - axis[low]) / (axis[low + 1] - axis[low])
    return low, low + 1, float(weight)

def load_slice(metric):
    key = KEY
    path = Path(INPUT_FILE).expanduser() if INPUT_FILE else Path(DATA_DIRECTORY).expanduser() / f"{metric}.npz"
    if not path.is_absolute():
        path = BASE / path
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}\nImpostare DATA_DIRECTORY oppure INPUT_FILE.")
    with np.load(path, allow_pickle=False) as data:
        axes = [np.asarray(data[name], dtype=float) for name in STATE_NAMES]
        if "state_names" in data and tuple(data["state_names"].tolist()) != STATE_NAMES:
            raise ValueError("Ordine degli stati incompatibile.")
        for name, axis in zip(STATE_NAMES, axes):
            if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0):
                raise ValueError(f"Asse non valido: {name}")
        if key not in data:
            raise KeyError(f"{path.name} non contiene {key}. Chiavi: {data.files}")
        field = data[key]
        if field.shape != tuple(len(a) for a in axes):
            raise ValueError(f"Forma inattesa per {key}: {field.shape}")
        if key == "BRT":
            if "target_time" not in data:
                raise KeyError(f"{path.name}: target_time assente.")
            target_time = float(data["target_time"])
            if not np.isfinite(target_time):
                raise ValueError(f"{path.name}: target_time non finito.")
            label = rf"V({target_time:g},\mathbf{{x}}_R)"
        elif key == "V0":
            label = r"V_0(\mathbf{x}_R)"
        else:
            label = key.replace("_", r"\_")
        # Interpolazione solo delle tre coordinate fissate, senza mesh 6D.
        for dimension, value in reversed(list(zip((3, 4, 5), (V_H, DELTA_E, V_E)))):
            lo, hi, w = bracket(axes[dimension], value, STATE_NAMES[dimension])
            field = ((1 - w) * np.take(field, lo, axis=dimension)
                     + w * np.take(field, hi, axis=dimension))
    if not np.isfinite(field).all():
        raise ValueError("La fetta contiene NaN o infiniti.")
    print(f"Input: {path}\nCampo: {key}; v_E={V_E:g}, v_H={V_H:g}, delta_E={DELTA_E:g} rad")
    return axes[:3], np.asarray(field, dtype=float), label

def prepare_domain(axes, field):
    x, y, theta = axes
    if min(len(x), len(y), len(theta)) < 2:
        raise ValueError("Servono almeno due nodi per ogni asse visualizzato.")
    # Estensione periodica prima di interpolare o smussare: evita la cucitura a +/-pi.
    period = 2 * np.pi
    theta_ext = np.concatenate((theta - period, theta, theta + period))
    field_ext = np.concatenate((field, field, field), axis=2)
    limits = [X_LIMITS or (x[0], x[-1]), Y_LIMITS or (y[0], y[-1]),
              THETA_LIMITS or (theta[0], theta[0] + period)]
    for axis, (lo, hi), name in zip((x, y, theta_ext), limits, STATE_NAMES):
        if hi <= lo or lo < axis[0] - 1e-7 or hi > axis[-1] + 1e-7:
            raise ValueError(f"Limiti non validi per {name}: {(lo, hi)}")
    return (x, y, theta_ext), field_ext, limits

def configure_axis(ax, limits):
    ax.set_xlabel(r"$x_{\mathrm{rel}}\ [\mathrm{m}]$", labelpad=7)
    ax.set_ylabel(r"$y_{\mathrm{rel}}\ [\mathrm{m}]$", labelpad=7)
    ax.set_zlabel(r"$\theta_{\mathrm{rel}}\ [\mathrm{rad}]$", labelpad=7)
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1]); ax.set_zlim(limits[2])
    ticks = np.arange(-4, 5) * np.pi / 4
    labels = [r"$-\pi$", r"$-3\pi/4$", r"$-\pi/2$", r"$-\pi/4$", "$0$",
              r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
    visible = [i for i, t in enumerate(ticks) if limits[2][0] - 1e-6 <= t <= limits[2][1] + 1e-6]
    ax.set_zticks(ticks[visible], [labels[i] for i in visible])
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_box_aspect((1.1, 1, 1))
    ax.tick_params(labelsize=9, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
        axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 1)
        axis._axinfo["grid"]["linewidth"] = 0.5
    ax.figure.canvas.draw()
    angle = ax.zaxis.label.get_rotation()
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation((angle + 180) % 360)

def make_figure(metric):
    axes, field, label = load_slice(metric)
    ext_axes, ext_field, limits = prepare_domain(axes, field)
    interpolate = RegularGridInterpolator(ext_axes, ext_field, bounds_error=True)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=FIGURE_SIZE)
    fig.subplots_adjust(left=0.03, right=0.96, bottom=0.12, top=0.98, wspace=0.22)
    left = fig.add_subplot(121, projection="3d")
    right = fig.add_subplot(122, projection="3d")
    for ax in (left, right):
        configure_axis(ax, limits)
    scatter_axes = []
    for axis, (lo, hi), stride in zip(axes, limits, SCATTER_STRIDE):
        if stride < 1:
            raise ValueError("SCATTER_STRIDE deve contenere interi positivi.")
        selected = axis[(axis >= lo - 1e-7) & (axis <= hi + 1e-7)][::stride]
        if len(selected) == 0:
            raise ValueError("Nessun nodo scatter nei limiti scelti.")
        scatter_axes.append(selected)
    points = np.stack(np.meshgrid(*scatter_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    values = interpolate(points)
    vmin, vmax = COLOR_LIMITS or (float(field.min()), float(field.max()))
    if vmax <= vmin:
        raise ValueError("Colorbar degenere: scegliere una fetta non costante.")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(COLORMAP)
    left.scatter(*points.T, s=SCATTER_SIZE, facecolors="none",
                 edgecolors=cmap(norm(values)), linewidths=0.35,
                 depthshade=False, rasterized=True)
    colorbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=left,
                            shrink=0.68, pad=0.08, fraction=0.035)
    colorbar.set_label(rf"${label}$")
    colorbar.ax.tick_params(labelsize=8)
    refinement = int(SURFACE_REFINEMENT)
    if refinement < 1 or SURFACE_SMOOTHING < 0:
        raise ValueError("Raffinamento >= 1 e smoothing >= 0 richiesti.")
    fine_axes = [np.linspace(lo, hi, max(3, int(np.ceil((hi-lo) / np.median(np.diff(a)))) * refinement + 1))
                 for a, (lo, hi) in zip(axes, limits)]
    spacing = tuple(a[1] - a[0] for a in fine_axes)
    # Halo angolare per applicare il filtro anche agli estremi senza artefatti.
    halo = int(np.ceil(4 * SURFACE_SMOOTHING)) + 1
    padded_theta = fine_axes[2][0] + np.arange(-halo, len(fine_axes[2]) + halo) * spacing[2]
    grid = np.stack(np.meshgrid(fine_axes[0], fine_axes[1], padded_theta, indexing="ij"), axis=-1)
    fine = interpolate(grid)
    if SURFACE_SMOOTHING:
        fine = gaussian_filter(fine, sigma=SURFACE_SMOOTHING, mode="nearest")
    fine = fine[:, :, halo:-halo]
    if fine.min() < 0 < fine.max():
        vertices, faces, _, _ = marching_cubes(fine, level=0, spacing=spacing, allow_degenerate=False)
        vertices += np.array([a[0] for a in fine_axes])
        mesh = Poly3DCollection(vertices[faces], facecolors=SURFACE_COLOR,
                                edgecolor="none", linewidth=0, antialiased=False, alpha=1.0, rasterized=True,
                                shade=True, lightsource=colors.LightSource(azdeg=310, altdeg=40))
        right.add_collection3d(mesh)
    else:
        warnings.warn(f"{metric}: nessuna superficie zero nel dominio visualizzato.")
        right.text2D(0.5, 0.5, "No zero level set in the displayed domain",
                     transform=right.transAxes, ha="center", va="center", fontsize=9)
    for dimension, name in enumerate(STATE_NAMES[:2]):
        for edge in (0, -1):
            slab = np.take(fine, edge, axis=dimension)
            if slab.min() <= 0 <= slab.max():
                warnings.warn(f"La superficie zero raggiunge il bordo {name}. Valutare un dominio piu ampio.")
                break
    folder = Path(OUTPUT_DIRECTORY).expanduser()
    if not folder.is_absolute():
        folder = BASE / folder
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{metric}_{KEY.lower()}.pdf"
    fig.savefig(destination, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    print(f"Export: {destination}\nSmoothing: sigma={SURFACE_SMOOTHING:g} celle raffinate (solo grafica).")
    if not SHOW_FIGURE:
        plt.close(fig)
    return destination



def main():
    if isinstance(METRICS, str) or not METRICS:
        raise ValueError('METRICS deve essere una lista o tupla, es.: ("euclidean",).')
    if len(set(METRICS)) != len(METRICS):
        raise ValueError("METRICS contiene nomi duplicati.")
    if INPUT_FILE is not None and len(METRICS) != 1:
        raise ValueError("INPUT_FILE puo essere usato solo con una singola metrica.")
    if int(SURFACE_REFINEMENT) != SURFACE_REFINEMENT or SURFACE_REFINEMENT < 1:
        raise ValueError("SURFACE_REFINEMENT deve essere un intero >= 1.")
    if not np.isfinite(SURFACE_SMOOTHING) or SURFACE_SMOOTHING < 0:
        raise ValueError("SURFACE_SMOOTHING deve essere finito e >= 0.")
    if len(SCATTER_STRIDE) != 3 or any(int(s) != s or s < 1 for s in SCATTER_STRIDE):
        raise ValueError("SCATTER_STRIDE richiede tre interi >= 1.")
    outputs = []
    for index, metric in enumerate(METRICS, start=1):
        print(f"\n[{index}/{len(METRICS)}] {metric} - {KEY}", flush=True)
        outputs.append(make_figure(metric))
    print(f"\nGenerati {len(outputs)} PDF.")
    if SHOW_FIGURE:
        plt.show()
    return outputs



if __name__ == "__main__":
    main()
