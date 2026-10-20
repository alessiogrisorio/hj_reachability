"""Due superfici zero 3D e tre confronti XY a v_H fisso.

Dipendenze: numpy scipy matplotlib scikit-image.
Eseguire: python scripts/compare_terminal_values.py
I percorsi relativi sono riferiti allo script. Nessun calcolo BRT viene eseguito.
Interpolazione e smoothing riguardano solo la visualizzazione.
"""
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

# ---------------- Configurazione ----------------
DATA_DIRECTORY = Path("../results/brt")
# file: percorso assoluto o relativo a DATA_DIRECTORY; key: campo nel NPZ.
METRICS = (
    {"file": "euclidean.npz", "label": "Euclidean", "key": "V0"},
    {"file": "long_lat.npz", "label": "min(BTN,STN)", "key": "V0"},
)
OUTPUT_FILE = "euclidean_vs_long_lat_v0.pdf"  # Esportato accanto allo script

V_H = 6.0
DELTA_E = 0.0  # rad
V_E_3D = 6.0
V_E_SLICES = (2.0, 6.0, 10.0)  # Dall'alto in basso
THETA_SLICE = 0.0  # rad, fissato per le tre slice XY
THETA_LIMITS_3D = (-np.pi / 2, np.pi / 2)  # None: intero periodo
X_LIMITS = None  # None: dominio comune alle due metriche
Y_LIMITS = None

COLORMAP = "RdBu_r"
COLOR_POSITIONS = (0.12, 0.88)
SURFACE_ALPHA = 0.35
SURFACE_REFINEMENT = 4
SLICE_REFINEMENT = 6
SMOOTHING_SIGMA = 0.8  # Celle raffinate; 0 disattiva il filtro
CONTOUR_LINEWIDTH = 1.8
CONTOUR_STYLES = ("solid", "dashed")
EGO_LENGTH = 4.68
EGO_WIDTH = 2.20
EGO_FACE = "lightcyan"
EGO_EDGE = "#263746"
ELEVATION = 22
AZIMUTH = -120
FIGURE_SIZE = (11.5, 7.0)
DPI = 350
SHOW_FIGURE = True
# ------------------------------------------------

BASE = Path(__file__).resolve().parent
NAMES = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")


def bracket(axis, value, name, periodic=False):
    if not np.isfinite(value):
        raise ValueError(f"Valore non finito: {name}.")
    if periodic:
        value = (value - axis[0]) % (2 * np.pi) + axis[0]
        if value > axis[-1]:
            return len(axis)-1, 0, (value-axis[-1])/(axis[0]+2*np.pi-axis[-1])
    if value < axis[0]-1e-7 or value > axis[-1]+1e-7:
        raise ValueError(f"{name}={value:g} fuori da [{axis[0]:g}, {axis[-1]:g}].")
    value = np.clip(value, axis[0], axis[-1])
    if len(axis) == 1:
        return 0, 0, 0.0
    lo = int(np.clip(np.searchsorted(axis, value, side="right")-1, 0, len(axis)-2))
    return lo, lo+1, float((value-axis[lo])/(axis[lo+1]-axis[lo]))


def load_metric(config):
    folder = Path(DATA_DIRECTORY).expanduser()
    if not folder.is_absolute():
        folder = BASE / folder
    path = Path(config["file"]).expanduser()
    if not path.is_absolute():
        path = folder / path
    if not path.is_file():
        raise FileNotFoundError(f"File non trovato: {path}")
    with np.load(path, allow_pickle=False) as data:
        axes = [np.asarray(data[n], dtype=float) for n in NAMES]
        for name, axis in zip(NAMES, axes):
            if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or not np.all(np.diff(axis)>0):
                raise ValueError(f"Asse non valido in {path.name}: {name}")
        if "state_names" in data and tuple(data["state_names"].tolist()) != NAMES:
            raise ValueError(f"Ordine degli stati incompatibile in {path.name}.")
        key = config.get("key", "V0")
        if key not in data:
            raise KeyError(f"{path.name}: campo {key} assente. Disponibili: {data.files}")
        field = data[key]
        if field.shape != tuple(len(a) for a in axes):
            raise ValueError(f"Forma incompatibile in {path.name}: {field.shape}")
    print(f"{config['label']}: {path} [{key}]")
    return axes, field


def fixed_slice(axes, field, v_e, xy=False):
    values = field
    fixed = [(5, v_e), (4, DELTA_E), (3, V_H)]
    if xy:
        fixed.append((2, THETA_SLICE))
    for dim, value in fixed:
        lo, hi, w = bracket(axes[dim], value, NAMES[dim], periodic=(dim==2))
        values = ((1-w)*np.take(values, lo, axis=dim)
                  + w*np.take(values, hi, axis=dim))
    if not np.isfinite(values).all():
        raise ValueError("Fetta con NaN o infiniti.")
    return np.asarray(values, dtype=float)


def fine_axis(axis, refinement):
    if len(axis)<2:
        raise ValueError("Almeno due nodi per ogni asse visualizzato.")
    return np.linspace(axis[0], axis[-1], (len(axis)-1)*refinement+1)


def plot_surface(ax, axes, field, color, label, limits):
    x, y, theta = axes[:3]
    values = fixed_slice(axes, field, V_E_3D)
    theta_ext = np.r_[theta-2*np.pi, theta, theta+2*np.pi]
    values_ext = np.concatenate((values, values, values), axis=2)
    xf, yf = [fine_axis(a, SURFACE_REFINEMENT) for a in (x, y)]
    tlo, thi = limits[2]
    nt = max(3, int(np.ceil((thi-tlo)/np.median(np.diff(theta))))*SURFACE_REFINEMENT+1)
    tf = np.linspace(tlo, thi, nt)
    dt = tf[1]-tf[0]
    halo = int(np.ceil(4*SMOOTHING_SIGMA))+1
    tp = tf[0]+np.arange(-halo, len(tf)+halo)*dt
    # Wrapping delle query: anche un periodo completo con halo resta valido.
    tp = (tp-theta[0]) % (2*np.pi)+theta[0]
    query = np.stack(np.meshgrid(xf, yf, tp, indexing="ij"), axis=-1)
    fine = RegularGridInterpolator((x, y, theta_ext), values_ext)(query)
    if SMOOTHING_SIGMA:
        fine = gaussian_filter(fine, sigma=SMOOTHING_SIGMA, mode="nearest")
    fine = fine[:, :, halo:-halo]
    # Ritaglio esatto dopo il filtro, anche con limiti XY personalizzati.
    xr = np.unique(np.r_[limits[0][0], xf[(xf>limits[0][0]) & (xf<limits[0][1])], limits[0][1]])
    yr = np.unique(np.r_[limits[1][0], yf[(yf>limits[1][0]) & (yf<limits[1][1])], limits[1][1]])
    query = np.stack(np.meshgrid(xr, yr, tf, indexing="ij"), axis=-1)
    clipped = RegularGridInterpolator((xf, yf, tf), fine)(query)
    if not clipped.min()<0<clipped.max():
        warnings.warn(f"{label}: nessuna superficie zero nel dominio 3D scelto.")
        return
    # Marching cubes su indici; conversione anche per assi non uniformi ai bordi.
    vertices, faces, _, _ = marching_cubes(clipped, level=0, allow_degenerate=False)
    for dim, axis in enumerate((xr, yr, tf)):
        vertices[:, dim] = np.interp(vertices[:, dim], np.arange(len(axis)), axis)
    mesh = Poly3DCollection(vertices[faces], facecolors=color,
                            linewidths=0, alpha=SURFACE_ALPHA, antialiased=False,
                            rasterized=True, shade=True,
                            lightsource=LightSource(azdeg=310, altdeg=40))
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)
    for dim in (0, 1):
        if any(np.take(clipped, edge, axis=dim).min()<=0<=np.take(clipped, edge, axis=dim).max()
               for edge in (0,-1)):
            warnings.warn(f"{label}: superficie zero tagliata al bordo {NAMES[dim]}.")


def plot_contour(ax, axes, field, color, style, label, v_e, limits):
    x, y = axes[:2]
    values = fixed_slice(axes, field, v_e, xy=True)
    xf, yf = [fine_axis(a, SLICE_REFINEMENT) for a in (x,y)]
    query = np.stack(np.meshgrid(xf, yf, indexing="ij"), axis=-1)
    fine = RegularGridInterpolator((x,y), values)(query)
    if SMOOTHING_SIGMA:
        fine = gaussian_filter(fine, sigma=SMOOTHING_SIGMA, mode="nearest")
    xr = np.unique(np.r_[limits[0][0], xf[(xf>limits[0][0]) & (xf<limits[0][1])], limits[0][1]])
    yr = np.unique(np.r_[limits[1][0], yf[(yf>limits[1][0]) & (yf<limits[1][1])], limits[1][1]])
    query = np.stack(np.meshgrid(xr, yr, indexing="ij"), axis=-1)
    clipped = RegularGridInterpolator((xf,yf), fine)(query)
    if clipped.min()<0<clipped.max():
        ax.contour(xr, yr, clipped.T, levels=[0], colors=[color],
                   linewidths=CONTOUR_LINEWIDTH, linestyles=style)
    else:
        warnings.warn(f"{label}: nessun attraversamento zero nella slice v_E={v_e:g}.")


def make_figure():
    if len(METRICS)!=2 or len(V_E_SLICES)!=3:
        raise ValueError("Specificare due metriche e tre velocita ego.")
    if not 0<SURFACE_ALPHA<=1 or SMOOTHING_SIGMA<0:
        raise ValueError("Richiesti 0 < SURFACE_ALPHA <= 1 e SMOOTHING_SIGMA >= 0.")
    for refinement in (SURFACE_REFINEMENT, SLICE_REFINEMENT):
        if refinement<1 or int(refinement)!=refinement:
            raise ValueError("I raffinamenti devono essere interi positivi.")
    datasets = [load_metric(c) for c in METRICS]
    limits = []
    for dim, requested in enumerate((X_LIMITS,Y_LIMITS)):
        lo=max(a[dim][0] for a,_ in datasets)
        hi=min(a[dim][-1] for a,_ in datasets)
        chosen = requested if requested is not None else (lo,hi)
        if chosen[0]<lo or chosen[1]>hi or chosen[0]>=chosen[1]:
            raise ValueError(f"Limiti {NAMES[dim]} fuori dal dominio comune [{lo},{hi}].")
        limits.append(chosen)
    theta_limits = THETA_LIMITS_3D if THETA_LIMITS_3D is not None else (-np.pi,np.pi)
    if theta_limits[0]>=theta_limits[1] or theta_limits[1]-theta_limits[0]>2*np.pi+1e-7:
        raise ValueError("Specificare un intervallo angolare positivo lungo al massimo 2pi.")
    limits.append(theta_limits)
    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"cm", "font.size":13,
                         "axes.labelsize":14, "pdf.fonttype":42, "ps.fonttype":42})
    colors = [plt.get_cmap(COLORMAP)(p) for p in COLOR_POSITIONS]
    fig = plt.figure(figsize=FIGURE_SIZE)
    grid = fig.add_gridspec(3,2, width_ratios=(1.75,1), left=0.10, right=0.91,
                            bottom=0.12, top=0.88, wspace=0.22, hspace=0.24)
    spatial = fig.add_subplot(grid[:,0], projection="3d")
    for dataset,color,config in zip(datasets, colors, METRICS):
        plot_surface(spatial, *dataset, color, config["label"], limits)
    spatial.set_xlabel(r"$x_{\mathrm{rel}}\ [\mathrm{m}]$", labelpad=9)
    spatial.set_ylabel(r"$y_{\mathrm{rel}}\ [\mathrm{m}]$", labelpad=9)
    spatial.set_zlabel(r"$\theta_{\mathrm{rel}}\ [\mathrm{rad}]$", labelpad=9)
    spatial.set_xlim(limits[0]); spatial.set_ylim(limits[1]); spatial.set_zlim(limits[2])
    ticks=np.arange(-4,5)*np.pi/4
    labels=[r"$-\pi$",r"$-3\pi/4$",r"$-\pi/2$",r"$-\pi/4$","$0$",
            r"$\pi/4$",r"$\pi/2$",r"$3\pi/4$",r"$\pi$"]
    ids=[i for i,t in enumerate(ticks) if theta_limits[0]-1e-7<=t<=theta_limits[1]+1e-7]
    spatial.set_zticks(ticks[ids], [labels[i] for i in ids])
    spatial.view_init(elev=ELEVATION, azim=AZIMUTH)
    spatial.set_box_aspect((1.1,1,1))
    spatial.tick_params(labelsize=11, pad=1)
    for axis in (spatial.xaxis, spatial.yaxis, spatial.zaxis):
        axis.pane.fill=False
        axis._axinfo["grid"]["color"]=(0.78,0.78,0.78,1)
    panels=[]
    for row,v_e in enumerate(V_E_SLICES):
        ax=fig.add_subplot(grid[row,1], sharex=panels[0] if panels else None,
                           sharey=panels[0] if panels else None)
        panels.append(ax)
        for dataset,color,config,style in zip(datasets, colors, METRICS, CONTOUR_STYLES):
            plot_contour(ax, *dataset, color, style, config["label"], v_e, limits)
        ax.add_patch(plt.Rectangle((-EGO_LENGTH/2,-EGO_WIDTH/2), EGO_LENGTH,EGO_WIDTH,
                                   facecolor=EGO_FACE,edgecolor=EGO_EDGE,
                                   linewidth=1.4,linestyle="--",zorder=5))
        ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
        ax.set_aspect("equal",adjustable="box")
        ax.set_ylabel(r"$y_{\mathrm{rel}}\ [\mathrm{m}]$")
        ax.tick_params(labelsize=12, labelbottom=(row==2))
        ax.annotate(rf"$v_E={v_e:g}\,\mathrm{{m/s}}$",xy=(1.03,0.5),xycoords="axes fraction",
                     ha="left",va="center",rotation=90)
        if row==2:
            ax.set_xlabel(r"$x_{\mathrm{rel}}\ [\mathrm{m}]$")
    panels[0].annotate(
        rf"$v_H={V_H:g}\,\mathrm{{m/s}}$",
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=13,
    )
    handles=[Line2D([0],[0],color=c,linewidth=CONTOUR_LINEWIDTH,linestyle=s,
                    label=cfg["label"]+r" ($V_0=0$)")
             for c,s,cfg in zip(colors,CONTOUR_STYLES,METRICS)]
    handles.append(Patch(facecolor=EGO_FACE,edgecolor=EGO_EDGE,linestyle="--",label="Ego vehicle"))
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.99),
               ncol=3,frameon=False,handlelength=2.5,columnspacing=2)
    spatial.text2D(
        0.5, -0.08,
        rf"$v_E={V_E_3D:g}\,\mathrm{{m/s}}"
        rf"\qquad v_H={V_H:g}\,\mathrm{{m/s}}$",
        transform=spatial.transAxes,
        ha="center",
    )
    output=BASE / Path(OUTPUT_FILE).name
    if not output.suffix:
        raise ValueError("OUTPUT_FILE deve includere l'estensione.")
    fig.savefig(output,dpi=DPI,bbox_inches="tight",pad_inches=0.1,
                bbox_extra_artists=fig.get_default_bbox_extra_artists() + [spatial.zaxis.label])
    print(f"v_H={V_H:g}; delta_E={DELTA_E:g} rad; theta slice={THETA_SLICE:g} rad")
    print(f"Export: {output}\nSmoothing solo grafico: {SMOOTHING_SIGMA:g} celle raffinate.")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return output


if __name__ == "__main__":
    make_figure()
