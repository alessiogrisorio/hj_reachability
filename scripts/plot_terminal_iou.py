"""Matrice IoU degli insiemi unsafe sull'intera griglia 6D.

Dipendenze: numpy matplotlib.
Esecuzione: python scripts/plot_terminal_iou.py
Percorsi relativi riferiti allo script; esportazione nella stessa cartella.
Nessuna interpolazione, normalizzazione o smussatura dei campi.
"""
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------- Configurazione ----------------
DATA_DIRECTORY = Path("../results/brt")
# Aggiungere, rimuovere o riordinare le righe per scegliere il confronto.
# file: relativo a DATA_DIRECTORY oppure assoluto; key opzionale.
METRICS = (
    {"file": "euclidean.npz", "label": "Euclidean"},
    {"file": "dce.npz", "label": "DCE"},
    {"file": "btn.npz", "label": "BTN"},
    {"file": "stn.npz", "label": "STN"},
    {"file": "long_lat.npz", "label": "min(BTN,STN)"},
    {"file": "ttc.npz", "label": "TTC"},
    {"file": "time_eggert.npz", "label": "Time Eggert"},
    {"file": "rss.npz", "label": "RSS"},
    {"file": "eggert.npz", "label": "Eggert"},
)
FIELD_KEY = "V0"  # "V0": terminal set; "BRT": insieme dopo propagazione HJ
OUTPUT_FILE = "terminal_sets_iou.pdf"  # Anche .png, .svg, ecc.
DECIMALS = 2
COLORMAP = "Blues"
FIGURE_SIZE = None  # None: dimensioni automatiche; oppure (9.0, 7.8)
FONT_SIZE = 13
ANNOTATION_SIZE = 12
DPI = 350
SHOW_COLORBAR = True
SHOW_FIGURE = True
# ------------------------------------------------

BASE = Path(__file__).resolve().parent
STATE_NAMES = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")
POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def load_masks():
    if not METRICS:
        raise ValueError("L'elenco METRICS e vuoto.")
    folder = Path(DATA_DIRECTORY).expanduser()
    if not folder.is_absolute():
        folder = BASE / folder
    masks, counts = [], []
    reference_axes = None
    reference_time = None
    for config in METRICS:
        path = Path(config["file"]).expanduser()
        if not path.is_absolute():
            path = folder / path
        if not path.is_file():
            raise FileNotFoundError(f"File non trovato: {path}")
        key = config.get("key", FIELD_KEY)
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"{path.name}: campo {key} assente. Disponibili: {data.files}")
            axes = [np.asarray(data[name]) for name in STATE_NAMES]
            for name, axis in zip(STATE_NAMES, axes):
                if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all() or not np.all(np.diff(axis)>0):
                    raise ValueError(f"{path.name}: asse non valido {name}.")
            if "state_names" in data and tuple(data["state_names"].tolist()) != STATE_NAMES:
                raise ValueError(f"{path.name}: ordine degli stati incompatibile.")
            if reference_axes is None:
                reference_axes = axes
            else:
                for name, current, reference in zip(STATE_NAMES, axes, reference_axes):
                    if not np.array_equal(current, reference):
                        raise ValueError(f"{path.name}: l'asse {name} differisce dalla griglia del primo file.")
            if key == "BRT":
                if "target_time" not in data:
                    raise ValueError(f"{path.name}: target_time assente, confronto BRT non verificabile.")
                time = float(data["target_time"])
                if not np.isfinite(time):
                    raise ValueError(f"{path.name}: target_time non finito.")
                if reference_time is None:
                    reference_time = time
                elif not np.isclose(time, reference_time, rtol=0, atol=1e-7):
                    raise ValueError("I BRT devono essere valutati allo stesso istante.")
            field = data[key]
            if field.shape != tuple(len(a) for a in axes):
                raise ValueError(f"{path.name}: forma del campo incompatibile con gli assi.")
            finite = np.isfinite(field)
            if not finite.all():
                warnings.warn(f"{path.name}: {field.size - np.count_nonzero(finite)} valori non finiti "
                              "esclusi dalla maschera unsafe. Verificare il dataset.")
            unsafe = finite & (field <= 0)
            count = int(np.count_nonzero(unsafe))
            masks.append(np.packbits(unsafe.ravel()))
            counts.append(count)
            print(f"{config['label']}: {key}; unsafe {count:,}/{field.size:,} ({100*count/field.size:.3f}%)")
        del field, finite, unsafe
    return masks, counts


def compute_iou(masks, counts):
    n = len(masks)
    matrix = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            intersection = int(POPCOUNT[np.bitwise_and(masks[i], masks[j])].sum(dtype=np.uint64))
            union = counts[i] + counts[j] - intersection
            matrix[i,j] = matrix[j,i] = intersection / union if union else 1.0
    return matrix


def make_figure():
    if not isinstance(DECIMALS, int) or DECIMALS < 0:
        raise ValueError("DECIMALS deve essere un intero non negativo.")
    masks, counts = load_masks()
    matrix = compute_iou(masks, counts)
    del masks
    labels = [config["label"] for config in METRICS]
    n = len(labels)
    size = FIGURE_SIZE if FIGURE_SIZE is not None else (max(5.2, 0.72*n+2.2), max(4.4, 0.72*n+1.0))
    plt.rcParams.update({"font.family":"serif", "mathtext.fontset":"cm",
                         "font.size":FONT_SIZE, "pdf.fonttype":42, "ps.fonttype":42})
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    cmap = plt.get_cmap(COLORMAP)
    norm = Normalize(vmin=0, vmax=1)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_xticks(np.arange(n), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(n), labels)
    ax.tick_params(axis="both", which="major", length=0, pad=9)
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            rgb = np.array(cmap(norm(matrix[i,j]))[:3])
            linear = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
            luminance = float(np.dot(linear, [0.2126,0.7152,0.0722]))
            # Sceglie il testo con contrasto maggiore rispetto allo sfondo.
            white_contrast = 1.05/(luminance+0.05)
            black_contrast = (luminance+0.05)/0.05
            color = "white" if white_contrast > black_contrast else "#111111"
            ax.text(j, i, f"{matrix[i,j]:.{DECIMALS}f}", ha="center", va="center",
                    fontsize=ANNOTATION_SIZE, color=color,
                    fontweight="bold" if i==j else "normal")
    if SHOW_COLORBAR:
        bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035, shrink=0.9)
        ticks=np.linspace(0,1,6)
        bar.set_ticks(ticks, labels=[f"{v:g}" for v in ticks])
        bar.set_label("IoU", labelpad=9)
        bar.outline.set_visible(False)
    output = BASE / Path(OUTPUT_FILE).name
    if not output.suffix:
        raise ValueError("OUTPUT_FILE deve includere l'estensione, ad esempio .pdf.")
    fig.savefig(output, dpi=DPI, bbox_inches="tight", pad_inches=0.08)
    print("\nMatrice IoU:\n" + np.array2string(matrix, precision=DECIMALS))
    print(f"\nExport: {output}")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)
    return matrix


if __name__ == "__main__":
    make_figure()
