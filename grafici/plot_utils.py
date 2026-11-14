from pathlib import Path

import matplotlib as mpl
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.plotting_common import finish_figure


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def set_style(font_size=14, linewidth=1.8, figsize=(7, 4.5)):
    """Imposta lo stile comune prima di creare le figure."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": mpl.rcParamsDefault["font.serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,

        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,

        "lines.linewidth": linewidth,
        "axes.linewidth": 0.9,

        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,

        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 1.0,
        "legend.edgecolor": "0.8",

        "figure.figsize": figsize,
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.constrained_layout.use": True,

        "savefig.dpi": 350,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def export_figure(
    fig,
    nome,
    cartella="",
    formati=("pdf",),
    dpi=300,
    sovrascrivi=False,
    ritaglia=True,
):
    """Esporta una figura e restituisce i percorsi dei file salvati."""
    if not nome or nome in {".", ".."} or "/" in nome or "\\" in nome:
        raise ValueError(
            "Usa un nome file senza percorso; specifica la cartella separatamente."
        )

    if isinstance(formati, str):
        formati = (formati,)

    formati = tuple(dict.fromkeys(
        formato.lower().lstrip(".") for formato in formati
    ))

    if not formati:
        raise ValueError("Specifica almeno un formato.")

    finish_figure(fig)
    supportati = fig.canvas.get_supported_filetypes()
    for formato in formati:
        if formato not in supportati:
            raise ValueError(f"Formato non supportato: {formato}")

    destinazione = Path(cartella).expanduser()
    if not destinazione.is_absolute():
        destinazione = OUTPUT_DIR / destinazione

    percorsi = [
        destinazione / f"{nome}.{formato}"
        for formato in formati
    ]

    if not sovrascrivi:
        for percorso in percorsi:
            if percorso.exists():
                raise FileExistsError(
                    f"Il file esiste già: {percorso}\n"
                    "Usa sovrascrivi=True per sostituirlo."
                )

    destinazione.mkdir(parents=True, exist_ok=True)

    extra_artists = list(fig.get_default_bbox_extra_artists())
    extra_artists.extend(
        ax.zaxis.label for ax in fig.axes
        if getattr(ax, "name", None) == "3d"
    )
    for formato, percorso in zip(formati, percorsi):
        fig.savefig(
            percorso,
            format=formato,
            dpi=dpi,
            bbox_inches="tight" if ritaglia else None,
            bbox_extra_artists=extra_artists if ritaglia else None,
            pad_inches=0.08,
        )
        print(f"Salvato: {percorso}")

    return percorsi