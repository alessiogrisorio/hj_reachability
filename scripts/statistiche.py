from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


# File NPZ da analizzare (nomi senza estensione).
METRICS = [
    "euclidean",
    "dce",
    "ttc",
    "eggert",
    "time_eggert",
    "rss",
    "sff",
]

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ_DIR = SCRIPT_DIR.parent / "results" / "brt"
OUTPUT_PATH = SCRIPT_DIR / "statistiche.xlsx"

STATE_NAMES = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")
VALUE_ROWS = (
    "V0 min",
    "V0 max",
    "BRT min",
    "BRT max",
    "V0 > 0 [%]",
    "V0 <= 0 [%]",
    "BRT > 0 [%]",
    "BRT <= 0 [%]",
    "V0 > 0 -> BRT <= 0 [%]",
    "Delta V media",
    "Delta V deviazione standard",
)
GRADIENT_ROWS = tuple(
    label
    for name in STATE_NAMES
    for label in (f"dBRT/d{name} min", f"dBRT/d{name} max")
)


def finite_limits(values, metric, description):
    finite = np.isfinite(values)
    n_invalid = values.size - np.count_nonzero(finite)
    if n_invalid:
        print(f"  Attenzione: {description} contiene {n_invalid} valori non finiti ({metric}).")
    if n_invalid == values.size:
        return None, None
    minimum = float(np.min(values, where=finite, initial=np.inf))
    maximum = float(np.max(values, where=finite, initial=-np.inf))
    return minimum, maximum


def value_statistics(v0, brt, metric):
    if v0.shape != brt.shape or v0.ndim != 6:
        raise ValueError(f"{metric}: V0 e BRT devono avere la stessa forma 6D.")

    n_total = v0.size
    v0_min, v0_max = finite_limits(v0, metric, "V0")
    brt_min, brt_max = finite_limits(brt, metric, "BRT")

    finite_v0 = np.isfinite(v0)
    finite_brt = np.isfinite(brt)
    v0_positive = finite_v0 & (v0 > 0)
    brt_nonpositive = finite_brt & (brt <= 0)

    pct = lambda n: 100.0 * n / n_total
    pct_v0_positive = pct(np.count_nonzero(v0_positive))
    pct_v0_nonpositive = pct(np.count_nonzero(finite_v0 & (v0 <= 0)))
    pct_brt_positive = pct(np.count_nonzero(finite_brt & (brt > 0)))
    pct_brt_nonpositive = pct(np.count_nonzero(brt_nonpositive))
    pct_transition = pct(np.count_nonzero(v0_positive & brt_nonpositive))

    delta = brt - v0
    finite_delta = finite_v0 & finite_brt & np.isfinite(delta)
    if np.any(finite_delta):
        valid_delta = delta[finite_delta]
        mean_delta = float(np.mean(valid_delta, dtype=np.float64))
        std_delta = float(np.std(valid_delta, dtype=np.float64, ddof=0))
    else:
        mean_delta = None
        std_delta = None

    return (
        v0_min, v0_max, brt_min, brt_max,
        pct_v0_positive, pct_v0_nonpositive,
        pct_brt_positive, pct_brt_nonpositive,
        pct_transition, mean_delta, std_delta,
    )


def gradient_statistics(gradients, expected_shape, metric):
    if gradients.shape != (*expected_shape, 6):
        raise ValueError(
            f"{metric}: gradients.shape={gradients.shape}; "
            f"atteso {(*expected_shape, 6)}."
        )
    statistics = []
    for i, name in enumerate(STATE_NAMES):
        statistics.extend(finite_limits(gradients[..., i], metric, f"dBRT/d{name}"))
    return tuple(statistics)


def write_sheet(sheet, labels, data):
    sheet.append(["Statistica", *METRICS])
    for i, label in enumerate(labels):
        sheet.append([label, *(data[metric][i] for metric in METRICS)])

    sheet.freeze_panes = "B2"
    sheet.column_dimensions["A"].width = 37
    sheet.row_dimensions[1].height = 24

    for col in range(1, len(METRICS) + 2):
        cell = sheet.cell(row=1, column=col)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        if col > 1:
            sheet.column_dimensions[get_column_letter(col)].width = 19

    for row in range(2, len(labels) + 2):
        sheet.cell(row=row, column=1).font = Font(bold=True)
        is_percent = labels[row - 2].endswith("[%]")
        for col in range(2, len(METRICS) + 2):
            cell = sheet.cell(row=row, column=col)
            cell.number_format = '0.00"%"' if is_percent else "0.000000"
            cell.alignment = Alignment(horizontal="right")


def main():
    if not METRICS:
        raise ValueError("METRICS non può essere vuoto.")
    if len(METRICS) != len(set(METRICS)):
        raise ValueError("METRICS contiene nomi duplicati.")

    values_data = {}
    gradients_data = {}

    for metric in METRICS:
        npz_path = NPZ_DIR / f"{metric}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"File non trovato: {npz_path}")

        print(f"Analisi di {npz_path.name}...")
        with np.load(npz_path, allow_pickle=False) as archive:
            v0 = archive["V0"]
            brt = archive["BRT"]
            values_data[metric] = value_statistics(v0, brt, metric)
            shape = v0.shape
            del v0, brt

            gradients = archive["gradients"]
            gradients_data[metric] = gradient_statistics(gradients, shape, metric)
            del gradients

    workbook = Workbook()
    write_sheet(workbook.active, VALUE_ROWS, values_data)
    workbook.active.title = "Value_Function"
    write_sheet(workbook.create_sheet("Gradients"), GRADIENT_ROWS, gradients_data)
    workbook.save(OUTPUT_PATH)
    print(f"\nStatistiche salvate in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
