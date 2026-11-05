from pathlib import Path
import csv

import numpy as np


# Configuration
METRICS = (
    ("euclidean", "Euclidean"),
    ("dce", "DCE"),
    ("btn", "BTN"),
    ("stn", "STN"),
    ("long_lat", "min(BTN;STN)"),
    ("ttc", "TTC"),
    ("time_eggert", "Time Eggert"),
    ("rss", "RSS"),
    ("eggert", "Eggert"),
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIRECTORY = SCRIPT_DIR.parent / "results" / "brt"

OUTPUT_FILE = SCRIPT_DIR / "statistic_tab.csv"
GRADIENT_OUTPUT_FILE = SCRIPT_DIR / "gradient_tab.csv"

REFERENCE_METRIC = "euclidean"
REFERENCE_THRESHOLD = 0.5
SCALE_PERCENTILE = 97.0
GRADIENT_PERCENTILE = 95.0

DECIMALS = 2
GRADIENT_DECIMALS = 6

STATE_NAMES = (
    "x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E",
)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    if not METRICS:
        raise ValueError("METRICS must not be empty.")

    if not 0 < SCALE_PERCENTILE <= 100:
        raise ValueError("SCALE_PERCENTILE must be in (0, 100].")

    if not 0 <= GRADIENT_PERCENTILE <= 100:
        raise ValueError("GRADIENT_PERCENTILE must be in [0, 100].")

    if not np.isfinite(REFERENCE_THRESHOLD) or REFERENCE_THRESHOLD <= 0:
        raise ValueError("REFERENCE_THRESHOLD must be positive and finite.")

    reference_path = DATA_DIRECTORY / f"{REFERENCE_METRIC}.npz"

    with np.load(reference_path, allow_pickle=False) as data:
        reference_brt = data["BRT"]
        reference_axes = tuple(data[name] for name in STATE_NAMES)
        reference_time = float(data["target_time"])

    if not np.isfinite(reference_brt).all():
        raise ValueError("Reference BRT contains non-finite values.")

    if not np.isfinite(reference_time):
        raise ValueError("Invalid reference target time.")

    reference_scale = float(
        np.percentile(reference_brt, SCALE_PERCENTILE)
    )

    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Reference BRT percentile must be positive.")

    threshold_fraction = REFERENCE_THRESHOLD / reference_scale
    del reference_brt

    value_rows = []
    gradient_rows = []

    print(f"BRT target time: {reference_time:g} s")
    print(f"Scale: percentile {SCALE_PERCENTILE:g} of BRT")
    print(
        f"Reference: {REFERENCE_METRIC}, "
        f"threshold = {REFERENCE_THRESHOLD:g}"
    )
    print(
        f"Common threshold fraction: "
        f"{threshold_fraction:.6f} "
        f"({100 * threshold_fraction:.3f}%)\n"
    )

    for filename, label in METRICS:
        path = DATA_DIRECTORY / f"{filename}.npz"

        with np.load(path, allow_pickle=False) as data:
            v0 = data["V0"]
            brt = data["BRT"]
            axes = tuple(data[name] for name in STATE_NAMES)
            target_time = float(data["target_time"])

            expected_shape = tuple(len(axis) for axis in axes)

            if v0.shape != expected_shape or brt.shape != expected_shape:
                raise ValueError(f"{path.name}: incompatible array shapes.")

            if not np.isfinite(v0).all() or not np.isfinite(brt).all():
                raise ValueError(f"{path.name}: non-finite values detected.")

            if not all(
                np.array_equal(axis, reference)
                for axis, reference in zip(axes, reference_axes)
            ):
                raise ValueError(f"{path.name}: grid differs from reference.")

            if not np.isfinite(target_time) or not np.isclose(
                target_time, reference_time, rtol=0, atol=1e-7
            ):
                raise ValueError(f"{path.name}: inconsistent BRT horizon.")

            # Unsafe grid fractions
            n_v0 = int(np.count_nonzero(v0 <= 0))
            n_brt = int(np.count_nonzero(brt <= 0))

            unsafe_v0 = 100.0 * n_v0 / v0.size
            unsafe_brt = 100.0 * n_brt / brt.size
            ratio = n_brt / n_v0 if n_v0 else None

            value_rows.append((
                label,
                f"{unsafe_v0:.{DECIMALS}f}",
                f"{unsafe_brt:.{DECIMALS}f}",
                f"{ratio:.{DECIMALS}f}" if ratio is not None else "--",
            ))

            # Region: BRT < fraction * percentile(BRT)
            scale = float(np.percentile(brt, SCALE_PERCENTILE))

            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    f"{path.name}: BRT percentile must be positive."
                )

            threshold = threshold_fraction * scale
            selected = brt < threshold
            n_selected = int(np.count_nonzero(selected))

            if n_selected == 0:
                raise ValueError(f"{path.name}: empty analysis region.")

            selected_pct = 100.0 * n_selected / brt.size
            del v0

            # Saved gradient order: (..., x, y, theta, v_H, delta_E, v_E)
            gradients = data["gradients"]

            if gradients.shape != (*expected_shape, 6):
                raise ValueError(
                    f"{path.name}: incompatible gradient shape."
                )

            statistics = []

            for component in (4, 5):
                values = np.asarray(
                    gradients[..., component][selected],
                    dtype=np.float64,
                )

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{path.name}: non-finite gradients "
                        "in the selected region."
                    )

                values = np.abs(values) / scale

                statistics.extend((
                    float(np.mean(values)),
                    float(np.percentile(values, GRADIENT_PERCENTILE)),
                ))

            gradient_rows.append((
                label,
                f"{scale:.{GRADIENT_DECIMALS}f}",
                f"{threshold:.{GRADIENT_DECIMALS}f}",
                f"{selected_pct:.{DECIMALS}f}",
                *(
                    f"{value:.{GRADIENT_DECIMALS}f}"
                    for value in statistics
                ),
            ))

            del gradients, values, selected, brt

        print(
            f"{label:<16} "
            f"Unsafe V0: {unsafe_v0:6.2f}%   "
            f"Unsafe BRT: {unsafe_brt:6.2f}%   "
            f"Selected: {selected_pct:6.2f}%"
        )

    write_csv(
        OUTPUT_FILE,
        ("Metric", "UnsafeVzero", "UnsafeBRT", "Ratio"),
        value_rows,
    )

    write_csv(
        GRADIENT_OUTPUT_FILE,
        (
            "Metric",
            "Scale",
            "Threshold",
            "Selected",
            "MeanPfive",
            "PercentilePfive",
            "MeanPsix",
            "PercentilePsix",
        ),
        gradient_rows,
    )

    print(f"\nCSV saved to: {OUTPUT_FILE}")
    print(f"CSV saved to: {GRADIENT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()