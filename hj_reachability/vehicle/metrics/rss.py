from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class RSSMetricResult:
    terminal_values: np.ndarray
    parameters: dict


def metricRSS(
    grid,
    *,
    transition_width_cells: float = 3.0,
    dtype=np.float32,
) -> RSSMetricResult:

    if not np.isfinite(transition_width_cells) or transition_width_cells <= 0.0:
        raise ValueError("transition_width_cells must be finite and positive")

    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    data_dir = Path(__file__).resolve().parent / "rss"
    mask_path = data_dir / "rss_unsafe_mask.bin"
    meta_path = data_dir / "rss_unsafe_mask_meta.txt"

    if not mask_path.exists():
        raise FileNotFoundError(f"RSS mask not found: {mask_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"RSS metadata not found: {meta_path}")

    metadata = {}

    for line in meta_path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    required_metadata = {
        "shape",
        "order",
        "dtype",
        "unsafe",
        "safe",
    }

    missing = required_metadata - metadata.keys()

    if missing:
        raise ValueError(
            f"Missing RSS metadata fields: {sorted(missing)}"
        )

    mask_shape = tuple(
        int(value)
        for value in metadata["shape"].split(",")
    )

    expected_order = (
        "x_rel,y_rel,theta_rel,v_H,delta_E,v_E"
    )

    if metadata["order"] != expected_order:
        raise ValueError(
            f"Unexpected RSS state order: {metadata['order']}"
        )

    if metadata["dtype"] != "uint8":
        raise ValueError(
            f"Unexpected RSS mask dtype: {metadata['dtype']}"
        )

    if metadata["unsafe"] != "1" or metadata["safe"] != "0":
        raise ValueError(
            "Expected RSS convention unsafe=1, safe=0"
        )

    grid_shape = tuple(int(n) for n in grid.shape)

    if mask_shape != grid_shape:
        raise ValueError(
            f"RSS mask shape {mask_shape} does not match "
            f"grid shape {grid_shape}"
        )

    expected_axes = (
        np.linspace(-12.0, 17.0, 30),
        np.linspace(-8.0, 8.0, 17),
        np.linspace(-np.pi, np.pi, 48, endpoint=False),
        np.linspace(1.0, 11.0, 8),
        np.linspace(-np.pi / 12.0, np.pi / 12.0, 11),
        np.linspace(1.0, 11.0, 8),
    )

    for i, (axis, expected) in enumerate(
        zip(grid.coordinate_vectors, expected_axes)
    ):
        axis = np.asarray(axis)

        if axis.shape != expected.shape or not np.allclose(
            axis,
            expected,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                f"Grid axis {i} does not match the grid "
                "used to generate the Intel RSS mask"
            )

    total_states = int(np.prod(mask_shape))

    unsafe_mask = np.fromfile(
        mask_path,
        dtype=np.uint8,
    )

    if unsafe_mask.size != total_states:
        raise ValueError(
            f"RSS mask contains {unsafe_mask.size} states, "
            f"expected {total_states}"
        )

    unsafe_mask = unsafe_mask.reshape(mask_shape).astype(bool)

    n_theta = mask_shape[2]

    periodic_mask = np.concatenate(
        (
            unsafe_mask,
            unsafe_mask,
            unsafe_mask,
        ),
        axis=2,
    )

    safe_distance_periodic = distance_transform_edt(
        ~periodic_mask
    )

    safe_distance = safe_distance_periodic[
        :,
        :,
        n_theta:2 * n_theta,
        :,
        :,
        :,
    ].astype(dtype, copy=True)

    del safe_distance_periodic

    unsafe_distance_periodic = distance_transform_edt(
        periodic_mask
    )

    unsafe_distance = unsafe_distance_periodic[
        :,
        :,
        n_theta:2 * n_theta,
        :,
        :,
        :,
    ].astype(dtype, copy=True)

    del unsafe_distance_periodic
    del periodic_mask

    signed_distance = safe_distance
    signed_distance -= unsafe_distance

    del unsafe_distance

    terminal_values = np.clip(
        signed_distance / transition_width_cells,
        -1.0,
        1.0,
    ).astype(dtype, copy=False)

    parameters = {
        "source": "Intel ad-rss-lib unstructured RSS",
        "mask_file": mask_path.name,
        "mask_shape": mask_shape,
        "unsafe_states": int(np.count_nonzero(unsafe_mask)),
        "safe_states": int(total_states - np.count_nonzero(unsafe_mask)),
        "total_states": total_states,
        "transition_width_cells": float(transition_width_cells),
        "distance_definition": "6D Euclidean distance in grid-index coordinates",
        "theta_periodic": True,
        "sign_convention": "V0 <= 0 unsafe",
        "saturation": "clip(signed_distance / transition_width_cells, -1, 1)",
        "storage_dtype": dtype.name,
    }

    return RSSMetricResult(
        terminal_values=terminal_values,
        parameters=parameters,
    )