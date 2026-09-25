from dataclasses import dataclass
from math import prod

import numpy as np

from ..geometry.rectangles import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
)

@dataclass
class RSSMetricResult:
    terminal_values: np.ndarray
    parameters: dict


def metricRSS(
    grid,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    rho: float = 0.496,
    mu: float = 0.2,
    a_max_accel: float = 2.5,
    a_min_brake: float = 7.0,
    a_max_brake: float = 7.0,
    a_lat_max_accel: float = 0.68,
    a_lat_min_brake: float = 0.45,
    use_symmetry: bool = True,
    dtype=np.float32,
) -> RSSMetricResult:

    # Validation
    parameters = {
        "rho": float(rho),
        "mu": float(mu),
        "a_max_accel": float(a_max_accel),
        "a_min_brake": float(a_min_brake),
        "a_max_brake": float(a_max_brake),
        "a_lat_max_accel": float(a_lat_max_accel),
        "a_lat_min_brake": float(a_lat_min_brake),
    }
    for name, value in parameters.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("a_min_brake", "a_max_brake", "a_lat_min_brake"):
        if parameters[name] == 0.0:
            raise ValueError(f"{name} must be strictly positive")
    if a_min_brake > a_max_brake:
        raise ValueError("a_min_brake must not exceed a_max_brake")
    if not isinstance(use_symmetry, (bool, np.bool_)):
        raise ValueError("use_symmetry must be a boolean")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64")

    axes = tuple(np.asarray(a, dtype=np.float64) for a in grid.coordinate_vectors)
    if len(axes) != 6 or any(
        a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a))
        or np.any(np.diff(a) <= 0.0)
        for a in axes
    ):
        raise ValueError("Expected six finite, nonempty, strictly increasing axes")
    shape = tuple(a.size for a in axes)
    if tuple(grid.shape) != shape:
        raise ValueError("grid.shape does not match its coordinate vectors")
    x, y, theta, vh, _, ve = axes
    if np.any(vh < 0.0) or np.any(ve < 0.0):
        raise ValueError("v_H and v_E must be nonnegative")


    ##
    values = np.empty((shape[0], shape[1], shape[2], shape[3], shape[5]), dtype=dtype)

    tolerance = 1e-6
    reflection_maps = []
    if use_symmetry:
        for axis, angular in ((y, False), (theta, True)):
            difference = axis[:, None] + axis[None, :]
            if angular:
                difference = np.arctan2(np.sin(difference), np.cos(difference))
            errors = np.abs(difference)
            indices = np.argmin(errors, axis=1)
            identity = np.arange(axis.size)
            if (
                np.any(errors[identity, indices] > tolerance)
                or np.unique(indices).size != axis.size
                or not np.array_equal(indices[indices], identity)
            ):
                break
            reflection_maps.append(indices)
    symmetry_used = len(reflection_maps) == 2
    if symmetry_used:
        y_reflection, theta_reflection = reflection_maps

    response_distance = 0.5 * a_max_accel * rho**2
    response_speed = a_max_accel * rho
    ego_stop = ve * rho + response_distance + (ve + response_speed)**2 / (2 * a_min_brake)
    ego_front_braking = ve**2 / (2 * a_max_brake)
    lateral_response_speed = a_lat_max_accel * rho
    lateral_response_distance = a_lat_max_accel * rho**2
    evaluated_states = 0
    evaluated_headings = 0

    for it, angle in enumerate(theta):
        reflected_it = int(theta_reflection[it]) if symmetry_used else it
        if symmetry_used and it > reflected_it:
            continue

        # Self-reflected headings (0 or pi) require only one side of y.
        ny = (shape[1] + 1) // 2 if symmetry_used and it == reflected_it else shape[1]
        selected_y = y[:ny]

        cosine, sine = np.cos(angle), np.sin(angle)
        if abs(cosine) < tolerance:
            cosine, sine = 0.0, float(np.copysign(1.0, sine))
        elif abs(sine) < tolerance:
            sine, cosine = 0.0, float(np.copysign(1.0, cosine))

        # Signed geometric gaps; do not clip overlapping projections to zero.
        extent_x = ego.half_length + human.half_length * abs(cosine) + human.half_width * abs(sine)
        extent_y = ego.half_width + human.half_length * abs(sine) + human.half_width * abs(cosine)
        gap_x = np.abs(x) - extent_x
        gap_y = np.abs(selected_y) - extent_y

        human_vx, human_vy = vh * cosine, vh * sine
        human_speed_x = np.abs(human_vx)
        human_stop = (
            human_speed_x * rho + response_distance
            + (human_speed_x + response_speed)**2 / (2 * a_min_brake)
        )
        human_front_braking = human_vx**2 / (2 * a_max_brake)

        # Thresholds have shape (NvH, NvE). Negative vx at zero speed is
        # treated as stationary, not as an oncoming moving vehicle.
        opposite = human_vx[:, None] < 0.0
        safe_front = np.where(
            opposite,
            human_stop[:, None] + ego_stop[None, :],
            np.maximum(ego_stop[None, :] - human_front_braking[:, None], 0.0),
        )
        safe_behind = np.where(
            opposite,
            0.0,
            np.maximum(human_stop[:, None] - ego_front_braking[None, :], 0.0),
        )
        safe_long = np.where(
            x[:, None, None] > 0.0,
            safe_front,
            np.where(x[:, None, None] < 0.0, safe_behind, np.maximum(safe_front, safe_behind)),
        )
        long_margin = gap_x[:, None, None] - safe_long

        # With ego lateral speed zero, Lemma 4 reduces to one closing speed:
        # positive means human moves towards ego. At y=0 take the larger
        # threshold of both possible orderings (closing = abs(human_vy)).
        closing_speed = -np.sign(selected_y[:, None]) * human_vy[None, :]
        closing_speed = np.where(selected_y[:, None] == 0.0, np.abs(human_vy), closing_speed)
        lateral_closure = (
            closing_speed * rho + lateral_response_distance
            + ((closing_speed + lateral_response_speed)**2 + lateral_response_speed**2)
            / (2 * a_lat_min_brake)
        )
        safe_lat = mu + np.maximum(lateral_closure, 0.0)
        lat_margin = gap_y[:, None] - safe_lat

        # Write the (Nx, Ny, NvH, NvE) maximum directly into the 5D result.
        np.maximum(
            long_margin[:, None, :, :], lat_margin[None, :, :, None],
            out=values[:, :ny, it, :, :],
        )
        if symmetry_used:
            values[:, y_reflection[:ny], reflected_it, :, :] = values[:, :ny, it, :, :]
        evaluated_states += shape[0] * ny * shape[3] * shape[5]
        evaluated_headings += 1

    terminal_values = np.broadcast_to(values[:, :, :, :, None, :], shape)
    parameters.update({
        "ego_length": float(ego.length), "ego_width": float(ego.width),
        "human_length": float(human.length), "human_width": float(human.width),
        "definition": "max(d_long - d_safe_long, d_lat - d_safe_lat)",
        "lateral_threshold": "mu + max(predicted_closure, 0)",
        "coordinate_tie_policy": "larger threshold over both orderings",
        "diverging_longitudinal_threshold": 0.0,
        "delta_invariant": True,
        "use_symmetry": bool(use_symmetry), "symmetry_used": symmetry_used,
        "symmetry_tolerance": tolerance, "trig_zero_tolerance": tolerance,
        "evaluated_states": evaluated_states, "evaluated_headings": evaluated_headings,
        "reduced_states": prod(values.shape), "total_states": prod(shape),
        "storage_dtype": dtype.name, "stored_bytes": int(values.nbytes),
        "broadcast_delta_axis": True,
    })
    return RSSMetricResult(terminal_values=terminal_values, parameters=parameters)