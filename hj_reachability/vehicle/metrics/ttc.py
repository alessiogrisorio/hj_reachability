"""Signed Time To Collision for frozen-input relative vehicle motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..geometry import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
    collision_margin_sat,
)
from ..kinematics import find_first_contact_time


@dataclass(frozen=True, slots=True)
class TTCMetricResult:
    """Result of computing signed TTC on a six-dimensional grid."""

    values: Any
    valid: Any
    terminal_values: Any
    horizon: float
    dt: float
    time_tolerance: float
    no_collision_value: float

    @property
    def ttc(self) -> Any:
        """Alias that makes the metric-specific output explicit."""

        return self.values


def metricTTC(
    grid: Any,
    dynamics: Any,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    horizon: float,
    dt: float,
    collision_tolerance: float = 1e-10,
    time_tolerance: float | None = None,
    batch_size: int = 100_000,
    max_bisection_iterations: int = 60,
) -> TTCMetricResult:
    """Compute signed TTC at every node of a 6D relative-state grid.

    State ordering is
    ``[x_rel, y_rel, theta_rel, v_human, delta_ego, v_ego]``.
    """

    if grid.ndim != 6:
        raise ValueError("grid must be 6-dimensional")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0 or dt > horizon:
        raise ValueError("dt must be finite, positive, and no greater than horizon")
    if not np.isfinite(collision_tolerance) or collision_tolerance < 0.0:
        raise ValueError("collision_tolerance must be finite and non-negative")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        not isinstance(max_bisection_iterations, int)
        or max_bisection_iterations <= 0
    ):
        raise ValueError("max_bisection_iterations must be a positive integer")

    if time_tolerance is None:
        time_tolerance = max(1e-8, dt / 1000.0)
    elif not np.isfinite(time_tolerance) or time_tolerance <= 0.0:
        raise ValueError("time_tolerance must be finite and positive")

    axes = tuple(
        np.asarray(axis, dtype=float)
        for axis in grid.coordinate_vectors
    )
    if len(axes) != 6:
        raise ValueError("grid must provide six coordinate vectors")
    if any(axis.ndim != 1 or axis.size == 0 for axis in axes):
        raise ValueError("grid coordinate vectors must be one-dimensional and non-empty")
    if any(not np.all(np.isfinite(axis)) for axis in axes):
        raise ValueError("grid coordinate vectors must contain only finite values")

    shape = tuple(axis.size for axis in axes)
    n_points = int(np.prod(shape))
    no_collision_value = 2.0 * horizon
    values_flat = np.empty(n_points, dtype=float)

    for first in range(0, n_points, batch_size):
        last = min(first + batch_size, n_points)
        flat_indices = np.arange(first, last)
        grid_indices = np.unravel_index(flat_indices, shape)
        states = np.column_stack(
            tuple(axis[index] for axis, index in zip(axes, grid_indices))
        )

        margin_0 = collision_margin_sat(
            states[:, 0],
            states[:, 1],
            states[:, 2],
            ego=ego,
            human=human,
        )
        outside = margin_0 > collision_tolerance
        inside = margin_0 < -collision_tolerance
        contact = ~(outside | inside)

        batch_values = np.empty(last - first, dtype=float)
        batch_values[contact] = 0.0

        if np.any(outside):
            future_contact = find_first_contact_time(
                states[outside],
                direction=1,
                horizon=horizon,
                dt=dt,
                time_tolerance=time_tolerance,
                dynamics=dynamics,
                ego=ego,
                human=human,
                max_iterations=max_bisection_iterations,
            )
            outside_values = np.full(np.count_nonzero(outside), no_collision_value)
            found = ~np.isnan(future_contact)
            outside_values[found] = future_contact[found]
            batch_values[outside] = outside_values

        if np.any(inside):
            past_contact = find_first_contact_time(
                states[inside],
                direction=-1,
                horizon=horizon,
                dt=dt,
                time_tolerance=time_tolerance,
                dynamics=dynamics,
                ego=ego,
                human=human,
                max_iterations=max_bisection_iterations,
            )
            inside_values = np.full(np.count_nonzero(inside), -horizon)
            found = ~np.isnan(past_contact)
            inside_values[found] = -past_contact[found]
            batch_values[inside] = inside_values

        values_flat[first:last] = batch_values

    values = jnp.asarray(values_flat.reshape(shape))
    valid = jnp.ones(shape, dtype=bool)
    return TTCMetricResult(
        values=values,
        valid=valid,
        terminal_values=values,
        horizon=float(horizon),
        dt=float(dt),
        time_tolerance=float(time_tolerance),
        no_collision_value=no_collision_value,
    )
