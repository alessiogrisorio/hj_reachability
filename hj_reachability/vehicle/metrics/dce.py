"""Distance of Closest Encounter for nominal relative motion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..geometry import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    TerminalSetBoundary,
    VehicleGeometry,
    build_signed_distance_evaluator,
)
from ..kinematics import (
    prepare_relative_kinematics,
    propagate_relative_pose,
)


@dataclass(frozen=True, slots=True)
class DCEMetricResult:
    """Result of computing DCE on a six-dimensional grid."""

    values: Any
    valid: Any
    terminal_values: Any
    time_of_closest_encounter: Any
    horizon: float
    dt: float
    angular_scale: float

    @property
    def dce(self) -> Any:
        """Alias for the Distance of Closest Encounter."""

        return self.values

    @property
    def tce(self) -> Any:
        """Alias for the Time of Closest Encounter."""

        return self.time_of_closest_encounter


def metricDCE(
    grid: Any,
    dynamics: Any,
    boundary: TerminalSetBoundary,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    horizon: float,
    dt: float,
    collision_tolerance: float = 1e-9,
    batch_size: int = 100_000,
) -> DCEMetricResult:
    """Compute signed DCE and TCE on a 6D grid.

    State ordering is:

    [x_rel, y_rel, theta_rel, v_H, delta_E, v_E]

    The nominal motion assumes constant v_H, delta_E and v_E.

    DCE is the minimum signed distance from the collision set
    over the interval [0, horizon].

    TCE is the earliest time at which this minimum occurs.
    """

    if grid.ndim != 6:
        raise ValueError(
            "grid must be 6-dimensional"
        )

    if (
        not np.isfinite(horizon)
        or horizon <= 0.0
    ):
        raise ValueError(
            "horizon must be finite and positive"
        )

    if (
        not np.isfinite(dt)
        or dt <= 0.0
        or dt > horizon
    ):
        raise ValueError(
            "dt must be finite, positive, "
            "and no greater than horizon"
        )

    if (
        not np.isfinite(collision_tolerance)
        or collision_tolerance < 0.0
    ):
        raise ValueError(
            "collision_tolerance must be finite "
            "and non-negative"
        )

    if (
        not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            "batch_size must be a positive integer"
        )

    axes = tuple(
        np.asarray(axis, dtype=float)
        for axis in grid.coordinate_vectors
    )

    if len(axes) != 6:
        raise ValueError(
            "grid must provide six coordinate vectors"
        )

    if any(
        axis.ndim != 1 or axis.size == 0
        for axis in axes
    ):
        raise ValueError(
            "grid coordinate vectors must be "
            "one-dimensional and non-empty"
        )

    if any(
        not np.all(np.isfinite(axis))
        for axis in axes
    ):
        raise ValueError(
            "grid coordinate vectors must contain "
            "only finite values"
        )

    shape = tuple(
        axis.size
        for axis in axes
    )

    number_of_points = int(
        np.prod(shape)
    )

    distance_evaluator = (
        build_signed_distance_evaluator(
            grid=grid,
            boundary=boundary,
            ego=ego,
            human=human,
        )
    )

    number_of_steps = int(
        np.ceil(horizon / dt)
    )

    time_samples = np.minimum(
        np.arange(
            number_of_steps + 1,
            dtype=float,
        ) * dt,
        horizon,
    )

    dce_flat = np.empty(
        number_of_points,
        dtype=float,
    )

    tce_flat = np.empty(
        number_of_points,
        dtype=float,
    )

    for first in range(
        0,
        number_of_points,
        batch_size,
    ):
        last = min(
            first + batch_size,
            number_of_points,
        )

        flat_indices = np.arange(
            first,
            last,
        )

        grid_indices = np.unravel_index(
            flat_indices,
            shape,
        )

        states = np.column_stack(
            tuple(
                axis[index]
                for axis, index
                in zip(axes, grid_indices)
            )
        )

        kinematics = prepare_relative_kinematics(
            states=states,
            dynamics=dynamics,
        )

        batch_dce = np.full(
            last - first,
            np.inf,
            dtype=float,
        )

        batch_tce = np.zeros(
            last - first,
            dtype=float,
        )

        for time in time_samples:
            (
                x_rel,
                y_rel,
                theta_rel,
            ) = propagate_relative_pose(
                state_or_kinematics=kinematics,
                time=time,
            )

            signed_distance = (
                distance_evaluator.evaluate(
                    x_rel=x_rel,
                    y_rel=y_rel,
                    theta_rel=theta_rel,
                    collision_tolerance=(
                        collision_tolerance
                    ),
                )
            )

            improved = (
                signed_distance < batch_dce
            )

            batch_dce[improved] = (
                signed_distance[improved]
            )

            batch_tce[improved] = time

        dce_flat[first:last] = batch_dce
        tce_flat[first:last] = batch_tce

    dce = jnp.asarray(
        dce_flat.reshape(shape)
    )

    tce = jnp.asarray(
        tce_flat.reshape(shape)
    )

    valid = jnp.ones(
        shape,
        dtype=bool,
    )

    return DCEMetricResult(
        values=dce,
        valid=valid,
        terminal_values=dce,
        time_of_closest_encounter=tce,
        horizon=float(horizon),
        dt=float(dt),
        angular_scale=(
            distance_evaluator.angular_scale
        ),
    )