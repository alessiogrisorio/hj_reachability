"""Detection and bisection refinement of SAT contact times."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..geometry import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
    collision_margin_sat,
)
from .relative_motion import (
    prepare_relative_kinematics,
    propagate_relative_pose,
)


def _refine_contact_time(
    states: NDArray[np.float64],
    direction: int,
    time_low: float,
    time_high: float,
    time_tolerance: float,
    dynamics: Any,
    ego: VehicleGeometry,
    human: VehicleGeometry,
    max_iterations: int,
) -> NDArray[np.float64]:
    """Refine a known boundary crossing using vectorized bisection."""

    kinematics = prepare_relative_kinematics(states, dynamics)
    low = np.full(states.shape[0], time_low, dtype=float)
    high = np.full(states.shape[0], time_high, dtype=float)

    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        x_rel, y_rel, theta_rel = propagate_relative_pose(
            kinematics,
            direction * middle,
        )
        margin = collision_margin_sat(
            x_rel,
            y_rel,
            theta_rel,
            ego=ego,
            human=human,
        )

        before_contact = margin > 0.0 if direction > 0 else margin < 0.0
        low[before_contact] = middle[before_contact]
        high[~before_contact] = middle[~before_contact]

        if np.max(high - low) <= time_tolerance:
            break

    return high


def find_first_contact_time(
    states: ArrayLike,
    direction: int,
    horizon: float,
    dt: float,
    time_tolerance: float,
    dynamics: Any,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    max_iterations: int = 60,
) -> NDArray[np.float64]:
    """Find the first SAT crossing; return positive times or ``NaN``."""

    state_array = np.asarray(states, dtype=float)
    if state_array.ndim == 1:
        state_array = state_array[None, :]
    if state_array.ndim != 2 or state_array.shape[1] != 6:
        raise ValueError("states must have shape (N, 6) or (6,)")
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if not np.isfinite(dt) or dt <= 0.0 or dt > horizon:
        raise ValueError("dt must be finite, positive, and no greater than horizon")
    if not np.isfinite(time_tolerance) or time_tolerance <= 0.0:
        raise ValueError("time_tolerance must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    n_states = state_array.shape[0]
    contact_time = np.full(n_states, np.nan, dtype=float)
    active = np.ones(n_states, dtype=bool)
    n_steps = int(np.ceil(horizon / dt))
    time_previous = 0.0

    for step in range(1, n_steps + 1):
        time_current = min(step * dt, horizon)
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break

        active_states = state_array[active_indices]
        x_rel, y_rel, theta_rel = propagate_relative_pose(
            active_states,
            direction * time_current,
            dynamics,
        )
        margin = collision_margin_sat(
            x_rel,
            y_rel,
            theta_rel,
            ego=ego,
            human=human,
        )
        crossed = margin <= 0.0 if direction > 0 else margin >= 0.0

        if np.any(crossed):
            crossed_indices = active_indices[crossed]
            contact_time[crossed_indices] = _refine_contact_time(
                state_array[crossed_indices],
                direction,
                time_previous,
                time_current,
                time_tolerance,
                dynamics,
                ego,
                human,
                max_iterations,
            )
            active[crossed_indices] = False

        time_previous = time_current

    return contact_time
