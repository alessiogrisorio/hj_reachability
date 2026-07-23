"""Analytical propagation of the frozen-input relative vehicle dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class PreparedRelativeKinematics:
    """Terms that do not change during nominal trajectory propagation."""

    x_rel_0: NDArray[np.float64]
    y_rel_0: NDArray[np.float64]
    theta_rel_0: NDArray[np.float64]
    v_human: NDArray[np.float64]
    v_ego: NDArray[np.float64]
    cos_theta_rel_0: NDArray[np.float64]
    sin_theta_rel_0: NDArray[np.float64]
    beta_ego: NDArray[np.float64]
    omega_ego: NDArray[np.float64]
    turning: NDArray[np.bool_]

    @property
    def size(self) -> int:
        """Number of states represented by this object."""

        return self.x_rel_0.size


def prepare_relative_kinematics(
    states: ArrayLike,
    dynamics: Any,
    *,
    turning_tolerance: float = 1e-12,
) -> PreparedRelativeKinematics:
    """Precompute the time-independent terms for states shaped ``(N, 6)``."""

    state_array = np.asarray(states, dtype=float)
    if state_array.ndim == 1:
        state_array = state_array[None, :]
    if state_array.ndim != 2 or state_array.shape[1] != 6:
        raise ValueError("states must have shape (N, 6) or (6,)")
    if not np.all(np.isfinite(state_array)):
        raise ValueError("states must contain only finite values")
    if turning_tolerance < 0.0:
        raise ValueError("turning_tolerance must be non-negative")

    try:
        lf = float(dynamics.lf)
        lr = float(dynamics.lr)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("dynamics must provide finite positive lf and lr") from error

    if not np.isfinite(lf) or not np.isfinite(lr) or lf <= 0.0 or lr <= 0.0:
        raise ValueError("dynamics.lf and dynamics.lr must be finite and positive")

    x_rel_0 = state_array[:, 0]
    y_rel_0 = state_array[:, 1]
    theta_rel_0 = state_array[:, 2]
    v_human = state_array[:, 3]
    delta_ego = state_array[:, 4]
    v_ego = state_array[:, 5]

    wheelbase = lf + lr
    beta_ego = np.arctan((lr / wheelbase) * np.tan(delta_ego))
    omega_ego = (
        v_ego * np.cos(beta_ego) / wheelbase * np.tan(delta_ego)
    )

    return PreparedRelativeKinematics(
        x_rel_0=x_rel_0,
        y_rel_0=y_rel_0,
        theta_rel_0=theta_rel_0,
        v_human=v_human,
        v_ego=v_ego,
        cos_theta_rel_0=np.cos(theta_rel_0),
        sin_theta_rel_0=np.sin(theta_rel_0),
        beta_ego=beta_ego,
        omega_ego=omega_ego,
        turning=np.abs(omega_ego) > turning_tolerance,
    )


def propagate_relative_pose(
    state_or_kinematics: ArrayLike | PreparedRelativeKinematics,
    time: ArrayLike,
    dynamics: Any | None = None,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Propagate relative pose analytically at one time per initial state."""

    if isinstance(state_or_kinematics, PreparedRelativeKinematics):
        kinematics = state_or_kinematics
    else:
        if dynamics is None:
            raise ValueError("dynamics is required when initial states are passed")
        kinematics = prepare_relative_kinematics(state_or_kinematics, dynamics)

    time_array = np.asarray(time, dtype=float)
    if time_array.ndim == 0:
        time_array = np.full(kinematics.size, float(time_array))
    else:
        time_array = np.ravel(time_array)
        if time_array.size != kinematics.size:
            raise ValueError("time must be scalar or contain one value per state")
    if not np.all(np.isfinite(time_array)):
        raise ValueError("time must contain only finite values")

    x_human = (
        kinematics.x_rel_0
        + kinematics.v_human * time_array * kinematics.cos_theta_rel_0
    )
    y_human = (
        kinematics.y_rel_0
        + kinematics.v_human * time_array * kinematics.sin_theta_rel_0
    )

    theta_ego = kinematics.omega_ego * time_array
    x_ego = np.empty(kinematics.size, dtype=float)
    y_ego = np.empty(kinematics.size, dtype=float)

    turning = kinematics.turning
    straight = ~turning

    x_ego[turning] = (
        kinematics.v_ego[turning] / kinematics.omega_ego[turning]
        * (
            np.sin(theta_ego[turning] + kinematics.beta_ego[turning])
            - np.sin(kinematics.beta_ego[turning])
        )
    )
    y_ego[turning] = (
        kinematics.v_ego[turning] / kinematics.omega_ego[turning]
        * (
            np.cos(kinematics.beta_ego[turning])
            - np.cos(theta_ego[turning] + kinematics.beta_ego[turning])
        )
    )

    x_ego[straight] = (
        kinematics.v_ego[straight]
        * time_array[straight]
        * np.cos(kinematics.beta_ego[straight])
    )
    y_ego[straight] = (
        kinematics.v_ego[straight]
        * time_array[straight]
        * np.sin(kinematics.beta_ego[straight])
    )

    delta_x = x_human - x_ego
    delta_y = y_human - y_ego
    cos_theta_ego = np.cos(theta_ego)
    sin_theta_ego = np.sin(theta_ego)

    x_rel = cos_theta_ego * delta_x + sin_theta_ego * delta_y
    y_rel = -sin_theta_ego * delta_x + cos_theta_ego * delta_y
    theta_rel = kinematics.theta_rel_0 - theta_ego
    return x_rel, y_rel, theta_rel
