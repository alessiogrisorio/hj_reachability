"""Rectangle collision test based on the Separating Axis Theorem (SAT)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .rectangles import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
)


def sat_halfspaces(
    theta_rel: ArrayLike,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the SAT half-spaces of the collision set.

    For each relative heading, the collision set is represented as
    ``axes @ [x_rel, y_rel] <= bounds``. The returned shapes are
    ``theta_rel.shape + (8, 2)`` and ``theta_rel.shape + (8,)``.
    """

    theta = np.asarray(theta_rel, dtype=float)
    cosine = np.cos(theta)
    sine = np.sin(theta)

    ex = np.broadcast_to(np.array([1.0, 0.0]), theta.shape + (2,))
    ey = np.broadcast_to(np.array([0.0, 1.0]), theta.shape + (2,))
    ux = np.stack((cosine, sine), axis=-1)
    uy = np.stack((-sine, cosine), axis=-1)

    axes = np.stack((ex, -ex, ey, -ey, ux, -ux, uy, -uy), axis=-2)

    ego_projection = (
        ego.half_length * np.abs(axes[..., 0])
        + ego.half_width * np.abs(axes[..., 1])
    )
    human_projection = (
        human.half_length * np.abs(np.sum(axes * ux[..., None, :], axis=-1))
        + human.half_width * np.abs(np.sum(axes * uy[..., None, :], axis=-1))
    )
    bounds = ego_projection + human_projection
    return axes, bounds


def collision_margin_sat(
    x_rel: ArrayLike,
    y_rel: ArrayLike,
    theta_rel: ArrayLike,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
) -> NDArray[np.float64]:
    """Return a signed SAT collision margin.

    The sign convention is negative for overlap, zero for contact, and
    positive for separation. This margin is an implicit function of the
    collision set; it is not the exact Euclidean signed distance.
    """

    x, y, theta = np.broadcast_arrays(
        np.asarray(x_rel, dtype=float),
        np.asarray(y_rel, dtype=float),
        np.asarray(theta_rel, dtype=float),
    )
    axes, bounds = sat_halfspaces(theta, ego=ego, human=human)
    position = np.stack((x, y), axis=-1)
    residuals = np.sum(axes * position[..., None, :], axis=-1) - bounds
    return np.max(residuals, axis=-1)


def is_collision(
    x_rel: ArrayLike,
    y_rel: ArrayLike,
    theta_rel: ArrayLike,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    tolerance: float = 1e-9,
) -> NDArray[np.bool_]:
    """Return ``True`` for contact or overlap of the two rectangles."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    return collision_margin_sat(x_rel, y_rel, theta_rel, ego, human) <= tolerance
