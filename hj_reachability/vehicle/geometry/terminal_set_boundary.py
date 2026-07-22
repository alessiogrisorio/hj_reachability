"""Discretisation of the geometric collision-set boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .collision_sat import sat_halfspaces
from .rectangles import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
)


@dataclass(frozen=True, slots=True)
class TerminalSetBoundary:
    """Sampled collision boundary indexed by relative heading and polar angle."""

    theta: NDArray[np.float64]
    phi: NDArray[np.float64]
    points: NDArray[np.float64]

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        phi = np.asarray(self.phi, dtype=float)
        points = np.asarray(self.points, dtype=float)
        expected_shape = (theta.size, phi.size, 2)
        if theta.ndim != 1 or phi.ndim != 1:
            raise ValueError("theta and phi must be one-dimensional")
        if points.shape != expected_shape:
            raise ValueError(f"points must have shape {expected_shape}")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must contain only finite values")

        theta.setflags(write=False)
        phi.setflags(write=False)
        points.setflags(write=False)
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "phi", phi)
        object.__setattr__(self, "points", points)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.points.shape

    def flattened_points(self) -> NDArray[np.float64]:
        """Return rows ``[x_rel, y_rel, theta_rel]`` for all samples."""

        theta_column = np.broadcast_to(
            self.theta[:, None], self.points.shape[:2]
        )[..., None]
        return np.concatenate((self.points, theta_column), axis=-1).reshape(-1, 3)


def build_terminal_set_boundary(
    theta: ArrayLike | None = None,
    phi: ArrayLike | None = None,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    n_theta: int = 360,
    n_phi: int = 360,
    denominator_tolerance: float = 1e-12,
) -> TerminalSetBoundary:
    """Build the exact radial boundary sampled over ``theta`` and ``phi``.

    Default headings cover ``[-pi/2, pi/2)`` because rectangular collision
    geometry is pi-periodic. Polar directions cover ``[0, 2*pi)``.
    """

    if denominator_tolerance <= 0.0:
        raise ValueError("denominator_tolerance must be positive")
    if theta is None:
        if n_theta < 1:
            raise ValueError("n_theta must be positive")
        theta_array = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_theta, endpoint=False)
    else:
        theta_array = np.asarray(theta, dtype=float)
    if phi is None:
        if n_phi < 1:
            raise ValueError("n_phi must be positive")
        phi_array = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    else:
        phi_array = np.asarray(phi, dtype=float)

    if theta_array.ndim != 1 or theta_array.size == 0:
        raise ValueError("theta must be a non-empty one-dimensional array")
    if phi_array.ndim != 1 or phi_array.size == 0:
        raise ValueError("phi must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(theta_array)) or not np.all(np.isfinite(phi_array)):
        raise ValueError("theta and phi must contain only finite values")

    axes, bounds = sat_halfspaces(theta_array, ego=ego, human=human)
    directions = np.stack((np.cos(phi_array), np.sin(phi_array)), axis=-1)
    denominators = np.einsum("tai,pi->tap", axes, directions)

    candidates = np.full(denominators.shape, np.inf, dtype=float)
    valid = denominators > denominator_tolerance
    np.divide(bounds[..., None], denominators, out=candidates, where=valid)
    radii = np.min(candidates, axis=1)
    if not np.all(np.isfinite(radii)):
        raise RuntimeError("failed to intersect one or more boundary rays")

    points = radii[..., None] * directions[None, ...]
    return TerminalSetBoundary(theta=theta_array, phi=phi_array, points=points)
