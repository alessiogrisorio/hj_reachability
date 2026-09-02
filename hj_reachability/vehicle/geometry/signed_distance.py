"""Signed distance from the vehicle collision set."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .collision_sat import collision_margin_sat
from .rectangles import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
)
from .terminal_set_boundary import TerminalSetBoundary


class SignedDistanceEvaluator:
    """Evaluate signed distance from the collision-set boundary."""

    def __init__(
        self,
        boundary: TerminalSetBoundary,
        angular_scale: float,
        ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
        human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    ) -> None:
        if (
            not np.isfinite(angular_scale)
            or angular_scale <= 0.0
        ):
            raise ValueError(
                "angular_scale must be finite and positive"
            )

        self.boundary = boundary
        self.angular_scale = float(angular_scale)
        self.ego = ego
        self.human = human

        boundary_points = (
            boundary.flattened_points().copy()
        )

        periodic_boundary = np.concatenate(
            (
                boundary_points
                + np.array([0.0, 0.0, -np.pi]),
                boundary_points,
                boundary_points
                + np.array([0.0, 0.0, np.pi]),
            ),
            axis=0,
        )

        periodic_boundary[:, 2] *= (
            self.angular_scale
        )

        self._tree = cKDTree(
            periodic_boundary
        )

    def evaluate(
        self,
        x_rel: ArrayLike,
        y_rel: ArrayLike,
        theta_rel: ArrayLike,
        *,
        collision_tolerance: float = 1e-9,
    ) -> NDArray[np.float64]:
        """Evaluate signed distance at arbitrary relative poses."""

        if (
            not np.isfinite(collision_tolerance)
            or collision_tolerance < 0.0
        ):
            raise ValueError(
                "collision_tolerance must be finite "
                "and non-negative"
            )

        x, y, theta = np.broadcast_arrays(
            np.asarray(x_rel, dtype=float),
            np.asarray(y_rel, dtype=float),
            np.asarray(theta_rel, dtype=float),
        )

        if (
            not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(theta))
        ):
            raise ValueError(
                "relative poses must contain only "
                "finite values"
            )

        query_points = np.column_stack(
            (
                x.ravel(),
                y.ravel(),
                (
                    self.angular_scale
                    * theta.ravel()
                ),
            )
        )

        unsigned_distance, _ = self._tree.query(
            query_points,
            k=1,
        )

        unsigned_distance = (
            unsigned_distance.reshape(x.shape)
        )

        collision_margin = collision_margin_sat(
            x,
            y,
            theta,
            ego=self.ego,
            human=self.human,
        )

        signed_distance = np.where(
            collision_margin
            <= -collision_tolerance,
            -unsigned_distance,
            unsigned_distance,
        )

        signed_distance = np.where(
            np.abs(collision_margin)
            <= collision_tolerance,
            0.0,
            signed_distance,
        )

        return signed_distance


def build_signed_distance_evaluator(
    grid: Any,
    boundary: TerminalSetBoundary,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
) -> SignedDistanceEvaluator:
    """Build a signed-distance evaluator from a grid."""

    if grid.ndim != 6:
        raise ValueError(
            "grid must be 6-dimensional"
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

    x_axis = axes[0]
    y_axis = axes[1]

    x_extent = float(
        np.max(np.abs(x_axis))
    )

    y_extent = float(
        np.max(np.abs(y_axis))
    )

    angular_scale = float(
        np.hypot(x_extent, y_extent)
        / np.pi
    )

    if angular_scale == 0.0:
        raise ValueError(
            "grid must have non-zero extent "
            "in x_rel or y_rel"
        )

    return SignedDistanceEvaluator(
        boundary=boundary,
        angular_scale=angular_scale,
        ego=ego,
        human=human,
    )