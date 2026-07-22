"""Euclidean distance from terminal set"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from ..geometry import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    TerminalSetBoundary,
    VehicleGeometry,
    collision_margin_sat,
)

@dataclass(frozen=True, slots=True)
class EuclideanMetricResult:
    """Result of computing the Euclidean distance from the terminal set."""

    values: Any
    valid: Any
    terminal_values: Any
    angular_scale: float


def metricEuclidean(
    grid: Any,
    boundary: TerminalSetBoundary,
    ego: VehicleGeometry = DEFAULT_EGO_GEOMETRY,
    human: VehicleGeometry = DEFAULT_HUMAN_GEOMETRY,
    *,
    collision_tolerance: float = 1e-9,
) -> EuclideanMetricResult:
    
    if grid.ndim != 6:
        raise ValueError("grid must be 6-dimensional")
    
    if collision_tolerance < 0.0:
        raise ValueError("collision_tolerance must be non-negative")
    
    axes = tuple(
        np.asarray(axis, dtype=float)
        for axis in grid.coordinate_vectors
    )

    if any(axis.ndim != 1 or axis.size == 0 for axis in axes):
        raise ValueError("grid coordinate vectors must be one-dimensional and non-empty")
    
    if any(not np.all(np.isfinite(axis)) for axis in axes):
        raise ValueError("grid coordinate vectors must contain only finite values")
    
    x_axis, y_axis, theta_axis = axes[:3]

    x_extent = np.max(np.abs(x_axis))
    y_extent = np.max(np.abs(y_axis))

    angular_scale = float(
        np.hypot(x_extent, y_extent) / np.pi
    )

    if angular_scale == 0.0:
        raise ValueError("grid must have non-zero extent in x or y")
    
    boundary_points = boundary.flattened_points()
    periodic_boundary = np.concatenate(
        (
            boundary_points + np.array([0.0, 0.0, -np.pi]),
            boundary_points,
            boundary_points + np.array([0.0, 0.0, np.pi]),
        ),
        axis=0,
    )

    periodic_boundary[:, 2] *= angular_scale
    
    tree = cKDTree(periodic_boundary)

    x_pose, y_pose, theta_pose = np.meshgrid(
        x_axis,
        y_axis,
        theta_axis,
        indexing="ij",
    )

    query_points = np.column_stack(
        (
            x_pose.ravel(),
            y_pose.ravel(),
            angular_scale * theta_pose.ravel(),
        )
    )

    unsigned_distance, _ = tree.query(
        query_points,
        k=1,
    )

    unsigned_distance = unsigned_distance.reshape(x_pose.shape)

    collision_margin = collision_margin_sat(
        x_pose,
        y_pose,
        theta_pose,
        ego=ego,
        human=human,
    )

    pose_values = np.where(
        collision_margin <= -collision_tolerance,
        -unsigned_distance,
        unsigned_distance,
    )

    pose_values = np.where(
        np.abs(collision_margin) <= collision_tolerance,
        0.0,
        pose_values,
    )

    values = jnp.asarray(
        np.broadcast_to(
            pose_values[..., None, None, None],
            grid.shape,
        )
    )

    valid = jnp.ones(
        grid.shape,
        dtype=bool,
    )

    return EuclideanMetricResult(
        values=values,
        valid=valid,
        terminal_values=values,
        angular_scale=angular_scale,
    )