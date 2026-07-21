"""Geometry shared by vehicle safety metrics."""

from .collision_sat import collision_margin_sat, is_collision, sat_halfspaces
from .rectangles import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    VehicleGeometry,
    rotation_matrix,
)
from .terminal_set_boundary import (
    TerminalSetBoundary,
    build_terminal_set_boundary,
)

__all__ = [
    "DEFAULT_EGO_GEOMETRY",
    "DEFAULT_HUMAN_GEOMETRY",
    "TerminalSetBoundary",
    "VehicleGeometry",
    "build_terminal_set_boundary",
    "collision_margin_sat",
    "is_collision",
    "rotation_matrix",
    "sat_halfspaces",
]
