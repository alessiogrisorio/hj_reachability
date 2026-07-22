"""Vehicle geometry, metrics, and terminal-condition utilities."""

from .geometry import (
    DEFAULT_EGO_GEOMETRY,
    DEFAULT_HUMAN_GEOMETRY,
    TerminalSetBoundary,
    VehicleGeometry,
    build_terminal_set_boundary,
    collision_margin_sat,
    is_collision,
)

__all__ = [
    "DEFAULT_EGO_GEOMETRY",
    "DEFAULT_HUMAN_GEOMETRY",
    "TerminalSetBoundary",
    "VehicleGeometry",
    "build_terminal_set_boundary",
    "collision_margin_sat",
    "is_collision",
]
