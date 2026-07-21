"""Rectangular vehicle geometry in the Ego-fixed reference frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class VehicleGeometry:
    """Dimensions of a rectangular vehicle footprint.

    ``length`` is measured along the vehicle longitudinal axis and ``width``
    along its lateral axis. Both quantities are expressed in metres.
    """

    length: float
    width: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be a finite positive number")
        if not np.isfinite(self.width) or self.width <= 0.0:
            raise ValueError("width must be a finite positive number")

    @property
    def half_length(self) -> float:
        return 0.5 * self.length

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    def vertices(self) -> NDArray[np.float64]:
        """Return the four body-frame vertices with shape ``(4, 2)``."""

        hl = self.half_length
        hw = self.half_width
        return np.array(
            [[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]],
            dtype=float,
        )


DEFAULT_EGO_GEOMETRY = VehicleGeometry(length=4.68, width=2.20)
DEFAULT_HUMAN_GEOMETRY = VehicleGeometry(length=4.28, width=1.80)


def rotation_matrix(angle: ArrayLike) -> NDArray[np.float64]:
    """Return planar rotation matrices with shape ``angle.shape + (2, 2)``."""

    angle_array = np.asarray(angle, dtype=float)
    cosine = np.cos(angle_array)
    sine = np.sin(angle_array)

    matrix = np.empty(angle_array.shape + (2, 2), dtype=float)
    matrix[..., 0, 0] = cosine
    matrix[..., 0, 1] = -sine
    matrix[..., 1, 0] = sine
    matrix[..., 1, 1] = cosine
    return matrix
