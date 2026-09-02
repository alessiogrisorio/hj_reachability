"""Tests for the Distance of Closest Encounter metric."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hj_reachability.systems.relative_vehicle_6d import (
    RelativeVehicle6D,
)
from hj_reachability.vehicle.geometry import (
    build_terminal_set_boundary,
)
from hj_reachability.vehicle.metrics import metricDCE


@dataclass
class DummyGrid:
    """Minimal grid interface required by metricDCE."""

    coordinate_vectors: tuple[np.ndarray, ...]
    ndim: int = 6

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(
            axis.size
            for axis in self.coordinate_vectors
        )


def build_test_grid() -> DummyGrid:
    """Build a small grid containing simple test cases."""

    return DummyGrid(
        coordinate_vectors=(
            # x_rel:
            # 0 m  -> initial overlap
            # 10 m -> initial separation
            np.array([0.0, 10.0]),

            # y_rel
            np.array([0.0]),

            # theta_rel
            np.array([0.0]),

            # v_H
            np.array([0.0]),

            # delta_E
            np.array([0.0]),

            # v_E:
            # 0 m/s -> stationary Ego
            # 5 m/s -> Ego approaches the Human
            np.array([0.0, 5.0]),
        )
    )


def compute_test_result():
    """Compute DCE for the common test configuration."""

    grid = build_test_grid()

    boundary = build_terminal_set_boundary(
        n_theta=90,
        n_phi=180,
    )

    dynamics = RelativeVehicle6D(
        lf=1.2,
        lr=1.5,
    )

    result = metricDCE(
        grid=grid,
        dynamics=dynamics,
        boundary=boundary,
        horizon=2.0,
        dt=0.05,
        batch_size=10,
    )

    return grid, result


def test_output_shapes() -> None:
    """DCE, TCE and validity must match the grid."""

    grid, result = compute_test_result()

    assert result.values.shape == grid.shape
    assert result.terminal_values.shape == grid.shape
    assert (
        result.time_of_closest_encounter.shape
        == grid.shape
    )
    assert result.valid.shape == grid.shape


def test_outputs_are_finite() -> None:
    """DCE and TCE must contain finite values."""

    _, result = compute_test_result()

    dce = np.asarray(result.dce)
    tce = np.asarray(result.tce)

    assert np.all(np.isfinite(dce))
    assert np.all(np.isfinite(tce))
    assert np.asarray(result.valid).all()


def test_tce_inside_horizon() -> None:
    """TCE must remain inside the prediction interval."""

    _, result = compute_test_result()

    tce = np.asarray(result.tce)

    assert np.all(tce >= 0.0)
    assert np.all(tce <= result.horizon)


def test_stationary_overlapping_vehicles() -> None:
    """A stationary initial overlap must have negative DCE."""

    _, result = compute_test_result()

    # State:
    # x_rel=0, y_rel=0, theta_rel=0,
    # v_H=0, delta_E=0, v_E=0.
    index = (0, 0, 0, 0, 0, 0)

    dce = float(
        np.asarray(result.dce)[index]
    )

    tce = float(
        np.asarray(result.tce)[index]
    )

    assert dce < 0.0
    assert np.isclose(tce, 0.0)


def test_stationary_separated_vehicles() -> None:
    """Stationary separated vehicles must have positive DCE."""

    _, result = compute_test_result()

    # State:
    # x_rel=10, y_rel=0, theta_rel=0,
    # v_H=0, delta_E=0, v_E=0.
    index = (1, 0, 0, 0, 0, 0)

    dce = float(
        np.asarray(result.dce)[index]
    )

    tce = float(
        np.asarray(result.tce)[index]
    )

    assert dce > 0.0
    assert np.isclose(tce, 0.0)


def test_approaching_vehicles() -> None:
    """A faster Ego must reach a negative closest distance."""

    _, result = compute_test_result()

    # State:
    # Human is 10 m ahead and stationary.
    # Ego travels at 5 m/s toward the Human.
    index = (1, 0, 0, 0, 0, 1)

    dce = float(
        np.asarray(result.dce)[index]
    )

    tce = float(
        np.asarray(result.tce)[index]
    )

    assert dce < 0.0
    assert tce > 0.0
    assert tce <= result.horizon


def test_terminal_values_equal_dce() -> None:
    """The DCE must be used directly as terminal value."""

    _, result = compute_test_result()

    np.testing.assert_allclose(
        np.asarray(result.terminal_values),
        np.asarray(result.dce),
    )