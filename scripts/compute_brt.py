from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hj_reachability as hj

from hj_reachability.systems.relative_vehicle_6d import RelativeVehicle6D
from hj_reachability.vehicle.geometry import build_terminal_set_boundary
from hj_reachability.vehicle.metrics import metricEuclidean

#---- Configuration ----#

STATE_NAMES = (
    "x_rel",
    "y_rel",
    "theta_rel",
    "v_H",
    "delta_E",
    "v_E",
)

INITIAL_TIME = 0.0
TARGET_TIME = -3.0

SOLVER_ACCURACY = "high"

TERMINAL_SET_N_THETA = 90
TERMINAL_SET_N_PHI = 180

GRID_LO = np.array(
    [
        -8.0,
        -6.0,
        -np.pi / 4,
        1.0,
        -np.pi / 12,
        1.0,
    ],
    dtype=np.float32,
)

GRID_HI = np.array(
    [
        17.0,
        6.0,
        np.pi / 4,
        11.0,
        np.pi / 12,
        11.0,
    ],
    dtype=np.float32,
)

GRID_SHAPE = (26, 13, 15, 8, 21, 8)
PERIODIC_DIMS: tuple[int, ...] = ()

#---- Dynamics ----#
DYNAMICS_PARAMETERS = {
    "lf": 1.2,
    "lr": 1.5,
    "ego_min_acceleration": -7.0,
    "ego_max_acceleration": 2.5,
    "ego_max_steering_rate": 0.087,
    "human_min_acceleration": -7.0,
    "human_max_acceleration": 2.5,
    "human_max_yaw_rate": 1.0,
    "control_mode": "max",
    "disturbance_mode": "min",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "brt"
    / "euclidean.npz"
)

#---- Local functions ----#
def build_grid() -> hj.Grid:
    """Build the non-periodic 6D grid."""

    return hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(
            lo=jnp.asarray(GRID_LO),
            hi=jnp.asarray(GRID_HI),
        ),
        shape=GRID_SHAPE,
        periodic_dims=PERIODIC_DIMS,
    )


def print_grid_information(grid: hj.Grid) -> None:
    """Print grid size and resolution."""

    spacing = (
        GRID_HI - GRID_LO
    ) / (np.asarray(GRID_SHAPE) - 1)

    total_points = int(np.prod(GRID_SHAPE))

    print("State order:")
    print(STATE_NAMES)
    print()

    for dimension, name in enumerate(STATE_NAMES):
        print(
            f"{dimension}: {name:10s} | "
            f"range = [{GRID_LO[dimension]: .6f}, "
            f"{GRID_HI[dimension]: .6f}] | "
            f"N = {GRID_SHAPE[dimension]:2d} | "
            f"spacing = {spacing[dimension]:.6f}"
        )

    print()
    print(f"Grid points: {total_points:,}")
    print(f"Periodic dimensions: {PERIODIC_DIMS}")
    print(f"Grid shape: {grid.shape}")


def create_metadata(
    dynamics: RelativeVehicle6D,
    brt: np.ndarray,
    gradients: np.ndarray,
) -> dict:
    """Create metadata describing the complete BRT calculation."""

    grid_spacing = (
        GRID_HI - GRID_LO
    ) / (np.asarray(GRID_SHAPE) - 1)

    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "state_names": list(STATE_NAMES),
        "grid": {
            "lo": GRID_LO.tolist(),
            "hi": GRID_HI.tolist(),
            "shape": list(GRID_SHAPE),
            "spacing": grid_spacing.tolist(),
            "periodic_dims": list(PERIODIC_DIMS),
            "total_points": int(np.prod(GRID_SHAPE)),
        },
        "terminal_set": {
            "metric": "metricEuclidean",
            "n_theta": TERMINAL_SET_N_THETA,
            "n_phi": TERMINAL_SET_N_PHI,
        },
        "dynamics": {
            "class": type(dynamics).__name__,
            "module": type(dynamics).__module__,
            "parameters": DYNAMICS_PARAMETERS,
        },
        "solver": {
            "accuracy": SOLVER_ACCURACY,
            "hamiltonian_postprocessor": (
                "backwards_reachable_tube"
            ),
            "initial_time": INITIAL_TIME,
            "target_time": TARGET_TIME,
        },
        "arrays": {
            "BRT_shape": list(brt.shape),
            "gradients_shape": list(gradients.shape),
            "storage_dtype": "float32",
        },
        "software": {
            "python_version": sys.version,
            "jax_version": jax.__version__,
        },
    }

#---- Main ----#
if __name__ == "__main__":
    print("=" * 70)
    print("6D BACKWARD REACHABLE TUBE CALCULATION")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Grid
    # -----------------------------------------------------------------

    print("\n[1/6] Building grid...")

    grid = build_grid()

    print_grid_information(grid)

    # -----------------------------------------------------------------
    # 2. Terminal-set boundary
    # -----------------------------------------------------------------

    print("\n[2/6] Building terminal-set boundary...")

    boundary = build_terminal_set_boundary(
        n_theta=TERMINAL_SET_N_THETA,
        n_phi=TERMINAL_SET_N_PHI,
    )

    # -----------------------------------------------------------------
    # 3. Terminal value function
    # -----------------------------------------------------------------

    print("\n[3/6] Computing terminal value function V0...")

    metric_result = metricEuclidean(
        grid=grid,
        boundary=boundary,
    )

    V0 = metric_result.terminal_values

    print(f"V0 shape: {V0.shape}")
    print(
        "V0 range: "
        f"[{float(jnp.min(V0)):.6f}, "
        f"{float(jnp.max(V0)):.6f}]"
    )
    print(
        "Angular scale: "
        f"{float(metric_result.angular_scale):.6f} m/rad"
    )

    if tuple(V0.shape) != tuple(grid.shape):
        raise ValueError(
            "V0 shape does not match the grid shape: "
            f"{V0.shape} != {grid.shape}"
        )

    # -----------------------------------------------------------------
    # 4. Dynamics and solver
    # -----------------------------------------------------------------

    print("\n[4/6] Building dynamics and solver settings...")

    dynamics = RelativeVehicle6D(
        **DYNAMICS_PARAMETERS,
    )

    solver_settings = hj.SolverSettings.with_accuracy(
        SOLVER_ACCURACY,
        hamiltonian_postprocessor=(
            hj.solver.backwards_reachable_tube
        ),
    )

    print(f"Dynamics: {type(dynamics).__name__}")
    print(f"Dynamics parameters: {DYNAMICS_PARAMETERS}")
    print(f"Solver accuracy: {SOLVER_ACCURACY}")

    # -----------------------------------------------------------------
    # 5. BRT propagation
    # -----------------------------------------------------------------

    print(
        f"\n[5/6] Propagating BRT from "
        f"t={INITIAL_TIME:.1f} s to "
        f"t={TARGET_TIME:.1f} s..."
    )

    BRT = hj.step(
        solver_settings,
        dynamics,
        grid,
        INITIAL_TIME,
        V0,
        TARGET_TIME,
    )

    print(f"BRT shape: {BRT.shape}")
    print(
        "BRT range: "
        f"[{float(jnp.min(BRT)):.6f}, "
        f"{float(jnp.max(BRT)):.6f}]"
    )

    if tuple(BRT.shape) != tuple(grid.shape):
        raise ValueError(
            "BRT shape does not match the grid shape: "
            f"{BRT.shape} != {grid.shape}"
        )

    # -----------------------------------------------------------------
    # 6. Gradients and saving
    # -----------------------------------------------------------------

    print("\n[6/6] Computing gradients...")

    gradients = grid.grad_values(
        BRT,
        solver_settings.upwind_scheme,
    )

    print(f"Gradient shape: {gradients.shape}")
    print(
        "Gradient range: "
        f"[{float(jnp.min(gradients)):.6f}, "
        f"{float(jnp.max(gradients)):.6f}]"
    )

    expected_gradient_shape = (*tuple(grid.shape), 6)

    if tuple(gradients.shape) != expected_gradient_shape:
        raise ValueError(
            "Unexpected gradient shape: "
            f"{gradients.shape} != "
            f"{expected_gradient_shape}"
        )

    print("\nConverting JAX arrays to NumPy...")

    V0_save = np.asarray(V0, dtype=np.float32)
    BRT_save = np.asarray(BRT, dtype=np.float32)
    gradients_save = np.asarray(
        gradients,
        dtype=np.float32,
    )

    coordinate_vectors = [
        np.asarray(vector, dtype=np.float32)
        for vector in grid.coordinate_vectors
    ]

    metadata = create_metadata(
        dynamics=dynamics,
        brt=BRT_save,
        gradients=gradients_save,
    )

    metadata_json = json.dumps(
        metadata,
        indent=4,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = OUTPUT_PATH

    np.savez_compressed(
        save_path,
        V0=V0_save,
        BRT=BRT_save,
        gradients=gradients_save,
        x_rel=coordinate_vectors[0],
        y_rel=coordinate_vectors[1],
        theta_rel=coordinate_vectors[2],
        v_H=coordinate_vectors[3],
        delta_E=coordinate_vectors[4],
        v_E=coordinate_vectors[5],
        grid_lo=GRID_LO,
        grid_hi=GRID_HI,
        grid_shape=np.asarray(
            GRID_SHAPE,
            dtype=np.int32,
        ),
        periodic_dims=np.asarray(
            PERIODIC_DIMS,
            dtype=np.int32,
        ),
        initial_time=np.float32(INITIAL_TIME),
        target_time=np.float32(TARGET_TIME),
        state_names=np.asarray(STATE_NAMES),
        metadata_json=np.asarray(metadata_json),
    )

    print("\n" + "=" * 70)
    print("CALCULATION COMPLETED")
    print("=" * 70)
    print(f"File saved in: {save_path.resolve()}")
    print(
        "File size: "
        f"{save_path.stat().st_size / 1024**2:.1f} MB"
    )
