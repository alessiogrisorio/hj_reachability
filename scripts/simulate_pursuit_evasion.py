from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from scipy.integrate import solve_ivp

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hj_reachability as hj

from hj_reachability.systems.relative_vehicle_6d import RelativeVehicle6D

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BRT_FILENAME = "brt_euclidean_2.npz"
INITIAL_STATE_MODE = "random_negative"

MANUAL_INITIAL_STATE = np.array(
    [
        6.0,
        2.0,
        np.deg2rad(0.0),
        6.0,
        np.deg2rad(0.0),
        6.0,
    ],
    dtype=float,
)

RANDOM_SEED = 42

# Thresholds used only for random initial-state selection.
NEGATIVE_THRESHOLD = -0.20
POSITIVE_THRESHOLD = 0.20
BOUNDARY_THRESHOLD = 0.05
MINIMUM_INITIAL_TERMINAL_VALUE = 0.10
GRID_BORDER_NODES_TO_EXCLUDE = 1
MAX_RANDOM_SELECTION_ATTEMPTS = 200_000

# Simulation and validation horizons.
BRT_HORIZON = 3.0
MAX_SIMULATION_TIME = 7.0
DT = 0.05
COLLISION_TOLERANCE = 0.0
VALUE_CLASSIFICATION_TOLERANCE = 0.05

# Output configuration.
SAVE_RESULTS = True
SAVE_STATIC_FIGURE = True
SAVE_ANIMATION = True
SHOW_STATIC_FIGURE = True
SHOW_ANIMATION_WINDOW = False
ANIMATION_FRAME_STRIDE = 5
ANIMATION_FPS = 20

# Simple vehicle dimensions used only in the relative-frame animation.
EGO_LENGTH = 4.68
EGO_WIDTH = 2.20
HUMAN_LENGTH = 4.28
HUMAN_WIDTH = 1.80

STATE_NAMES = (
    "x_rel",
    "y_rel",
    "theta_rel",
    "v_H",
    "delta_E",
    "v_E",
)

CONTROL_NAMES = (
    "ego steering rate",
    "ego acceleration",
)

DISTURBANCE_NAMES = (
    "human yaw rate",
    "human acceleration",
)

# -----------------------------------------------------------------------------
# Local functions I
# -----------------------------------------------------------------------------

def load_saved_brt(brt_path: Path) -> dict:
    """Load the BRT and reconstruct its grid and dynamics."""

    if not brt_path.is_file():
        raise FileNotFoundError(
            f"BRT file not found: {brt_path}"
        )

    data = np.load(
        brt_path,
        allow_pickle=False,
    )

    metadata = json.loads(
        str(data["metadata_json"].item())
    )

    grid_lo = np.asarray(
        data["grid_lo"],
        dtype=float,
    )

    grid_hi = np.asarray(
        data["grid_hi"],
        dtype=float,
    )

    grid_shape = tuple(
        int(value)
        for value in data["grid_shape"]
    )

    periodic_dims = tuple(
        int(value)
        for value in data["periodic_dims"]
    )

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(
            lo=jnp.asarray(grid_lo),
            hi=jnp.asarray(grid_hi),
        ),
        shape=grid_shape,
        periodic_dims=periodic_dims,
    )

    dynamics_parameters = metadata["dynamics"]["parameters"]

    dynamics = RelativeVehicle6D(
        **dynamics_parameters
    )

    V0 = jnp.asarray(data["V0"])
    BRT = jnp.asarray(data["BRT"])
    gradients = jnp.asarray(data["gradients"])

    if tuple(V0.shape) != grid_shape:
        raise ValueError(
            f"V0 shape {V0.shape} does not match "
            f"grid shape {grid_shape}"
        )

    if tuple(BRT.shape) != grid_shape:
        raise ValueError(
            f"BRT shape {BRT.shape} does not match "
            f"grid shape {grid_shape}"
        )

    expected_gradient_shape = (
        *grid_shape,
        len(grid_shape),
    )

    if tuple(gradients.shape) != expected_gradient_shape:
        raise ValueError(
            f"Gradient shape {gradients.shape} does not match "
            f"expected shape {expected_gradient_shape}"
        )

    return {
        "grid": grid,
        "dynamics": dynamics,
        "V0": V0,
        "BRT": BRT,
        "gradients": gradients,
        "grid_lo": grid_lo,
        "grid_hi": grid_hi,
        "metadata": metadata,
    }

# -----------------------------------------------------------------------------
# Local functions II
# -----------------------------------------------------------------------------
def evaluate_game(
    brt_data: dict,
    state: np.ndarray,
    time: float,
) -> dict:
    """Evaluate the BRT feedback game at one state."""

    grid = brt_data["grid"]
    dynamics = brt_data["dynamics"]

    state_jax = jnp.asarray(state)

    brt_value = grid.interpolate(
        brt_data["BRT"],
        state_jax,
    )

    terminal_value = grid.interpolate(
        brt_data["V0"],
        state_jax,
    )

    gradient = grid.interpolate(
        brt_data["gradients"],
        state_jax,
    )

    control, disturbance = (
        dynamics.optimal_control_and_disturbance(
            state_jax,
            time,
            gradient,
        )
    )

    state_dot = (
        dynamics.open_loop_dynamics(
            state_jax,
            time,
        )
        + dynamics.control_jacobian(
            state_jax,
            time,
        ) @ control
        + dynamics.disturbance_jacobian(
            state_jax,
            time,
        ) @ disturbance
    )

    hamiltonian = jnp.dot(
        gradient,
        state_dot,
    )

    return {
        "brt_value": float(brt_value),
        "terminal_value": float(terminal_value),
        "gradient": np.asarray(gradient, dtype=float),
        "control": np.asarray(control, dtype=float),
        "disturbance": np.asarray(
            disturbance,
            dtype=float,
        ),
        "state_dot": np.asarray(
            state_dot,
            dtype=float,
        ),
        "hamiltonian": float(hamiltonian),
    }


# -----------------------------------------------------------------------------
# Local functions III
# -----------------------------------------------------------------------------
def select_initial_state(
    brt_data: dict,
) -> np.ndarray:
    """Select the initial state according to INITIAL_STATE_MODE."""

    grid = brt_data["grid"]
    grid_lo = brt_data["grid_lo"]
    grid_hi = brt_data["grid_hi"]

    if INITIAL_STATE_MODE == "manual":
        state = np.asarray(
            MANUAL_INITIAL_STATE,
            dtype=float,
        )

        if state.shape != (6,):
            raise ValueError(
                "MANUAL_INITIAL_STATE must contain "
                "exactly six values."
            )

        if (
            np.any(state < grid_lo)
            or np.any(state > grid_hi)
        ):
            raise ValueError(
                "MANUAL_INITIAL_STATE is outside the grid."
            )

        return state

    valid_modes = {
        "random_negative",
        "random_positive",
        "near_boundary",
    }

    if INITIAL_STATE_MODE not in valid_modes:
        raise ValueError(
            f"Unknown INITIAL_STATE_MODE: "
            f"{INITIAL_STATE_MODE}"
        )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    BRT = np.asarray(
        brt_data["BRT"]
    )

    V0 = np.asarray(
        brt_data["V0"]
    )

    grid_shape = BRT.shape
    total_points = BRT.size

    for _ in range(MAX_RANDOM_SELECTION_ATTEMPTS):
        flat_index = rng.integers(
            0,
            total_points,
        )

        index = np.unravel_index(
            flat_index,
            grid_shape,
        )

        close_to_grid_boundary = any(
            node_index < GRID_BORDER_NODES_TO_EXCLUDE
            or node_index >= dimension_size - GRID_BORDER_NODES_TO_EXCLUDE
            for node_index, dimension_size
            in zip(index, grid_shape)
        )

        if close_to_grid_boundary:
            continue

        brt_value = float(BRT[index])
        terminal_value = float(V0[index])

        if terminal_value <= MINIMUM_INITIAL_TERMINAL_VALUE:
            continue

        if (
            INITIAL_STATE_MODE == "random_negative"
            and brt_value > NEGATIVE_THRESHOLD
        ):
            continue

        if (
            INITIAL_STATE_MODE == "random_positive"
            and brt_value < POSITIVE_THRESHOLD
        ):
            continue

        if (
            INITIAL_STATE_MODE == "near_boundary"
            and abs(brt_value) > BOUNDARY_THRESHOLD
        ):
            continue

        state = np.array(
            [
                float(
                    grid.coordinate_vectors[dimension][
                        index[dimension]
                    ]
                )
                for dimension in range(6)
            ]
        )

        return state

    raise RuntimeError(
        "No suitable initial state was found. "
        "Try increasing MAX_RANDOM_SELECTION_ATTEMPTS "
        "or relaxing the thresholds."
    )


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def simulate(
    brt_data: dict,
    initial_state: np.ndarray,
) -> dict:
    """Simulate the closed-loop pursuit-evasion game."""

    grid = brt_data["grid"]
    dynamics = brt_data["dynamics"]
    grid_lo = brt_data["grid_lo"]
    grid_hi = brt_data["grid_hi"]

    # -----------------------------------------------------------------
    # Events used inside each integration interval
    # -----------------------------------------------------------------

    def collision_event(
        time: float,
        state: np.ndarray,
    ) -> float:
        """Become zero when the terminal set is reached."""

        state_for_interpolation = np.clip(
            state,
            grid_lo,
            grid_hi,
        )

        terminal_value = grid.interpolate(
            brt_data["V0"],
            jnp.asarray(state_for_interpolation),
        )

        return float(
            terminal_value
            - COLLISION_TOLERANCE
        )

    def grid_boundary_event(
        time: float,
        state: np.ndarray,
    ) -> float:
        """Become zero when a grid boundary is reached."""

        distance_from_lower_boundary = (
            state - grid_lo
        )

        distance_from_upper_boundary = (
            grid_hi - state
        )

        return float(
            min(
                np.min(
                    distance_from_lower_boundary
                ),
                np.min(
                    distance_from_upper_boundary
                ),
            )
        )

    collision_event.terminal = True
    collision_event.direction = -1

    grid_boundary_event.terminal = True
    grid_boundary_event.direction = -1

    # -----------------------------------------------------------------
    # Initial conditions and histories
    # -----------------------------------------------------------------

    time = 0.0

    state = np.asarray(
        initial_state,
        dtype=float,
    )

    times = []
    states = []
    brt_values = []
    terminal_values = []
    hamiltonians = []
    controls = []
    disturbances = []

    collision_time = None
    stop_reason = "maximum simulation time reached"

    pending_stop_reason = None

    number_of_steps = int(
        np.ceil(
            MAX_SIMULATION_TIME / DT
        )
    )

    # -----------------------------------------------------------------
    # Sample-and-hold simulation
    # -----------------------------------------------------------------

    for _ in range(
        number_of_steps + 1
    ):
        if not np.isfinite(state).all():
            stop_reason = "non-finite state reached"
            break

        if (
            np.any(state < grid_lo)
            or np.any(state > grid_hi)
        ):
            stop_reason = "state left grid"
            break

        game = evaluate_game(
            brt_data=brt_data,
            state=state,
            time=time,
        )

        times.append(time)
        states.append(state.copy())

        brt_values.append(
            game["brt_value"]
        )

        terminal_values.append(
            game["terminal_value"]
        )

        hamiltonians.append(
            game["hamiltonian"]
        )

        controls.append(
            game["control"].copy()
        )

        disturbances.append(
            game["disturbance"].copy()
        )

        # This happens when an event was detected during
        # the previous integration interval.
        if pending_stop_reason is not None:
            stop_reason = pending_stop_reason

            if (
                pending_stop_reason
                == "terminal set reached"
            ):
                collision_time = time

            break

        if (
            game["terminal_value"]
            <= COLLISION_TOLERANCE
        ):
            collision_time = time
            stop_reason = "terminal set reached"
            break

        if (
            time
            >= MAX_SIMULATION_TIME
            - 1e-12
        ):
            stop_reason = (
                "maximum simulation time reached"
            )
            break

        control = game["control"]
        disturbance = game["disturbance"]

        next_time = min(
            time + DT,
            MAX_SIMULATION_TIME,
        )

        # -------------------------------------------------------------
        # Dynamics with control and disturbance fixed over [time,next_time]
        # -------------------------------------------------------------

        def fixed_input_dynamics(
            local_time: float,
            local_state: np.ndarray,
        ) -> np.ndarray:
            """Evaluate dynamics with fixed inputs."""

            state_jax = jnp.asarray(
                local_state
            )

            state_dot = (
                dynamics.open_loop_dynamics(
                    state_jax,
                    local_time,
                )
                + dynamics.control_jacobian(
                    state_jax,
                    local_time,
                ) @ jnp.asarray(control)
                + dynamics.disturbance_jacobian(
                    state_jax,
                    local_time,
                ) @ jnp.asarray(disturbance)
            )

            return np.asarray(
                state_dot,
                dtype=float,
            )

        interval_solution = solve_ivp(
            fun=fixed_input_dynamics,
            t_span=(
                time,
                next_time,
            ),
            y0=state,
            method="RK45",
            events=[
                collision_event,
                grid_boundary_event,
            ],
            max_step=DT,
            rtol=1e-6,
            atol=1e-8,
        )

        if not interval_solution.success:
            raise RuntimeError(
                "Integration failed: "
                f"{interval_solution.message}"
            )

        # -------------------------------------------------------------
        # Check interval events
        # -------------------------------------------------------------

        if (
            len(
                interval_solution.t_events[0]
            )
            > 0
        ):
            time = float(
                interval_solution.t_events[0][0]
            )

            state = np.asarray(
                interval_solution.y_events[0][0],
                dtype=float,
            )

            pending_stop_reason = (
                "terminal set reached"
            )

        elif (
            len(
                interval_solution.t_events[1]
            )
            > 0
        ):
            time = float(
                interval_solution.t_events[1][0]
            )

            state = np.asarray(
                interval_solution.y_events[1][0],
                dtype=float,
            )

            pending_stop_reason = (
                "grid boundary reached"
            )

        else:
            time = next_time

            state = np.asarray(
                interval_solution.y[:, -1],
                dtype=float,
            )

    # -----------------------------------------------------------------
    # Convert histories to NumPy arrays
    # -----------------------------------------------------------------

    return {
        "time": np.asarray(
            times,
            dtype=float,
        ),
        "state": np.asarray(
            states,
            dtype=float,
        ),
        "brt_value": np.asarray(
            brt_values,
            dtype=float,
        ),
        "terminal_value": np.asarray(
            terminal_values,
            dtype=float,
        ),
        "hamiltonian": np.asarray(
            hamiltonians,
            dtype=float,
        ),
        "control": np.asarray(
            controls,
            dtype=float,
        ),
        "disturbance": np.asarray(
            disturbances,
            dtype=float,
        ),
        "collision_time": collision_time,
        "stop_reason": stop_reason,
    }

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    brt_path = (
        project_root
        / "results"
        / BRT_FILENAME
    )

    brt_data = load_saved_brt(
        brt_path
    )

    initial_state = select_initial_state(
        brt_data
    )

    result = simulate(
        brt_data=brt_data,
        initial_state=initial_state,
    )

    print("Simulation finished")
    print("Stop reason:", result["stop_reason"])
    print(
        "Final time:",
        result["time"][-1],
    )
    print(
        "Number of samples:",
        len(result["time"]),
    )

    print(
        "Initial BRT value:",
        result["brt_value"][0],
    )
    print(
        "Final BRT value:",
        result["brt_value"][-1],
    )

    print(
        "Initial terminal value:",
        result["terminal_value"][0],
    )
    print(
        "Final terminal value:",
        result["terminal_value"][-1],
    )

    print(
        "Initial state:",
        result["state"][0],
    )
    print(
        "Final state:",
        result["state"][-1],
    )

# -----------------------------------------------------------------------------
# Local functions II
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Local functions II
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Local functions II
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Local functions II
# -----------------------------------------------------------------------------