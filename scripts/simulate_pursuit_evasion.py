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
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hj_reachability as hj

from hj_reachability.systems.relative_vehicle_6d import RelativeVehicle6D
from hj_reachability.vehicle.geometry import collision_margin_sat

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
MAX_SIMULATION_TIME = 3.0
DT = 0.05
COLLISION_TOLERANCE = 0.0
VALUE_CLASSIFICATION_TOLERANCE = 0.05

# Output configuration.
SAVE_RESULTS = True
SAVE_STATIC_FIGURE = True
SAVE_ANIMATION = True
SHOW_STATIC_FIGURE = False
SHOW_ANIMATION_WINDOW = False
ANIMATION_FORMAT = "mp4"
ANIMATION_FRAME_STRIDE = 1
ANIMATION_FPS = 20
ANIMATION_DPI = 120

# Continue after V0 <= 0 and after a geometrical collision.
STOP_ON_TERMINAL_VALUE = False
STOP_ON_PHYSICAL_COLLISION = False

# Keep velocity and steering states inside their grid ranges.
ENFORCE_CONTROLLED_STATE_BOUNDS = True

# For random modes, require a complete finite simulation.
REQUIRE_FULL_HORIZON_INSIDE_GRID = True
MAX_FULL_HORIZON_ATTEMPTS = 200

# Hybrid HJ/recovery controller. Recovery starts before an unconfined
# state reaches the grid boundary and ends only after a larger margin
# has been recovered (hysteresis).
RECOVERY_ENTRY_GRID_CELLS = 2.0
RECOVERY_EXIT_GRID_CELLS = 4.0
RECOVERY_PREDICTION_HORIZON = 0.75
RECOVERY_PREDICTION_STEPS = 15
RECOVERY_STEERING_SAMPLES = 5
RECOVERY_ACCELERATION_SAMPLES = 5
RECOVERY_SHADE_ALPHA = 0.14

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
def enforce_controlled_state_bounds(
    state: np.ndarray,
    control: np.ndarray,
    disturbance: np.ndarray,
    grid_lo: np.ndarray,
    grid_hi: np.ndarray,
    step_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Restrict the inputs so that v_H, delta_E and v_E cannot
    leave their grid intervals during the next sample-and-hold step.
    """

    corrected_control = np.asarray(
        control,
        dtype=float,
    ).copy()

    corrected_disturbance = np.asarray(
        disturbance,
        dtype=float,
    ).copy()

    if not ENFORCE_CONTROLLED_STATE_BOUNDS:
        return corrected_control, corrected_disturbance

    if step_duration <= 0.0:
        return corrected_control, corrected_disturbance

    # -------------------------------------------------------------
    # Human velocity:
    # v_H_dot = disturbance[1]
    # -------------------------------------------------------------

    minimum_human_acceleration = (
        grid_lo[3] - state[3]
    ) / step_duration

    maximum_human_acceleration = (
        grid_hi[3] - state[3]
    ) / step_duration

    corrected_disturbance[1] = np.clip(
        corrected_disturbance[1],
        minimum_human_acceleration,
        maximum_human_acceleration,
    )

    # -------------------------------------------------------------
    # Ego steering angle:
    # delta_E_dot = control[0]
    # -------------------------------------------------------------

    minimum_steering_rate = (
        grid_lo[4] - state[4]
    ) / step_duration

    maximum_steering_rate = (
        grid_hi[4] - state[4]
    ) / step_duration

    corrected_control[0] = np.clip(
        corrected_control[0],
        minimum_steering_rate,
        maximum_steering_rate,
    )

    # -------------------------------------------------------------
    # Ego velocity:
    # v_E_dot = control[1]
    # -------------------------------------------------------------

    minimum_ego_acceleration = (
        grid_lo[5] - state[5]
    ) / step_duration

    maximum_ego_acceleration = (
        grid_hi[5] - state[5]
    ) / step_duration

    corrected_control[1] = np.clip(
        corrected_control[1],
        minimum_ego_acceleration,
        maximum_ego_acceleration,
    )

    return corrected_control, corrected_disturbance


def evaluate_game(
    brt_data: dict,
    state: np.ndarray,
    time: float,
    step_duration: float = DT,
) -> dict:
    """Evaluate the BRT feedback game at one state."""

    grid = brt_data["grid"]
    dynamics = brt_data["dynamics"]
    grid_lo = brt_data["grid_lo"]
    grid_hi = brt_data["grid_hi"]

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

    optimal_control, optimal_disturbance = (
        dynamics.optimal_control_and_disturbance(
            state_jax,
            time,
            gradient,
        )
    )

    control, disturbance = (
        enforce_controlled_state_bounds(
            state=state,
            control=np.asarray(
                optimal_control,
                dtype=float,
            ),
            disturbance=np.asarray(
                optimal_disturbance,
                dtype=float,
            ),
            grid_lo=grid_lo,
            grid_hi=grid_hi,
            step_duration=step_duration,
        )
    )

    control_jax = jnp.asarray(control)
    disturbance_jax = jnp.asarray(disturbance)

    state_dot = (
        dynamics.open_loop_dynamics(
            state_jax,
            time,
        )
        + dynamics.control_jacobian(
            state_jax,
            time,
        ) @ control_jax
        + dynamics.disturbance_jacobian(
            state_jax,
            time,
        ) @ disturbance_jax
    )

    hamiltonian = jnp.dot(
        gradient,
        state_dot,
    )

    physical_collision_margin = float(
        collision_margin_sat(
            state[0],
            state[1],
            state[2],
        )
    )

    return {
        "brt_value": float(brt_value),
        "terminal_value": float(terminal_value),
        "gradient": np.asarray(
            gradient,
            dtype=float,
        ),
        "control": control,
        "disturbance": disturbance,
        "unconstrained_control": np.asarray(
            optimal_control,
            dtype=float,
        ),
        "unconstrained_disturbance": np.asarray(
            optimal_disturbance,
            dtype=float,
        ),
        "state_dot": np.asarray(
            state_dot,
            dtype=float,
        ),
        "hamiltonian": float(hamiltonian),
        "collision_margin": physical_collision_margin,
    }


# -----------------------------------------------------------------------------
# Local functions III
# -----------------------------------------------------------------------------
def select_initial_state(
    brt_data: dict,
    rng: np.random.Generator | None = None,
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

    if rng is None:
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
# Hybrid HJ/recovery controller
# -----------------------------------------------------------------------------
def unconfined_grid_spacing(
    brt_data: dict,
) -> np.ndarray:
    """Return the grid spacing of x_rel, y_rel and theta_rel."""

    grid_shape = np.asarray(
        brt_data["BRT"].shape,
        dtype=float,
    )

    return (
        brt_data["grid_hi"][:3]
        - brt_data["grid_lo"][:3]
    ) / (grid_shape[:3] - 1.0)


def should_enter_recovery(
    brt_data: dict,
    state: np.ndarray,
) -> bool:
    """Check whether an unconfined state is close to a grid boundary."""

    spacing = unconfined_grid_spacing(
        brt_data
    )

    lower_guard = (
        brt_data["grid_lo"][:3]
        + RECOVERY_ENTRY_GRID_CELLS * spacing
    )

    upper_guard = (
        brt_data["grid_hi"][:3]
        - RECOVERY_ENTRY_GRID_CELLS * spacing
    )

    return bool(
        np.any(state[:3] <= lower_guard)
        or np.any(state[:3] >= upper_guard)
    )


def can_exit_recovery(
    brt_data: dict,
    state: np.ndarray,
) -> bool:
    """Check whether the state has returned to the inner HJ region."""

    spacing = unconfined_grid_spacing(
        brt_data
    )

    lower_guard = (
        brt_data["grid_lo"][:3]
        + RECOVERY_EXIT_GRID_CELLS * spacing
    )

    upper_guard = (
        brt_data["grid_hi"][:3]
        - RECOVERY_EXIT_GRID_CELLS * spacing
    )

    return bool(
        np.all(state[:3] >= lower_guard)
        and np.all(state[:3] <= upper_guard)
    )


def normalized_unconfined_margins(
    brt_data: dict,
    state: np.ndarray,
) -> np.ndarray:
    """
    Return signed distances from the closest grid boundaries,
    normalized by the corresponding domain widths.
    """

    grid_lo = brt_data["grid_lo"][:3]
    grid_hi = brt_data["grid_hi"][:3]
    widths = grid_hi - grid_lo

    return np.minimum(
        state[:3] - grid_lo,
        grid_hi - state[:3],
    ) / widths


def recovery_control(
    brt_data: dict,
    state: np.ndarray,
    time: float,
) -> np.ndarray:
    """
    Select an ego command with a short receding-horizon grid search.

    During recovery the human travels straight at constant speed:
    yaw rate = 0 and acceleration = 0. Candidate ego commands are
    ranked by their predicted signed margin from the x/y/theta bounds.
    """

    dynamics = brt_data["dynamics"]
    grid_lo = brt_data["grid_lo"]
    grid_hi = brt_data["grid_hi"]

    control_lo = np.asarray(
        dynamics.control_space.lo,
        dtype=float,
    )

    control_hi = np.asarray(
        dynamics.control_space.hi,
        dtype=float,
    )

    steering_candidates = np.linspace(
        control_lo[0],
        control_hi[0],
        RECOVERY_STEERING_SAMPLES,
    )

    acceleration_candidates = np.linspace(
        control_lo[1],
        control_hi[1],
        RECOVERY_ACCELERATION_SAMPLES,
    )

    human_straight = np.zeros(
        2,
        dtype=float,
    )

    rollout_dt = (
        RECOVERY_PREDICTION_HORIZON
        / RECOVERY_PREDICTION_STEPS
    )

    best_control = np.zeros(
        2,
        dtype=float,
    )

    best_score = -np.inf

    for steering_rate in steering_candidates:
        for acceleration in acceleration_candidates:
            candidate_control = np.array(
                [
                    steering_rate,
                    acceleration,
                ],
                dtype=float,
            )

            predicted_state = np.asarray(
                state,
                dtype=float,
            ).copy()

            minimum_rollout_margin = np.inf

            for rollout_index in range(
                RECOVERY_PREDICTION_STEPS
            ):
                bounded_control, bounded_disturbance = (
                    enforce_controlled_state_bounds(
                        state=predicted_state,
                        control=candidate_control,
                        disturbance=human_straight,
                        grid_lo=grid_lo,
                        grid_hi=grid_hi,
                        step_duration=rollout_dt,
                    )
                )

                predicted_state_jax = jnp.asarray(
                    predicted_state
                )

                predicted_state_dot = (
                    dynamics.open_loop_dynamics(
                        predicted_state_jax,
                        time
                        + rollout_index * rollout_dt,
                    )
                    + dynamics.control_jacobian(
                        predicted_state_jax,
                        time,
                    ) @ jnp.asarray(bounded_control)
                    + dynamics.disturbance_jacobian(
                        predicted_state_jax,
                        time,
                    ) @ jnp.asarray(bounded_disturbance)
                )

                predicted_state = (
                    predicted_state
                    + rollout_dt
                    * np.asarray(
                        predicted_state_dot,
                        dtype=float,
                    )
                )

                predicted_state[3:6] = np.clip(
                    predicted_state[3:6],
                    grid_lo[3:6],
                    grid_hi[3:6],
                )

                rollout_margin = np.min(
                    normalized_unconfined_margins(
                        brt_data,
                        predicted_state,
                    )
                )

                minimum_rollout_margin = min(
                    minimum_rollout_margin,
                    rollout_margin,
                )

            final_margins = (
                normalized_unconfined_margins(
                    brt_data,
                    predicted_state,
                )
            )

            normalized_effort = np.mean(
                (
                    2.0
                    * (candidate_control - control_lo)
                    / (control_hi - control_lo)
                    - 1.0
                ) ** 2
            )

            score = (
                2.0 * np.min(final_margins)
                + 0.5 * np.mean(final_margins)
                + 0.25 * minimum_rollout_margin
                - 0.01 * normalized_effort
            )

            if score > best_score:
                best_score = score
                best_control = candidate_control

    bounded_control, _ = enforce_controlled_state_bounds(
        state=state,
        control=best_control,
        disturbance=human_straight,
        grid_lo=grid_lo,
        grid_hi=grid_hi,
        step_duration=DT,
    )

    return bounded_control


def evaluate_recovery(
    brt_data: dict,
    state: np.ndarray,
    time: float,
) -> dict:
    """Evaluate the physical dynamics while the HJ feedback is suspended."""

    dynamics = brt_data["dynamics"]

    control = recovery_control(
        brt_data=brt_data,
        state=state,
        time=time,
    )

    disturbance = np.zeros(
        2,
        dtype=float,
    )

    state_jax = jnp.asarray(state)

    state_dot = (
        dynamics.open_loop_dynamics(
            state_jax,
            time,
        )
        + dynamics.control_jacobian(
            state_jax,
            time,
        ) @ jnp.asarray(control)
        + dynamics.disturbance_jacobian(
            state_jax,
            time,
        ) @ jnp.asarray(disturbance)
    )

    return {
        "brt_value": np.nan,
        "terminal_value": np.nan,
        "gradient": np.full(6, np.nan),
        "control": control,
        "disturbance": disturbance,
        "unconstrained_control": control.copy(),
        "unconstrained_disturbance": disturbance.copy(),
        "state_dot": np.asarray(
            state_dot,
            dtype=float,
        ),
        "hamiltonian": np.nan,
        "collision_margin": float(
            collision_margin_sat(
                state[0],
                state[1],
                state[2],
            )
        ),
    }


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
def simulate(
    brt_data: dict,
    initial_state: np.ndarray,
) -> dict:
    """Simulate the closed-loop pursuit-evasion game."""

    dynamics = brt_data["dynamics"]
    grid_lo = brt_data["grid_lo"]
    grid_hi = brt_data["grid_hi"]

    # Only x_rel, y_rel and theta_rel are not directly confined
    # through acceleration or steering-rate bounds.
    unconfined_dimensions = np.array(
        [0, 1, 2],
        dtype=int,
    )

    # -----------------------------------------------------------------
    # Events
    # -----------------------------------------------------------------

    def physical_collision_event(
        local_time: float,
        local_state: np.ndarray,
    ) -> float:
        del local_time

        return float(
            collision_margin_sat(
                local_state[0],
                local_state[1],
                local_state[2],
            )
            - COLLISION_TOLERANCE
        )

    def unconfined_grid_boundary_event(
        local_time: float,
        local_state: np.ndarray,
    ) -> float:
        del local_time

        selected_state = local_state[
            unconfined_dimensions
        ]

        selected_lo = grid_lo[
            unconfined_dimensions
        ]

        selected_hi = grid_hi[
            unconfined_dimensions
        ]

        distance_from_lower_boundary = (
            selected_state - selected_lo
        )

        distance_from_upper_boundary = (
            selected_hi - selected_state
        )

        return float(
            min(
                np.min(distance_from_lower_boundary),
                np.min(distance_from_upper_boundary),
            )
        )

    physical_collision_event.terminal = (
        STOP_ON_PHYSICAL_COLLISION
    )
    physical_collision_event.direction = -1

    # The event is retained for diagnostics but no longer terminates
    # the integration. Recovery handles boundary crossings.
    unconfined_grid_boundary_event.terminal = False
    unconfined_grid_boundary_event.direction = -1

    # -----------------------------------------------------------------
    # Initial conditions and histories
    # -----------------------------------------------------------------

    time = 0.0

    state = np.asarray(
        initial_state,
        dtype=float,
    ).copy()

    times = []
    states = []
    brt_values = []
    terminal_values = []
    collision_margins = []
    hamiltonians = []
    controls = []
    disturbances = []
    unconstrained_controls = []
    unconstrained_disturbances = []
    modes = []

    mode = (
        "recovery"
        if should_enter_recovery(
            brt_data,
            state,
        )
        else "hj"
    )

    first_terminal_value_time = None
    collision_time = None

    stop_reason = "maximum simulation time reached"

    number_of_steps = int(
        np.ceil(
            MAX_SIMULATION_TIME / DT
        )
    )

    # -----------------------------------------------------------------
    # Sample-and-hold simulation
    # -----------------------------------------------------------------

    for _ in range(number_of_steps + 1):
        if not np.isfinite(state).all():
            stop_reason = "non-finite state reached"
            break

        # Correct only negligible numerical drift in the directly
        # confined states. Spatial/angular states are never clipped.
        numerical_tolerance = 1e-9

        controlled_dimensions = np.array(
            [3, 4, 5],
            dtype=int,
        )

        for dimension in controlled_dimensions:
            if (
                state[dimension]
                < grid_lo[dimension]
                and state[dimension]
                >= grid_lo[dimension] - numerical_tolerance
            ):
                state[dimension] = grid_lo[dimension]

            if (
                state[dimension]
                > grid_hi[dimension]
                and state[dimension]
                <= grid_hi[dimension] + numerical_tolerance
            ):
                state[dimension] = grid_hi[dimension]

        if (
            np.any(state[3:6] < grid_lo[3:6])
            or np.any(state[3:6] > grid_hi[3:6])
        ):
            stop_reason = (
                "controlled state left grid"
            )
            break

        remaining_time = max(
            MAX_SIMULATION_TIME - time,
            0.0,
        )

        step_duration = min(
            DT,
            remaining_time,
        )

        evaluation_duration = (
            step_duration
            if step_duration > 0.0
            else DT
        )

        if (
            mode == "hj"
            and should_enter_recovery(
                brt_data,
                state,
            )
        ):
            mode = "recovery"

        elif (
            mode == "recovery"
            and can_exit_recovery(
                brt_data,
                state,
            )
        ):
            mode = "hj"

        if mode == "hj":
            game = evaluate_game(
                brt_data=brt_data,
                state=state,
                time=time,
                step_duration=evaluation_duration,
            )
        else:
            game = evaluate_recovery(
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

        collision_margins.append(
            game["collision_margin"]
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

        unconstrained_controls.append(
            game["unconstrained_control"].copy()
        )

        unconstrained_disturbances.append(
            game["unconstrained_disturbance"].copy()
        )

        modes.append(mode)

        # -------------------------------------------------------------
        # Record V0 <= 0 without stopping
        # -------------------------------------------------------------

        if (
            game["terminal_value"]
            <= COLLISION_TOLERANCE
            and first_terminal_value_time is None
        ):
            first_terminal_value_time = time

        if (
            game["terminal_value"]
            <= COLLISION_TOLERANCE
            and STOP_ON_TERMINAL_VALUE
        ):
            stop_reason = "terminal value became non-positive"
            break

        # -------------------------------------------------------------
        # Record physical collision without necessarily stopping
        # -------------------------------------------------------------

        if (
            game["collision_margin"]
            <= COLLISION_TOLERANCE
            and collision_time is None
        ):
            collision_time = time

        if (
            game["collision_margin"]
            <= COLLISION_TOLERANCE
            and STOP_ON_PHYSICAL_COLLISION
        ):
            stop_reason = "physical collision detected"
            break

        # -------------------------------------------------------------
        # Final time
        # -------------------------------------------------------------

        if (
            time
            >= MAX_SIMULATION_TIME - 1e-12
        ):
            stop_reason = "maximum simulation time reached"
            break

        next_time = min(
            time + DT,
            MAX_SIMULATION_TIME,
        )

        control = game["control"]
        disturbance = game["disturbance"]

        # -------------------------------------------------------------
        # Dynamics with sample-and-hold inputs
        # -------------------------------------------------------------

        def fixed_input_dynamics(
            local_time: float,
            local_state: np.ndarray,
        ) -> np.ndarray:
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
                physical_collision_event,
                unconfined_grid_boundary_event,
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

        # Record the first continuous-time collision event.
        if (
            collision_time is None
            and len(interval_solution.t_events[0]) > 0
        ):
            collision_time = float(
                interval_solution.t_events[0][0]
            )

        time = next_time

        state = np.asarray(
            interval_solution.y[:, -1],
            dtype=float,
        )

        # Numerical cleanup only for v_H, delta_E and v_E.
        state[3:6] = np.clip(
            state[3:6],
            grid_lo[3:6],
            grid_hi[3:6],
        )

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
        "collision_margin": np.asarray(
            collision_margins,
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
        "unconstrained_control": np.asarray(
            unconstrained_controls,
            dtype=float,
        ),
        "unconstrained_disturbance": np.asarray(
            unconstrained_disturbances,
            dtype=float,
        ),
        "mode": np.asarray(
            modes,
            dtype="<U8",
        ),
        "first_terminal_value_time": (
            first_terminal_value_time
        ),
        "terminal_value_became_non_positive": (
            first_terminal_value_time is not None
        ),
        "collision_time": collision_time,
        "collision_occurred": (
            collision_time is not None
        ),
        "stop_reason": stop_reason,
    }


def select_and_simulate_valid_case(
    brt_data: dict,
) -> tuple[np.ndarray, dict, int]:
    """
    Select a case that remains inside the non-periodic BRT grid
    for the complete simulation horizon.
    """

    if INITIAL_STATE_MODE == "manual":
        initial_state = select_initial_state(
            brt_data
        )

        result = simulate(
            brt_data=brt_data,
            initial_state=initial_state,
        )

        if (
            REQUIRE_FULL_HORIZON_INSIDE_GRID
            and result["stop_reason"]
            != "maximum simulation time reached"
        ):
            raise RuntimeError(
                "The manual initial state does not remain inside "
                "the BRT grid for the complete simulation horizon. "
                f"Stop reason: {result['stop_reason']}"
            )

        return initial_state, result, 1

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    for attempt in range(
        1,
        MAX_FULL_HORIZON_ATTEMPTS + 1,
    ):
        initial_state = select_initial_state(
            brt_data=brt_data,
            rng=rng,
        )

        result = simulate(
            brt_data=brt_data,
            initial_state=initial_state,
        )

        completed_full_horizon = (
            result["stop_reason"]
            == "maximum simulation time reached"
        )

        if (
            completed_full_horizon
            or not REQUIRE_FULL_HORIZON_INSIDE_GRID
        ):
            return initial_state, result, attempt

    raise RuntimeError(
        "No random initial state remained inside the BRT grid "
        f"for {MAX_SIMULATION_TIME:.2f} s after "
        f"{MAX_FULL_HORIZON_ATTEMPTS} attempts. "
        "Increase the excluded border, change the initial-state "
        "thresholds or enlarge the grid."
    )


def save_simulation_results(
    result: dict,
    initial_state: np.ndarray,
    brt_path: Path,
) -> Path:
    """Save simulation histories and configuration to an NPZ file."""

    save_directory = project_root / "results" / "simulations"

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    save_path = (
        save_directory
        / f"pursuit_evasion_{timestamp}.npz"
    )

    simulation_metadata = {
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "source_brt_file": brt_path.name,
        "initial_state_mode": INITIAL_STATE_MODE,
        "initial_state": initial_state.tolist(),
        "dt": DT,
        "maximum_simulation_time": (
            MAX_SIMULATION_TIME
        ),
        "brt_horizon": BRT_HORIZON,
        "collision_tolerance": (
            COLLISION_TOLERANCE
        ),
        "stop_reason": result["stop_reason"],
        "collision_time": (
            None
            if result["collision_time"] is None
            else float(result["collision_time"])
        ),
        "state_names": list(STATE_NAMES),
        "control_names": list(CONTROL_NAMES),
        "disturbance_names": list(
            DISTURBANCE_NAMES
        ),
        "first_terminal_value_time": (
            None
            if result["first_terminal_value_time"] is None
            else float(result["first_terminal_value_time"])
        ),
        "terminal_value_became_non_positive": bool(
            result["terminal_value_became_non_positive"]
        ),
        "collision_occurred": bool(
            result["collision_occurred"]
        ),
        "recovery": {
            "entry_grid_cells": RECOVERY_ENTRY_GRID_CELLS,
            "exit_grid_cells": RECOVERY_EXIT_GRID_CELLS,
            "prediction_horizon": RECOVERY_PREDICTION_HORIZON,
            "prediction_steps": RECOVERY_PREDICTION_STEPS,
            "human_yaw_rate": 0.0,
            "human_acceleration": 0.0,
        },
    }

    metadata_json = json.dumps(
        simulation_metadata,
        indent=4,
    )

    np.savez_compressed(
        save_path,
        time=result["time"],
        state=result["state"],
        brt_value=result["brt_value"],
        terminal_value=result["terminal_value"],
        mode=result["mode"],
        hamiltonian=result["hamiltonian"],
        control=result["control"],
        disturbance=result["disturbance"],
        ego_x=result["ego_x"],
        ego_y=result["ego_y"],
        ego_heading=result["ego_heading"],
        human_x=result["human_x"],
        human_y=result["human_y"],
        human_heading=result["human_heading"],
        initial_state=np.asarray(
            initial_state,
            dtype=float,
        ),
        collision_time=np.asarray(
            np.nan
            if result["collision_time"] is None
            else result["collision_time"],
            dtype=float,
        ),
        stop_reason=np.asarray(
            result["stop_reason"]
        ),
        metadata_json=np.asarray(
            metadata_json
        ),
        collision_margin=result["collision_margin"],
        first_terminal_value_time=np.asarray(
            np.nan
            if result["first_terminal_value_time"] is None
            else result["first_terminal_value_time"],
            dtype=float,
        ),
        collision_occurred=np.asarray(
            result["collision_occurred"],
            dtype=bool,
        ),
        terminal_value_became_non_positive=np.asarray(
            result["terminal_value_became_non_positive"],
            dtype=bool,
),
    )

    return save_path


def recovery_time_intervals(
    time: np.ndarray,
    modes: np.ndarray,
) -> list[tuple[float, float]]:
    """Convert the sampled mode history into recovery time intervals."""

    intervals = []
    start_time = None

    for index, mode in enumerate(modes):
        if mode == "recovery" and start_time is None:
            start_time = float(time[index])

        leaving_recovery = (
            mode != "recovery"
            and start_time is not None
        )

        if leaving_recovery:
            intervals.append(
                (
                    start_time,
                    float(time[index]),
                )
            )
            start_time = None

    if start_time is not None:
        intervals.append(
            (
                start_time,
                float(time[-1]),
            )
        )

    return intervals


def shade_recovery_intervals(
    axes,
    time: np.ndarray,
    modes: np.ndarray,
) -> None:
    """Shade recovery periods on time-history axes."""

    intervals = recovery_time_intervals(
        time=time,
        modes=modes,
    )

    for axis in axes:
        for start_time, end_time in intervals:
            axis.axvspan(
                start_time,
                end_time,
                color="red",
                alpha=RECOVERY_SHADE_ALPHA,
                linewidth=0.0,
                zorder=0,
            )


def interpolate_brt_xy_slice(
    brt_data: dict,
    state: np.ndarray,
    brt_array: np.ndarray,
) -> np.ndarray | None:
    """
    Interpolate the BRT over theta_rel, v_H, delta_E and v_E,
    leaving x_rel and y_rel as the two free dimensions.
    """

    coordinate_vectors = [
        np.asarray(vector, dtype=float)
        for vector in brt_data["grid"].coordinate_vectors
    ]

    fixed_indices = []
    fixed_weights = []

    for dimension in range(2, 6):
        coordinates = coordinate_vectors[dimension]
        value = float(state[dimension])

        if (
            value < coordinates[0]
            or value > coordinates[-1]
        ):
            return None

        upper_index = int(
            np.searchsorted(
                coordinates,
                value,
                side="right",
            )
        )

        upper_index = min(
            max(upper_index, 1),
            len(coordinates) - 1,
        )
        lower_index = upper_index - 1

        lower_coordinate = coordinates[lower_index]
        upper_coordinate = coordinates[upper_index]

        if upper_coordinate == lower_coordinate:
            weight = 0.0
        else:
            weight = (
                value - lower_coordinate
            ) / (
                upper_coordinate
                - lower_coordinate
            )

        fixed_indices.append(
            (
                lower_index,
                upper_index,
            )
        )
        fixed_weights.append(weight)

    xy_slice = np.zeros(
        brt_array.shape[:2],
        dtype=float,
    )

    for corner in range(16):
        indices = []
        corner_weight = 1.0

        for local_dimension in range(4):
            use_upper = (
                corner >> local_dimension
            ) & 1

            indices.append(
                fixed_indices[local_dimension][
                    use_upper
                ]
            )

            weight = fixed_weights[
                local_dimension
            ]

            corner_weight *= (
                weight
                if use_upper
                else 1.0 - weight
            )

        xy_slice += (
            corner_weight
            * brt_array[
                :,
                :,
                indices[0],
                indices[1],
                indices[2],
                indices[3],
            ]
        )

    return xy_slice


def create_static_figure(
    result: dict,
) -> tuple[plt.Figure, Path | None]:
    """Create the vertically aligned simulation plots."""

    time = result["time"]
    state = result["state"]
    control = result["control"]
    disturbance = result["disturbance"]

    figure, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(12, 16),
        sharex=True,
        constrained_layout=True,
    )

    # -----------------------------------------------------------------
    # 1. Value functions
    # -----------------------------------------------------------------

    axes[0].plot(
        time,
        result["brt_value"],
        label=r"$V(-3, x(t))$",
        color="tab:blue",
        linewidth=2.0,
    )

    axes[0].plot(
        time,
        result["terminal_value"],
        label=r"$V_0(x(t))$",
        color="tab:orange",
        linewidth=1.8,
    )

    axes[0].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    axes[0].set_ylabel("Value")
    axes[0].set_title(
        "Pursuit-evasion simulation"
    )
    axes[0].legend()
    axes[0].grid(True)

    # -----------------------------------------------------------------
    # 2. Hamiltonian
    # -----------------------------------------------------------------

    axes[1].plot(
        time,
        result["hamiltonian"],
        color="tab:purple",
        linewidth=2.0,
    )

    axes[1].axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_ylabel("Hamiltonian")
    axes[1].grid(True)

    # -----------------------------------------------------------------
    # 3. Ego controls
    # -----------------------------------------------------------------

    ego_steering_axis = axes[2]
    ego_acceleration_axis = ego_steering_axis.twinx()

    ego_steering_line, = ego_steering_axis.plot(
        time,
        np.rad2deg(control[:, 0]),
        label="steering rate",
        color="tab:green",
        linewidth=1.8,
    )

    ego_acceleration_line, = ego_acceleration_axis.plot(
        time,
        control[:, 1],
        label="acceleration",
        color="tab:red",
        linewidth=1.8,
    )

    ego_steering_axis.set_ylabel(
        "Steering rate [deg/s]",
        color="tab:green",
    )
    ego_acceleration_axis.set_ylabel(
        "Acceleration [m/s²]",
        color="tab:red",
    )
    ego_steering_axis.tick_params(
        axis="y",
        colors="tab:green",
    )
    ego_acceleration_axis.tick_params(
        axis="y",
        colors="tab:red",
    )
    ego_steering_axis.legend(
        [ego_steering_line, ego_acceleration_line],
        ["steering rate", "acceleration"],
        loc="upper right",
    )
    ego_steering_axis.grid(True)

    # -----------------------------------------------------------------
    # 4. Human disturbances
    # -----------------------------------------------------------------

    human_yaw_axis = axes[3]
    human_acceleration_axis = human_yaw_axis.twinx()

    human_yaw_line, = human_yaw_axis.plot(
        time,
        np.rad2deg(disturbance[:, 0]),
        label="yaw rate",
        color="tab:brown",
        linewidth=1.8,
    )

    human_acceleration_line, = human_acceleration_axis.plot(
        time,
        disturbance[:, 1],
        label="acceleration",
        color="tab:pink",
        linewidth=1.8,
    )

    human_yaw_axis.set_ylabel(
        "Yaw rate [deg/s]",
        color="tab:brown",
    )
    human_acceleration_axis.set_ylabel(
        "Acceleration [m/s²]",
        color="tab:pink",
    )
    human_yaw_axis.tick_params(
        axis="y",
        colors="tab:brown",
    )
    human_acceleration_axis.tick_params(
        axis="y",
        colors="tab:pink",
    )
    human_yaw_axis.legend(
        [human_yaw_line, human_acceleration_line],
        ["yaw rate", "acceleration"],
        loc="upper right",
    )
    human_yaw_axis.grid(True)

    # -----------------------------------------------------------------
    # 5. Vehicle speeds
    # -----------------------------------------------------------------

    axes[4].plot(
        time,
        state[:, 3],
        label=r"$v_H$",
        linewidth=1.8,
    )

    axes[4].plot(
        time,
        state[:, 5],
        label=r"$v_E$",
        linewidth=1.8,
    )

    axes[4].set_ylabel("Speed [m/s]")
    axes[4].set_xlabel("Simulation time [s]")
    axes[4].legend()
    axes[4].grid(True)

    # -----------------------------------------------------------------
    # Common temporal markers
    # -----------------------------------------------------------------

    for axis in axes:
        axis.axvline(
            BRT_HORIZON,
            color="gray",
            linestyle=":",
            linewidth=1.3,
            label="_nolegend_",
        )

    collision_time = result["collision_time"]

    if collision_time is not None:
        for axis in axes:
            axis.axvline(
                collision_time,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="_nolegend_",
            )

    shade_recovery_intervals(
        axes=axes,
        time=time,
        modes=result["mode"],
    )

    # -----------------------------------------------------------------
    # Optional saving
    # -----------------------------------------------------------------

    figure_path = None

    if SAVE_STATIC_FIGURE:
        figure_directory = (
            project_root
            / "results"
            / "figures"
        )

        figure_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        figure_path = (
            figure_directory
            / f"pursuit_evasion_{timestamp}.png"
        )

        figure.savefig(
            figure_path,
            dpi=200,
            bbox_inches="tight",
        )

    return figure, figure_path

def reconstruct_absolute_trajectories(
    result: dict,
    brt_data: dict,
) -> dict:
    """Reconstruct ego and human poses in an absolute frame."""

    time = result["time"]
    state = result["state"]

    dynamics = brt_data["dynamics"]

    number_of_samples = len(time)

    ego_x = np.zeros(
        number_of_samples,
        dtype=float,
    )

    ego_y = np.zeros(
        number_of_samples,
        dtype=float,
    )

    ego_heading = np.zeros(
        number_of_samples,
        dtype=float,
    )

    # -----------------------------------------------------------------
    # Ego absolute trajectory
    # -----------------------------------------------------------------

    delta_e = state[:, 4]
    v_e = state[:, 5]

    beta_e = np.arctan(
        dynamics.lr
        / (dynamics.lr + dynamics.lf)
        * np.tan(delta_e)
    )

    ego_yaw_rate = (
        v_e
        * np.cos(beta_e)
        / (dynamics.lr + dynamics.lf)
        * np.tan(delta_e)
    )

    for index in range(
        number_of_samples - 1
    ):
        local_dt = (
            time[index + 1]
            - time[index]
        )

        # The velocity direction is psi_E + beta_E.
        velocity_angle_now = (
            ego_heading[index]
            + beta_e[index]
        )

        # First estimate of the next ego heading.
        next_heading = (
            ego_heading[index]
            + 0.5
            * local_dt
            * (
                ego_yaw_rate[index]
                + ego_yaw_rate[index + 1]
            )
        )

        velocity_angle_next = (
            next_heading
            + beta_e[index + 1]
        )

        # Trapezoidal integration of the ego position.
        ego_x[index + 1] = (
            ego_x[index]
            + 0.5
            * local_dt
            * (
                v_e[index]
                * np.cos(
                    velocity_angle_now
                )
                + v_e[index + 1]
                * np.cos(
                    velocity_angle_next
                )
            )
        )

        ego_y[index + 1] = (
            ego_y[index]
            + 0.5
            * local_dt
            * (
                v_e[index]
                * np.sin(
                    velocity_angle_now
                )
                + v_e[index + 1]
                * np.sin(
                    velocity_angle_next
                )
            )
        )

        ego_heading[index + 1] = (
            next_heading
        )

    # -----------------------------------------------------------------
    # Human absolute trajectory
    # -----------------------------------------------------------------

    x_rel = state[:, 0]
    y_rel = state[:, 1]
    theta_rel = state[:, 2]

    cos_ego_heading = np.cos(
        ego_heading
    )

    sin_ego_heading = np.sin(
        ego_heading
    )

    human_x = (
        ego_x
        + cos_ego_heading * x_rel
        - sin_ego_heading * y_rel
    )

    human_y = (
        ego_y
        + sin_ego_heading * x_rel
        + cos_ego_heading * y_rel
    )

    human_heading = (
        ego_heading
        + theta_rel
    )

    return {
        "ego_x": ego_x,
        "ego_y": ego_y,
        "ego_heading": ego_heading,
        "human_x": human_x,
        "human_y": human_y,
        "human_heading": human_heading,
    }


def vehicle_polygon(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Return the absolute coordinates of a vehicle rectangle."""

    half_length = 0.5 * length
    half_width = 0.5 * width

    body_vertices = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=float,
    )

    rotation_matrix = np.array(
        [
            [
                np.cos(heading),
                -np.sin(heading),
            ],
            [
                np.sin(heading),
                np.cos(heading),
            ],
        ],
        dtype=float,
    )

    absolute_vertices = (
        body_vertices
        @ rotation_matrix.T
    )

    absolute_vertices[:, 0] += x
    absolute_vertices[:, 1] += y

    return absolute_vertices


def create_animation(
    result: dict,
    brt_data: dict,
) -> tuple[
    plt.Figure,
    animation.FuncAnimation,
    Path | None,
]:
    """Create and optionally save the pursuit-evasion animation."""

    time = result["time"]
    state = result["state"]
    control = result["control"]
    disturbance = result["disturbance"]
    modes = result["mode"]

    brt_array = np.asarray(
        brt_data["BRT"],
        dtype=float,
    )

    relative_x_coordinates = np.asarray(
        brt_data["grid"].coordinate_vectors[0],
        dtype=float,
    )

    relative_y_coordinates = np.asarray(
        brt_data["grid"].coordinate_vectors[1],
        dtype=float,
    )

    relative_x_mesh, relative_y_mesh = np.meshgrid(
        relative_x_coordinates,
        relative_y_coordinates,
        indexing="ij",
    )

    ego_x = result["ego_x"]
    ego_y = result["ego_y"]
    ego_heading = result["ego_heading"]

    human_x = result["human_x"]
    human_y = result["human_y"]
    human_heading = result["human_heading"]

    # -----------------------------------------------------------------
    # Animation frames
    # -----------------------------------------------------------------

    frame_indices = np.arange(
        0,
        len(time),
        ANIMATION_FRAME_STRIDE,
        dtype=int,
    )

    # Always include the final simulation sample.
    if frame_indices[-1] != len(time) - 1:
        frame_indices = np.append(
            frame_indices,
            len(time) - 1,
        )

    # -----------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------

    figure = plt.figure(
        figsize=(16, 11),
        constrained_layout=True,
    )

    grid_specification = figure.add_gridspec(
        nrows=5,
        ncols=2,
        width_ratios=(1.3, 1.0),
    )

    vehicle_axis = figure.add_subplot(
        grid_specification[:, 0]
    )

    value_axis = figure.add_subplot(
        grid_specification[0, 1]
    )

    hamiltonian_axis = figure.add_subplot(
        grid_specification[1, 1],
        sharex=value_axis,
    )

    control_axis = figure.add_subplot(
        grid_specification[2, 1],
        sharex=value_axis,
    )
    control_acceleration_axis = control_axis.twinx()

    disturbance_axis = figure.add_subplot(
        grid_specification[3, 1],
        sharex=value_axis,
    )
    disturbance_acceleration_axis = (
        disturbance_axis.twinx()
    )

    speed_axis = figure.add_subplot(
        grid_specification[4, 1],
        sharex=value_axis,
    )

    signal_axes = (
        value_axis,
        hamiltonian_axis,
        control_axis,
        disturbance_axis,
        speed_axis,
    )

    # -----------------------------------------------------------------
    # Vehicle panel
    # -----------------------------------------------------------------

    all_x = np.concatenate(
        [
            ego_x,
            human_x,
        ]
    )

    all_y = np.concatenate(
        [
            ego_y,
            human_y,
        ]
    )

    vehicle_margin = max(
        EGO_LENGTH,
        HUMAN_LENGTH,
    )

    vehicle_axis.set_xlim(
        np.min(all_x) - vehicle_margin,
        np.max(all_x) + vehicle_margin,
    )

    vehicle_axis.set_ylim(
        np.min(all_y) - vehicle_margin,
        np.max(all_y) + vehicle_margin,
    )

    vehicle_axis.set_aspect(
        "equal",
        adjustable="box",
    )

    vehicle_axis.set_xlabel(
        "Absolute x [m]"
    )

    vehicle_axis.set_ylabel(
        "Absolute y [m]"
    )

    vehicle_axis.set_title(
        "Absolute vehicle trajectories"
    )

    vehicle_axis.grid(True)

    ego_trajectory_line, = (
        vehicle_axis.plot(
            [],
            [],
            color="tab:blue",
            linewidth=2.0,
            label="Ego trajectory",
        )
    )

    human_trajectory_line, = (
        vehicle_axis.plot(
            [],
            [],
            color="tab:red",
            linewidth=2.0,
            label="Human trajectory",
        )
    )

    ego_polygon = Polygon(
        vehicle_polygon(
            x=ego_x[0],
            y=ego_y[0],
            heading=ego_heading[0],
            length=EGO_LENGTH,
            width=EGO_WIDTH,
        ),
        closed=True,
        facecolor="tab:blue",
        edgecolor="black",
        alpha=0.75,
        label="Ego",
    )

    human_polygon = Polygon(
        vehicle_polygon(
            x=human_x[0],
            y=human_y[0],
            heading=human_heading[0],
            length=HUMAN_LENGTH,
            width=HUMAN_WIDTH,
        ),
        closed=True,
        facecolor="tab:red",
        edgecolor="black",
        alpha=0.75,
        label="Human",
    )

    vehicle_axis.add_patch(
        ego_polygon
    )

    vehicle_axis.add_patch(
        human_polygon
    )

    information_text = vehicle_axis.text(
        0.02,
        0.98,
        "",
        transform=vehicle_axis.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        fontsize=10,
        bbox={
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "gray",
        },
    )

    brt_contour_proxy = Line2D(
        [],
        [],
        color="tab:blue",
        linestyle="--",
        linewidth=2.0,
        label=r"$V(-3,x)=0$",
    )

    vehicle_axis.legend(
        handles=[
            ego_trajectory_line,
            human_trajectory_line,
            ego_polygon,
            human_polygon,
            brt_contour_proxy,
        ],
        loc="lower right",
    )

    # Mutable containers allow the frame callback to replace the
    # Matplotlib contour without using global state.
    current_brt_contour = [None]
    last_valid_brt_slice = [None]


    # -----------------------------------------------------------------
    # Complete signal curves in the background
    # -----------------------------------------------------------------

    ego_steering_rate_deg = np.rad2deg(
        control[:, 0]
    )

    human_yaw_rate_deg = np.rad2deg(
        disturbance[:, 0]
    )

    value_axis.plot(
        time,
        result["brt_value"],
        color="tab:blue",
        alpha=0.20,
        linewidth=1.0,
    )

    value_axis.plot(
        time,
        result["terminal_value"],
        color="tab:orange",
        alpha=0.20,
        linewidth=1.0,
    )

    hamiltonian_axis.plot(
        time,
        result["hamiltonian"],
        color="tab:purple",
        alpha=0.20,
        linewidth=1.0,
    )

    control_axis.plot(
        time,
        ego_steering_rate_deg,
        color="tab:green",
        alpha=0.20,
        linewidth=1.0,
    )

    control_acceleration_axis.plot(
        time,
        control[:, 1],
        color="tab:red",
        alpha=0.20,
        linewidth=1.0,
    )

    disturbance_axis.plot(
        time,
        human_yaw_rate_deg,
        color="tab:brown",
        alpha=0.20,
        linewidth=1.0,
    )

    disturbance_acceleration_axis.plot(
        time,
        disturbance[:, 1],
        color="tab:pink",
        alpha=0.20,
        linewidth=1.0,
    )

    speed_axis.plot(
        time,
        state[:, 3],
        color="tab:blue",
        alpha=0.20,
        linewidth=1.0,
    )

    speed_axis.plot(
        time,
        state[:, 5],
        color="tab:orange",
        alpha=0.20,
        linewidth=1.0,
    )

    # -----------------------------------------------------------------
    # Progressive signal curves
    # -----------------------------------------------------------------

    brt_line, = value_axis.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.0,
        label=r"$V(-3,x(t))$",
    )

    terminal_line, = value_axis.plot(
        [],
        [],
        color="tab:orange",
        linewidth=2.0,
        label=r"$V_0(x(t))$",
    )

    hamiltonian_line, = (
        hamiltonian_axis.plot(
            [],
            [],
            color="tab:purple",
            linewidth=2.0,
            label="Hamiltonian",
        )
    )

    ego_steering_line, = (
        control_axis.plot(
            [],
            [],
            color="tab:green",
            linewidth=2.0,
            label="steering rate [deg/s]",
        )
    )

    ego_acceleration_line, = (
        control_acceleration_axis.plot(
            [],
            [],
            color="tab:red",
            linewidth=2.0,
            label="acceleration [m/s²]",
        )
    )

    human_yaw_line, = (
        disturbance_axis.plot(
            [],
            [],
            color="tab:brown",
            linewidth=2.0,
            label="yaw rate [deg/s]",
        )
    )

    human_acceleration_line, = (
        disturbance_acceleration_axis.plot(
            [],
            [],
            color="tab:pink",
            linewidth=2.0,
            label="acceleration [m/s²]",
        )
    )

    human_speed_line, = speed_axis.plot(
        [],
        [],
        color="tab:blue",
        linewidth=2.0,
        label=r"$v_H$",
    )

    ego_speed_line, = speed_axis.plot(
        [],
        [],
        color="tab:orange",
        linewidth=2.0,
        label=r"$v_E$",
    )

    # -----------------------------------------------------------------
    # Signal-axis formatting
    # -----------------------------------------------------------------

    value_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
    )

    hamiltonian_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=0.9,
    )

    value_axis.set_ylabel("Value")
    hamiltonian_axis.set_ylabel("H")
    control_axis.set_ylabel(
        "Steering rate [deg/s]",
        color="tab:green",
    )
    control_acceleration_axis.set_ylabel(
        "Acceleration [m/s²]",
        color="tab:red",
    )
    disturbance_axis.set_ylabel(
        "Yaw rate [deg/s]",
        color="tab:brown",
    )
    disturbance_acceleration_axis.set_ylabel(
        "Acceleration [m/s²]",
        color="tab:pink",
    )
    control_axis.tick_params(
        axis="y",
        colors="tab:green",
    )
    control_acceleration_axis.tick_params(
        axis="y",
        colors="tab:red",
    )
    disturbance_axis.tick_params(
        axis="y",
        colors="tab:brown",
    )
    disturbance_acceleration_axis.tick_params(
        axis="y",
        colors="tab:pink",
    )
    speed_axis.set_ylabel("Speed [m/s]")
    speed_axis.set_xlabel("Simulation time [s]")

    value_axis.legend(
        loc="upper right",
        fontsize=8,
    )

    hamiltonian_axis.legend(
        loc="upper right",
        fontsize=8,
    )

    control_axis.legend(
        [ego_steering_line, ego_acceleration_line],
        ["steering rate", "acceleration"],
        loc="upper right",
        fontsize=8,
    )

    disturbance_axis.legend(
        [human_yaw_line, human_acceleration_line],
        ["yaw rate", "acceleration"],
        loc="upper right",
        fontsize=8,
    )

    speed_axis.legend(
        loc="upper right",
        fontsize=8,
    )

    maximum_plot_time = max(
        float(time[-1]),
        DT,
    )

    for axis in signal_axes:
        axis.set_xlim(
            0.0,
            maximum_plot_time,
        )

        axis.grid(True)

        axis.axvline(
            BRT_HORIZON,
            color="gray",
            linestyle=":",
            linewidth=1.0,
        )

    shade_recovery_intervals(
        axes=signal_axes,
        time=time,
        modes=modes,
    )

    # -----------------------------------------------------------------
    # Moving time indicators
    # -----------------------------------------------------------------

    time_indicators = [
        axis.axvline(
            time[0],
            color="black",
            linewidth=1.2,
        )
        for axis in signal_axes
    ]

    progressive_lines = (
        brt_line,
        terminal_line,
        hamiltonian_line,
        ego_steering_line,
        ego_acceleration_line,
        human_yaw_line,
        human_acceleration_line,
        human_speed_line,
        ego_speed_line,
    )

    # -----------------------------------------------------------------
    # Frame update
    # -----------------------------------------------------------------

    def update_frame(
        frame_number: int,
    ) -> tuple:
        """Update all artists for one animation frame."""

        data_index = int(
            frame_indices[frame_number]
        )

        current_slice = slice(
            0,
            data_index + 1,
        )

        current_time = float(
            time[data_index]
        )

        ego_trajectory_line.set_data(
            ego_x[current_slice],
            ego_y[current_slice],
        )

        human_trajectory_line.set_data(
            human_x[current_slice],
            human_y[current_slice],
        )

        ego_polygon.set_xy(
            vehicle_polygon(
                x=ego_x[data_index],
                y=ego_y[data_index],
                heading=ego_heading[data_index],
                length=EGO_LENGTH,
                width=EGO_WIDTH,
            )
        )

        human_polygon.set_xy(
            vehicle_polygon(
                x=human_x[data_index],
                y=human_y[data_index],
                heading=human_heading[data_index],
                length=HUMAN_LENGTH,
                width=HUMAN_WIDTH,
            )
        )

        interpolated_slice = interpolate_brt_xy_slice(
            brt_data=brt_data,
            state=state[data_index],
            brt_array=brt_array,
        )

        contour_is_current = (
            interpolated_slice is not None
        )

        if contour_is_current:
            last_valid_brt_slice[0] = (
                interpolated_slice
            )
        else:
            interpolated_slice = (
                last_valid_brt_slice[0]
            )

        if current_brt_contour[0] is not None:
            current_brt_contour[0].remove()
            current_brt_contour[0] = None

        if (
            interpolated_slice is not None
            and np.nanmin(interpolated_slice) <= 0.0
            and np.nanmax(interpolated_slice) >= 0.0
        ):
            cosine_heading = np.cos(
                ego_heading[data_index]
            )
            sine_heading = np.sin(
                ego_heading[data_index]
            )

            contour_x = (
                ego_x[data_index]
                + cosine_heading
                * relative_x_mesh
                - sine_heading
                * relative_y_mesh
            )

            contour_y = (
                ego_y[data_index]
                + sine_heading
                * relative_x_mesh
                + cosine_heading
                * relative_y_mesh
            )

            current_brt_contour[0] = (
                vehicle_axis.contour(
                    contour_x,
                    contour_y,
                    interpolated_slice,
                    levels=[0.0],
                    colors=(
                        "tab:blue"
                        if contour_is_current
                        else "gray"
                    ),
                    linestyles="--",
                    linewidths=2.0,
                    alpha=(
                        0.85
                        if contour_is_current
                        else 0.30
                    ),
                    zorder=2,
                )
            )

        brt_line.set_data(
            time[current_slice],
            result["brt_value"][
                current_slice
            ],
        )

        terminal_line.set_data(
            time[current_slice],
            result["terminal_value"][
                current_slice
            ],
        )

        hamiltonian_line.set_data(
            time[current_slice],
            result["hamiltonian"][
                current_slice
            ],
        )

        ego_steering_line.set_data(
            time[current_slice],
            ego_steering_rate_deg[
                current_slice
            ],
        )

        ego_acceleration_line.set_data(
            time[current_slice],
            control[current_slice, 1],
        )

        human_yaw_line.set_data(
            time[current_slice],
            human_yaw_rate_deg[
                current_slice
            ],
        )

        human_acceleration_line.set_data(
            time[current_slice],
            disturbance[
                current_slice,
                1,
            ],
        )

        human_speed_line.set_data(
            time[current_slice],
            state[current_slice, 3],
        )

        ego_speed_line.set_data(
            time[current_slice],
            state[current_slice, 5],
        )

        for time_indicator in time_indicators:
            time_indicator.set_xdata(
                [
                    current_time,
                    current_time,
                ]
            )

        status_text = ""

        if data_index == len(time) - 1:
            status_text = (
                f"\nstop: "
                f"{result['stop_reason']}"
            )

        current_mode = modes[data_index].upper()

        information_text.set_color(
            "red"
            if current_mode == "RECOVERY"
            else "black"
        )

        information_text.set_text(
            f"t = {current_time:6.2f} s\n"
            f"mode = {current_mode}\n"
            f"V = "
            f"{result['brt_value'][data_index]: .4f}\n"
            f"V0 = "
            f"{result['terminal_value'][data_index]: .4f}\n"
            f"H = "
            f"{result['hamiltonian'][data_index]: .4f}"
            f"{status_text}"
        )

        return (
            ego_trajectory_line,
            human_trajectory_line,
            ego_polygon,
            human_polygon,
            information_text,
            *progressive_lines,
            *time_indicators,
        )

    # -----------------------------------------------------------------
    # Create animation
    # -----------------------------------------------------------------

    simulation_animation = (
        animation.FuncAnimation(
            figure,
            update_frame,
            frames=len(frame_indices),
            interval=(
                1000.0
                / ANIMATION_FPS
            ),
            repeat=False,
            blit=False,
        )
    )

    # Draw the initial frame immediately.
    update_frame(0)

    # -----------------------------------------------------------------
    # Optional saving
    # -----------------------------------------------------------------

    animation_path = None

    if SAVE_ANIMATION:
        animation_directory = (
            project_root
            / "results"
            / "animations"
        )

        animation_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        if ANIMATION_FORMAT == "mp4":
            if not animation.writers.is_available(
                "ffmpeg"
            ):
                raise RuntimeError(
                    "Matplotlib cannot find ffmpeg. "
                    "Install ffmpeg or set "
                    "ANIMATION_FORMAT = 'gif'."
                )

            animation_path = (
                animation_directory
                / f"pursuit_evasion_{timestamp}.mp4"
            )

            video_writer = (
                animation.FFMpegWriter(
                    fps=ANIMATION_FPS,
                    codec="libx264",
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p",
                    ],
                )
            )

            simulation_animation.save(
                animation_path,
                writer=video_writer,
                dpi=ANIMATION_DPI,
            )

        elif ANIMATION_FORMAT == "gif":
            animation_path = (
                animation_directory
                / f"pursuit_evasion_{timestamp}.gif"
            )

            gif_writer = (
                animation.PillowWriter(
                    fps=ANIMATION_FPS,
                )
            )

            simulation_animation.save(
                animation_path,
                writer=gif_writer,
                dpi=ANIMATION_DPI,
            )

        else:
            raise ValueError(
                "ANIMATION_FORMAT must be "
                "'mp4' or 'gif'."
            )

    return (
        figure,
        simulation_animation,
        animation_path,
    )
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

    initial_state, result, selection_attempt = (
        select_and_simulate_valid_case(
            brt_data
        )
    )

    print(
        "Full horizon selection attempts:",
        selection_attempt,
    )

    absolute_trajectories = (
        reconstruct_absolute_trajectories(
            result=result,
            brt_data=brt_data,
        )
    )

    result.update(
        absolute_trajectories
    )

    simulation_save_path = None

    if SAVE_RESULTS:
        simulation_save_path = (
            save_simulation_results(
                result=result,
                initial_state=initial_state,
                brt_path=brt_path,
            )
        )

    static_figure = None
    static_figure_path = None

    if (
        SAVE_STATIC_FIGURE
        or SHOW_STATIC_FIGURE
    ):
        (
            static_figure,
            static_figure_path,
        ) = create_static_figure(
            result=result,
        )

    animation_figure = None
    simulation_animation = None
    animation_path = None

    if (
        SAVE_ANIMATION
        or SHOW_ANIMATION_WINDOW
    ):
        print(
            "Creating animation..."
        )

        (
            animation_figure,
            simulation_animation,
            animation_path,
        ) = create_animation(
            result=result,
            brt_data=brt_data,
        )

        print(
            "Animation completed."
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

    if simulation_save_path is not None:
        print(
            "Simulation data saved in:",
            simulation_save_path.resolve(),
        )

    if static_figure_path is not None:
        print(
            "Static figure saved in:",
            static_figure_path.resolve(),
        )

    if animation_path is not None:
        print(
            "Animation saved in:",
            animation_path.resolve(),
        )

    if (
        SHOW_STATIC_FIGURE
        or SHOW_ANIMATION_WINDOW
    ):
        plt.show()

    else:
        if static_figure is not None:
            plt.close(
                static_figure
            )

        if animation_figure is not None:
            plt.close(
                animation_figure
            )

