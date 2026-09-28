from __future__ import annotations

import json
from itertools import product

import contourpy
import sys
import gc
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import transforms
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hj_reachability as hj
from hj_reachability.systems.relative_vehicle_6d import RelativeVehicle6D

RECOVERY_POSITION_GAIN = 0.5
RECOVERY_ACCELERATION_WEIGHT = 1.0
RECOVERY_YAW_WEIGHT = 1.0
RECOVERY_LOW_SPEED = 0.2
RECOVERY_HEADING_GAIN = 1.0

def simulate_pursuit_evasion(
    metric: str,
    ego_initial_state: np.ndarray,
    human_initial_state: np.ndarray,
    recovery: bool = False,
    T: float = 10.0,
    dt: float = 1 / 30,
    animation_time_scale_factor: float = 2.0,
):
    """Run the game once and return its HTML5 animation.

    ``metric`` is the exact file name without ``.npz`` inside
    ``results/brt``. The absolute states are

    ``ego_initial_state = [x_E, y_E, psi_E, delta_E, v_E]``
    ``human_initial_state = [x_H, y_H, psi_H, v_H]``.
    """
    gc.collect()
    jax.clear_caches()
    if not metric or Path(metric).name != metric or metric.endswith(".npz"):
        raise ValueError("metric must be a file name without '.npz', e.g. 'dce'.")
    if T <= 0 or dt <= 0 or animation_time_scale_factor <= 0:
        raise ValueError("T, dt and animation_time_scale_factor must be positive.")

    ego_initial_state = np.asarray(ego_initial_state, dtype=float)
    human_initial_state = np.asarray(human_initial_state, dtype=float)
    if ego_initial_state.shape != (5,):
        raise ValueError("ego_initial_state must contain five values.")
    if human_initial_state.shape != (4,):
        raise ValueError("human_initial_state must contain four values.")

    brt_path = ROOT / "results" / "brt" / f"{metric}.npz"
    if not brt_path.is_file():
        raise FileNotFoundError(f"BRT file not found: {brt_path}")

    with np.load(brt_path, allow_pickle=True) as saved_data:
        metadata = json.loads(str(saved_data["metadata_json"].item()))
        grid_lo = np.asarray(saved_data["grid_lo"], dtype=float)
        grid_hi = np.asarray(saved_data["grid_hi"], dtype=float)
        grid_shape = tuple(int(value) for value in saved_data["grid_shape"])
        periodic_dims = tuple(int(value) for value in saved_data["periodic_dims"])
        brt_values = jnp.asarray(saved_data["BRT"])
        terminal_values = jnp.asarray(saved_data["V0"])
        gradient_values = jnp.asarray(saved_data["gradients"])

    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=jnp.asarray(grid_lo), hi=jnp.asarray(grid_hi)),
        shape=grid_shape,
        periodic_dims=periodic_dims,
    )
    dynamics = RelativeVehicle6D(**metadata["dynamics"]["parameters"])

    def relative_state(joint_state):
        x_e, y_e, psi_e, delta_e, v_e, x_h, y_h, psi_h, v_h = joint_state
        rotation = jnp.array([
            [jnp.cos(psi_e), jnp.sin(psi_e)],
            [-jnp.sin(psi_e), jnp.cos(psi_e)],
        ])
        relative_position = rotation @ jnp.array([x_h - x_e, y_h - y_e])
        theta_rel = psi_h - psi_e
        if 2 in periodic_dims:
            period = grid_hi[2] - grid_lo[2]
            theta_rel = grid_lo[2] + jnp.mod(theta_rel - grid_lo[2], period)
        return jnp.array([
            relative_position[0], relative_position[1], theta_rel,
            v_h, delta_e, v_e,
        ])

    def joint_dynamics(joint_state, joint_input):
        _, _, psi_e, delta_e, v_e, _, _, psi_h, v_h = joint_state
        steering_rate_e, acceleration_e, yaw_rate_h, acceleration_h = joint_input
        beta_e = jnp.arctan(
            dynamics.lr / (dynamics.lr + dynamics.lf) * jnp.tan(delta_e)
        )
        yaw_rate_e = (
            v_e * jnp.cos(beta_e) / (dynamics.lr + dynamics.lf)
            * jnp.tan(delta_e)
        )
        return jnp.array([
            v_e * jnp.cos(psi_e + beta_e),
            v_e * jnp.sin(psi_e + beta_e),
            yaw_rate_e,
            steering_rate_e,
            acceleration_e,
            v_h * jnp.cos(psi_h),
            v_h * jnp.sin(psi_h),
            yaw_rate_h,
            acceleration_h,
        ])

    ego_input_min = np.asarray(dynamics.control_space.lo, dtype=float)
    ego_input_max = np.asarray(dynamics.control_space.hi, dtype=float)
    human_input_min = np.asarray(dynamics.disturbance_space.lo, dtype=float)
    human_input_max = np.asarray(dynamics.disturbance_space.hi, dtype=float)

    @jax.jit
    def hj_step(joint_state, step_index):
        state = relative_state(joint_state)
        value = grid.interpolate(brt_values, state)
        value_zero = grid.interpolate(terminal_values, state)
        gradient = grid.interpolate(gradient_values, state)
        control, disturbance = dynamics.optimal_control_and_disturbance(
            state, step_index * dt, gradient
        )
        control = jnp.asarray(control)
        disturbance = jnp.asarray(disturbance)
        v_h, delta_e, v_e = state[3], state[4], state[5]
        control = control.at[0].set(jnp.clip(
            control[0], (grid_lo[4] - delta_e) / dt,
            (grid_hi[4] - delta_e) / dt,
        ))
        control = control.at[1].set(jnp.clip(
            control[1], (grid_lo[5] - v_e) / dt,
            (grid_hi[5] - v_e) / dt,
        ))
        disturbance = disturbance.at[1].set(jnp.clip(
            disturbance[1], (grid_lo[3] - v_h) / dt,
            (grid_hi[3] - v_h) / dt,
        ))
        rate = dynamics(state, control, disturbance, step_index * dt)
        hamiltonian = jnp.dot(gradient, rate)
        next_state = joint_state + joint_dynamics(
            joint_state, jnp.concatenate([control, disturbance])
        ) * dt
        next_state = next_state.at[3].set(jnp.clip(
            next_state[3], grid_lo[4], grid_hi[4]
        ))
        next_state = next_state.at[4].set(jnp.clip(
            next_state[4], grid_lo[5], grid_hi[5]
        ))
        next_state = next_state.at[8].set(jnp.clip(
            next_state[8], grid_lo[3], grid_hi[3]
        ))
        return next_state, value, value_zero, hamiltonian, control, disturbance

    def is_inside_grid(state):
        return all(
            dimension in periodic_dims
            or grid_lo[dimension] <= state[dimension] <= grid_hi[dimension]
            for dimension in range(6)
        )

    joint_states = [np.concatenate([ego_initial_state, human_initial_state])]
    times, modes = [], []
    brt_history, terminal_history, hamiltonian_history = [], [], []
    ego_controls, human_controls = [], []
    recovery_active = False
    recovery_reference_speed = None
    last_valid_ego_speed = float(ego_initial_state[4])
    number_of_steps = int(np.floor(T / dt + 1e-12))

    for step_index in range(number_of_steps + 1):
        current_time = step_index * dt
        current_joint_state = np.asarray(joint_states[-1], dtype=float)
        current_relative_state = np.asarray(
            relative_state(jnp.asarray(current_joint_state)), dtype=float
        )
        inside = is_inside_grid(current_relative_state)

        if inside and not recovery_active:
            last_valid_ego_speed = float(current_relative_state[5])
        if recovery and not inside and not recovery_active:
            recovery_active = True
            recovery_reference_speed = last_valid_ego_speed
            state_names = ("x_rel", "y_rel", "theta_rel", "v_H", "delta_E", "v_E")
            outside_dimensions = [
                dimension for dimension in range(6)
                if dimension not in periodic_dims
                and not grid_lo[dimension] <= current_relative_state[dimension]
                <= grid_hi[dimension]
            ]
            print("\n" + "=" * 70)
            print("RECOVERY MODE ACTIVATED")
            print("=" * 70)
            print(f"Simulation time: {current_time:.6f} s")
            print(f"Simulation step: {step_index}")
            print("Outside dimensions:")
            for dimension in outside_dimensions:
                print(
                    f"  {state_names[dimension]} = "
                    f"{current_relative_state[dimension]:.8f}, grid = "
                    f"[{grid_lo[dimension]:.8f}, {grid_hi[dimension]:.8f}]"
                )
            print(
                f"Stored ego reference speed: {recovery_reference_speed:.8f} m/s"
            )
            print("=" * 70 + "\n")
        elif recovery_active and inside:
            recovery_active = False
            recovery_reference_speed = None

        times.append(current_time)
        modes.append("recovery" if recovery_active else "hj")

        if not recovery_active:
            output = hj_step(jnp.asarray(current_joint_state), step_index)
            next_state, value, value_zero, hamiltonian, control, disturbance = output
            brt_history.append(float(value))
            terminal_history.append(float(value_zero))
            hamiltonian_history.append(float(hamiltonian))
            ego_controls.append(np.asarray(control, dtype=float))
            human_controls.append(np.asarray(disturbance, dtype=float))
        else:
            delta_e = current_relative_state[4]
            v_e = current_relative_state[5]
            v_h = current_relative_state[3]

            ego_control = np.array([
                np.clip(
                    -delta_e / dt,
                    ego_input_min[0],
                    ego_input_max[0],
                ),
                np.clip(
                    np.clip(
                        (recovery_reference_speed - v_e) / dt,
                        ego_input_min[1],
                        ego_input_max[1],
                    ),
                    (grid_lo[5] - v_e) / dt,
                    (grid_hi[5] - v_e) / dt,
                ),
            ])

            psi_e = current_joint_state[2]
            psi_h = current_joint_state[7]

            position_error = (
                current_joint_state[0:2] - current_joint_state[5:7]
            )

            beta_e = np.arctan(
                dynamics.lr / (dynamics.lr + dynamics.lf)
                * np.tan(delta_e)
            )

            ego_velocity = v_e * np.array([
                np.cos(psi_e + beta_e),
                np.sin(psi_e + beta_e),
            ])

            desired_velocity = (
                ego_velocity
                + RECOVERY_POSITION_GAIN * position_error
            )

            human_heading = np.array([
                np.cos(psi_h),
                np.sin(psi_h),
            ])
            human_normal = np.array([
                -np.sin(psi_h),
                np.cos(psi_h),
            ])

            velocity_error = v_h * human_heading - desired_velocity

            acceleration_command = (
                -np.dot(velocity_error, human_heading)
                / RECOVERY_ACCELERATION_WEIGHT
            )

            yaw_command = (
                -v_h * np.dot(velocity_error, human_normal)
                / RECOVERY_YAW_WEIGHT
            )

            if abs(v_h) < RECOVERY_LOW_SPEED:
                if np.linalg.norm(desired_velocity) > 1e-9:
                    desired_heading = np.arctan2(
                        desired_velocity[1],
                        desired_velocity[0],
                    )
                    heading_error = np.arctan2(
                        np.sin(desired_heading - psi_h),
                        np.cos(desired_heading - psi_h),
                    )
                    yaw_command = (
                        RECOVERY_HEADING_GAIN * heading_error
                    )
                else:
                    yaw_command = 0.0

            acceleration_lower = max(
                human_input_min[1],
                (grid_lo[3] - v_h) / dt,
            )
            acceleration_upper = min(
                human_input_max[1],
                (grid_hi[3] - v_h) / dt,
            )

            if acceleration_lower > acceleration_upper:
                raise ValueError(
                    "Human speed cannot return to the grid within "
                    "one time step while respecting acceleration limits."
                )

            human_control = np.array([
                np.clip(
                    yaw_command,
                    human_input_min[0],
                    human_input_max[0],
                ),
                np.clip(
                    acceleration_command,
                    acceleration_lower,
                    acceleration_upper,
                ),
            ])

            next_state = current_joint_state + np.asarray(
                joint_dynamics(
                    jnp.asarray(current_joint_state),
                    jnp.asarray(
                        np.concatenate([ego_control, human_control])
                    ),
                ),
                dtype=float,
            ) * dt

            next_state[3] = np.clip(
                next_state[3], grid_lo[4], grid_hi[4]
            )
            next_state[4] = np.clip(
                next_state[4], grid_lo[5], grid_hi[5]
            )
            next_state[8] = np.clip(
                next_state[8], grid_lo[3], grid_hi[3]
            )

            brt_history.append(np.nan)
            terminal_history.append(np.nan)
            hamiltonian_history.append(np.nan)
            ego_controls.append(ego_control)
            human_controls.append(human_control)

        if step_index < number_of_steps:
            joint_states.append(np.asarray(next_state, dtype=float))

    times = np.asarray(times)
    joint_states = np.asarray(joint_states)
    modes = np.asarray(modes)
    brt_history = np.asarray(brt_history)
    terminal_history = np.asarray(terminal_history)
    hamiltonian_history = np.asarray(hamiltonian_history)
    ego_controls = np.asarray(ego_controls)
    human_controls = np.asarray(human_controls)

    dv_dt_history = np.full_like(brt_history, np.nan)

    valid_indices = np.flatnonzero(np.isfinite(brt_history))
    valid_segments = np.split(
        valid_indices,
        np.flatnonzero(np.diff(valid_indices) > 1) + 1,
    )

    for segment in valid_segments:
        if segment.size >= 2:
            dv_dt_history[segment] = np.gradient(
                brt_history[segment],
                times[segment],
                edge_order=2 if segment.size >= 3 else 1,
            )

    recovery_starts = np.flatnonzero(
        (modes == "recovery")
        & np.concatenate(([True], modes[:-1] != "recovery"))
    )
    recovery_intervals = []
    start = None
    for index, mode in enumerate(modes):
        if mode == "recovery" and start is None:
            start = times[index]
        if start is not None and (mode != "recovery" or index == len(modes) - 1):
            end = times[index] if mode != "recovery" else times[-1]
            recovery_intervals.append((start, end))
            start = None

    figure = plt.figure(figsize=(14, 9), dpi=60)
    figure.suptitle(f"Metric: {metric}", fontsize=17, fontweight="semibold")
    layout = figure.add_gridspec(
        1, 2, width_ratios=(1.25, 1.0), wspace=0.32
    )
    left_layout = layout[0, 0].subgridspec(2, 1, hspace=0.38)
    right_layout = layout[0, 1].subgridspec(5, 1, hspace=0.35)
    road = figure.add_subplot(left_layout[0, 0])
    relative_road = figure.add_subplot(left_layout[1, 0])
    axes = [figure.add_subplot(right_layout[row, 0]) for row in range(5)]
    ego_acceleration_axis = axes[2].twinx()
    human_acceleration_axis = axes[3].twinx()

    ego_x, ego_y, ego_heading = (
        joint_states[:, 0], joint_states[:, 1], joint_states[:, 2]
    )
    human_x, human_y, human_heading = (
        joint_states[:, 5], joint_states[:, 6], joint_states[:, 7]
    )
    margin = 4.5
    road.set_xlim(min(ego_x.min(), human_x.min()) - margin,
                  max(ego_x.max(), human_x.max()) + margin)
    road.set_ylim(min(ego_y.min(), human_y.min()) - margin,
                  max(ego_y.max(), human_y.max()) + margin)
    road.set_aspect("equal", adjustable="box")
    road.grid(True)
    road.set_xlabel("x [m]")
    road.set_ylabel("y [m]")

    ego_body = road.add_patch(plt.Rectangle(
        (-4.68 / 2, -2.20 / 2), 4.68, 2.20,
        facecolor="tab:blue", edgecolor="black", linewidth=1.5, alpha=0.85,
    ))
    human_body = road.add_patch(plt.Rectangle(
        (-4.28 / 2, -1.80 / 2), 4.28, 1.80,
        facecolor="tab:orange", edgecolor="black", linewidth=1.5, alpha=0.85,
    ))
    ego_path, = road.plot([], [], color="tab:blue", label="Ego")
    human_path, = road.plot([], [], color="tab:orange", label="Human")
    road.legend(loc="lower right")
    info = road.text(
        0.52, 1.08, "", transform=road.transAxes, va="bottom", ha="center",
        fontsize=10, fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor="0.35", alpha=0.95),
    )

    plotted_values = [
        brt_history,
        terminal_history,
        hamiltonian_history,
        dv_dt_history,
        np.rad2deg(ego_controls[:, 0]),
        ego_controls[:, 1],
        np.rad2deg(human_controls[:, 0]),
        human_controls[:, 1],
        joint_states[:, 8],
        joint_states[:, 4],
    ]
    lines = [
        axes[0].plot([], [], label=r"$V(-3)$")[0],
        axes[0].plot([], [], label=r"$V(0)$")[0],
        axes[1].plot([], [], color="tab:purple", label="Hamiltonian")[0],
        axes[1].plot([], [], color="tab:orange", linestyle="--", label="dV/dt")[0],
        axes[2].plot([], [], color="tab:green", label="steering rate")[0],
        ego_acceleration_axis.plot([], [], color="tab:red", label="acceleration")[0],
        axes[3].plot([], [], color="tab:brown", label="yaw rate")[0],
        human_acceleration_axis.plot([], [], color="tab:pink", label="acceleration")[0],
        axes[4].plot([], [], color="tab:orange", label=r"$v_H$")[0],
        axes[4].plot([], [], color="tab:blue", label=r"$v_E$")[0],
    ]
    value_axes = [
        axes[0], axes[0], axes[1], axes[1], axes[2], ego_acceleration_axis,
        axes[3], human_acceleration_axis, axes[4], axes[4],
    ]
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Value")
    axes[1].set_ylabel("H")
    axes[2].set_ylabel("Ego steer [deg/s]")
    ego_acceleration_axis.set_ylabel("Ego accel. [m/s²]")
    axes[3].set_ylabel("Human yaw [deg/s]")
    human_acceleration_axis.set_ylabel("Human accel. [m/s²]")
    axes[4].set_ylabel("Speed [m/s]")
    axes[4].set_xlabel("Time [s]")

    for axis in axes:
        axis.set_xlim(0, times[-1])
        axis.grid(True)
        for start, end in recovery_intervals:
            axis.axvspan(start, end, color="red", alpha=0.12)
        for index in recovery_starts:
            axis.axvline(times[index], color="red", linestyle="--",
                        linewidth=1.0, alpha=0.8)
        axis.legend(loc="upper right", fontsize=8)
    ego_acceleration_axis.legend(loc="lower right", fontsize=8)
    human_acceleration_axis.legend(loc="lower right", fontsize=8)
    for axis, values in zip(value_axes, plotted_values):
        finite = values[np.isfinite(values)]
        if finite.size:
            padding = max(0.08 * np.ptp(finite), 0.05)
            lower, upper = axis.get_ylim()
            axis.set_ylim(min(lower, finite.min() - padding),
                          max(upper, finite.max() + padding))

    relative_states = np.empty((len(times), 6))
    dx = human_x - ego_x
    dy = human_y - ego_y
    relative_states[:, 0] = np.cos(ego_heading) * dx + np.sin(ego_heading) * dy
    relative_states[:, 1] = -np.sin(ego_heading) * dx + np.cos(ego_heading) * dy
    relative_states[:, 2] = human_heading - ego_heading
    relative_states[:, 3] = joint_states[:, 8]
    relative_states[:, 4] = joint_states[:, 3]
    relative_states[:, 5] = joint_states[:, 4]

    coordinates = [np.asarray(axis) for axis in grid.coordinate_vectors]
    relative_road.set(
        xlim=(grid_lo[0], grid_hi[0]),
        ylim=(grid_lo[1], grid_hi[1]),
        xlabel=r"$x_{\mathrm{rel}}$ [m]",
        ylabel=r"$y_{\mathrm{rel}}$ [m]",
        title="Relative coordinates — Ego frame",
    )
    relative_road.set_aspect("equal", adjustable="box")
    relative_road.grid(True, alpha=0.35)
    relative_road.axhline(0, color="0.6", linewidth=0.6)
    relative_road.axvline(0, color="0.6", linewidth=0.6)
    relative_road.add_patch(plt.Rectangle(
        (-4.68 / 2, -2.20 / 2), 4.68, 2.20,
        facecolor="tab:blue", edgecolor="black", alpha=0.85, zorder=3,
    ))
    relative_human = relative_road.add_patch(plt.Rectangle(
        (-4.28 / 2, -1.80 / 2), 4.28, 1.80,
        facecolor="tab:orange", edgecolor="black", alpha=0.85, zorder=3,
    ))
    brt_contour = LineCollection(
        [], colors="tab:red", linewidths=1.8, label="BRT = 0", zorder=2
    )
    relative_road.add_collection(brt_contour)
    relative_road.legend(loc="upper right", fontsize=8)
    relative_info = relative_road.text(
        0.02, 0.02, "", transform=relative_road.transAxes,
        fontsize=8, va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )

    @jax.jit
    def interpolate_xy(values, lower, upper, weights):
        result = jnp.zeros(values.shape[:2], dtype=values.dtype)
        for corner in product((0, 1), repeat=4):
            indices = [upper[d] if corner[d] else lower[d] for d in range(4)]
            weight = jnp.prod(jnp.where(jnp.asarray(corner), weights, 1 - weights))
            result = result + weight * values[
                :, :, indices[0], indices[1], indices[2], indices[3]
            ]
        return result

    def contour_at(state):
        lower, upper, weights = [], [], []
        for dimension in range(2, 6):
            value = state[dimension]
            nodes = coordinates[dimension]
            if not np.isfinite(value):
                return [], "BRT slice unavailable"
            if dimension in periodic_dims:
                period = grid_hi[dimension] - grid_lo[dimension]
                value = grid_lo[dimension] + (value - grid_lo[dimension]) % period
                i = int(np.searchsorted(nodes, value, side="right") - 1)
                i = max(0, i)
                j = (i + 1) % len(nodes)
                right = nodes[j] if j else nodes[0] + period
            else:
                if not grid_lo[dimension] <= value <= grid_hi[dimension]:
                    return [], "BRT slice unavailable"
                if len(nodes) == 1:
                    lower.append(0)
                    upper.append(0)
                    weights.append(0.0)
                    continue
                i = int(np.clip(np.searchsorted(nodes, value) - 1, 0, len(nodes) - 2))
                j = i + 1
                right = nodes[j]
            lower.append(i)
            upper.append(j)
            weights.append((value - nodes[i]) / (right - nodes[i]))

        values = np.asarray(interpolate_xy(
            brt_values, jnp.asarray(lower), jnp.asarray(upper),
            jnp.asarray(weights, dtype=brt_values.dtype),
        ))
        paths = contourpy.contour_generator(
            x=coordinates[0], y=coordinates[1], z=values.T,
            line_type="Separate",
        ).lines(0.0)
        return paths, "" if paths else "No zero contour in this slice"

    contour_cache = {}

    def render_frame(frame):
        ego_body.set_transform(
            transforms.Affine2D().rotate(ego_heading[frame])
            .translate(ego_x[frame], ego_y[frame]) + road.transData
        )
        human_body.set_transform(
            transforms.Affine2D().rotate(human_heading[frame])
            .translate(human_x[frame], human_y[frame]) + road.transData
        )
        state = relative_states[frame]
        relative_human.set_transform(
            transforms.Affine2D().rotate(state[2])
            .translate(state[0], state[1]) + relative_road.transData
        )
        if frame not in contour_cache:
            contour_cache[frame] = contour_at(state)
        paths, message = contour_cache[frame]
        brt_contour.set_segments(paths)
        if not (
            grid_lo[0] <= state[0] <= grid_hi[0]
            and grid_lo[1] <= state[1] <= grid_hi[1]
        ):
            message = "Human outside XY view" + (f" · {message}" if message else "")
        relative_info.set_text(message)
        upto = slice(0, frame + 1)
        ego_path.set_data(ego_x[upto], ego_y[upto])
        human_path.set_data(human_x[upto], human_y[upto])
        for line, values in zip(lines, plotted_values):
            line.set_data(times[upto], values[upto])
        mode = modes[frame].upper()
        v_brt = brt_history[frame]
        v_zero = terminal_history[frame]
        v_brt_text = f"{v_brt:.4f}" if np.isfinite(v_brt) else "N/A"
        v_zero_text = f"{v_zero:.4f}" if np.isfinite(v_zero) else "N/A"
        info.set_color("red" if mode == "RECOVERY" else "black")
        info.set_text(
            f"t = {times[frame]:.2f} s    Mode = {mode}\n"
            f"V(-3) = {v_brt_text}    "
            f"V(0) = {v_zero_text}"
        )
        return (ego_body, human_body, ego_path, human_path, info,
                relative_human, brt_contour, relative_info, *lines)

    frame_stride = 4
    frame_indices = np.arange(0, len(times), frame_stride)
    if frame_indices[-1] != len(times) - 1:
        frame_indices = np.append(frame_indices, len(times) - 1)
    movie = animation.FuncAnimation(
        figure,
        render_frame,
        frames=frame_indices,
        interval=1000 * dt * frame_stride / animation_time_scale_factor,
        repeat=False,
        blit=False,
    )
    render_frame(0)
    figure.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.08,
        top=0.87,
    )
    try:
        return HTML(movie.to_html5_video())
    finally:
        plt.close(figure)

__all__ = ["simulate_pursuit_evasion"]
