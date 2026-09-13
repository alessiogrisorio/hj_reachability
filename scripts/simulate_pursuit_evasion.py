from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hj_reachability as hj
from hj_reachability.systems.relative_vehicle_6d import RelativeVehicle6D
from hj_reachability.vehicle.geometry import collision_margin_sat


@dataclass(frozen=True)
class Config:
    brt_file: Path = ROOT / "results" / "brt_euclidean_2.npz"
    initial_mode: str = "random_negative"  # manual, random_negative, random_positive
    manual_state: tuple[float, ...] = (6.0, 2.0, 0.0, 6.0, 0.0, 6.0)
    seed: int = 42
    dt: float = 0.05
    duration: float = 3.0
    negative_threshold: float = -0.20
    positive_threshold: float = 0.20
    minimum_v0: float = 0.10
    entry_cells: float = 2.0
    exit_cells: float = 4.0
    prediction_steps: int = 6
    prediction_dt: float = 0.10
    command_samples: int = 5
    recovery_iterations: int = 1


CFG = Config()


def load_brt(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    lo = np.asarray(data["grid_lo"], dtype=float)
    hi = np.asarray(data["grid_hi"], dtype=float)
    shape = tuple(int(v) for v in data["grid_shape"])
    periodic = tuple(int(v) for v in data["periodic_dims"])
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=jnp.asarray(lo), hi=jnp.asarray(hi)),
        shape=shape,
        periodic_dims=periodic,
    )
    return {
        "grid": grid,
        "dynamics": RelativeVehicle6D(**metadata["dynamics"]["parameters"]),
        "V0": jnp.asarray(data["V0"]),
        "BRT": jnp.asarray(data["BRT"]),
        "gradients": jnp.asarray(data["gradients"]),
        "lo": lo,
        "hi": hi,
        "shape": np.asarray(shape),
        "metadata": metadata,
    }


def select_initial_state(data: dict) -> np.ndarray:
    if CFG.initial_mode == "manual":
        state = np.asarray(CFG.manual_state, dtype=float)
        if state.shape != (6,) or np.any(state < data["lo"]) or np.any(state > data["hi"]):
            raise ValueError("The manual state must contain six in-grid values.")
        return state

    brt = np.asarray(data["BRT"])
    v0 = np.asarray(data["V0"])
    valid = np.isfinite(brt) & np.isfinite(v0) & (v0 > CFG.minimum_v0)
    if CFG.initial_mode == "random_negative":
        valid &= brt <= CFG.negative_threshold
    elif CFG.initial_mode == "random_positive":
        valid &= brt >= CFG.positive_threshold
    else:
        raise ValueError(f"Unknown initial mode: {CFG.initial_mode}")

    valid[[0, -1], ...] = False
    valid[:, [0, -1], ...] = False
    indices = np.argwhere(valid)
    if not len(indices):
        raise RuntimeError("No initial state satisfies the requested conditions.")
    index = indices[np.random.default_rng(CFG.seed).integers(len(indices))]
    return np.array([
        float(data["grid"].coordinate_vectors[d][index[d]]) for d in range(6)
    ])


def state_rate(dynamics, state, control, disturbance, time=0.0) -> np.ndarray:
    x = jnp.asarray(state)
    value = (
        dynamics.open_loop_dynamics(x, time)
        + dynamics.control_jacobian(x, time) @ jnp.asarray(control)
        + dynamics.disturbance_jacobian(x, time) @ jnp.asarray(disturbance)
    )
    return np.asarray(value, dtype=float)


def bound_inputs(data, state, control, disturbance, dt):
    control = np.asarray(control, dtype=float).copy()
    disturbance = np.asarray(disturbance, dtype=float).copy()
    lo, hi = data["lo"], data["hi"]
    disturbance[1] = np.clip(disturbance[1], (lo[3] - state[3]) / dt, (hi[3] - state[3]) / dt)
    control[0] = np.clip(control[0], (lo[4] - state[4]) / dt, (hi[4] - state[4]) / dt)
    control[1] = np.clip(control[1], (lo[5] - state[5]) / dt, (hi[5] - state[5]) / dt)
    return control, disturbance


def rk4_step(data, state, control, disturbance, time, dt):
    dynamics = data["dynamics"]
    f = lambda x, t: state_rate(dynamics, x, control, disturbance, t)
    k1 = f(state, time)
    k2 = f(state + 0.5 * dt * k1, time + 0.5 * dt)
    k3 = f(state + 0.5 * dt * k2, time + 0.5 * dt)
    k4 = f(state + dt * k3, time + dt)
    result = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    result[3:6] = np.clip(result[3:6], data["lo"][3:6], data["hi"][3:6])
    return result


def grid_margins(data, state):
    lo, hi = data["lo"][:3], data["hi"][:3]
    return np.minimum(state[:3] - lo, hi - state[:3]) / (hi - lo)


def inner(data, state, cells):
    spacing = (data["hi"][:3] - data["lo"][:3]) / (data["shape"][:3] - 1)
    return bool(np.all(state[:3] >= data["lo"][:3] + cells * spacing) and
                np.all(state[:3] <= data["hi"][:3] - cells * spacing))


def hj_inputs(data, state, time):
    x = jnp.asarray(state)
    grid = data["grid"]
    gradient = grid.interpolate(data["gradients"], x)
    control, disturbance = data["dynamics"].optimal_control_and_disturbance(
        x, time, gradient
    )
    control, disturbance = bound_inputs(
        data, state, np.asarray(control), np.asarray(disturbance), CFG.dt
    )
    rate = state_rate(data["dynamics"], state, control, disturbance, time)
    return {
        "control": control,
        "disturbance": disturbance,
        "V": float(grid.interpolate(data["BRT"], x)),
        "V0": float(grid.interpolate(data["V0"], x)),
        "H": float(np.dot(np.asarray(gradient), rate)),
    }


def recovery_score(data, state, control, disturbance, time):
    predicted = state.copy()
    worst_margin = np.inf
    return_step = CFG.prediction_steps + 1
    for step in range(CFG.prediction_steps):
        u, d = bound_inputs(data, predicted, control, disturbance, CFG.prediction_dt)
        predicted = rk4_step(
            data, predicted, u, d, time + step * CFG.prediction_dt, CFG.prediction_dt
        )
        worst_margin = min(worst_margin, float(np.min(grid_margins(data, predicted))))
        if return_step > CFG.prediction_steps and inner(data, predicted, CFG.exit_cells):
            return_step = step + 1
    final = grid_margins(data, predicted)
    returned = return_step <= CFG.prediction_steps
    return (int(returned), -return_step if returned else -np.inf,
            float(np.min(final)), float(np.mean(final)), worst_margin)


def best_pair_for_agent(data, state, time, fixed, optimize_ego):
    space = data["dynamics"].control_space if optimize_ego else data["dynamics"].disturbance_space
    lo, hi = np.asarray(space.lo, dtype=float), np.asarray(space.hi, dtype=float)
    first = np.linspace(lo[0], hi[0], CFG.command_samples)
    second = np.linspace(lo[1], hi[1], CFG.command_samples)
    best_input, best_score = np.zeros(2), None
    for a in first:
        for b in second:
            candidate = np.array([a, b])
            control, disturbance = (candidate, fixed) if optimize_ego else (fixed, candidate)
            score = recovery_score(data, state, control, disturbance, time)
            if best_score is None or score > best_score:
                best_input, best_score = candidate, score
    return best_input


def recovery_inputs(data, state, time):
    control = np.zeros(2)
    disturbance = np.zeros(2)
    for _ in range(CFG.recovery_iterations):
        control = best_pair_for_agent(data, state, time, disturbance, True)
        disturbance = best_pair_for_agent(data, state, time, control, False)
    return bound_inputs(data, state, control, disturbance, CFG.dt)


def simulate(data, initial_state):
    state = initial_state.copy()
    mode = "hj"
    history = {key: [] for key in (
        "time", "state", "mode", "control", "disturbance", "V", "V0", "H", "collision_margin"
    )}

    for time in np.arange(0.0, CFG.duration + 0.5 * CFG.dt, CFG.dt):
        if mode == "hj" and not inner(data, state, CFG.entry_cells):
            mode = "recovery"
        elif mode == "recovery" and inner(data, state, CFG.exit_cells):
            mode = "hj"

        if mode == "hj":
            game = hj_inputs(data, state, time)
        else:
            control, disturbance = recovery_inputs(data, state, time)
            game = {"control": control, "disturbance": disturbance,
                    "V": np.nan, "V0": np.nan, "H": np.nan}

        history["time"].append(time)
        history["state"].append(state.copy())
        history["mode"].append(mode)
        history["control"].append(game["control"])
        history["disturbance"].append(game["disturbance"])
        history["V"].append(game["V"])
        history["V0"].append(game["V0"])
        history["H"].append(game["H"])
        history["collision_margin"].append(float(collision_margin_sat(*state[:3])))

        if time >= CFG.duration - 1e-12:
            break
        state = rk4_step(
            data, state, game["control"], game["disturbance"], time, CFG.dt
        )

    return {key: np.asarray(value) for key, value in history.items()}


def recovery_intervals(time, modes):
    mask = modes == "recovery"
    intervals, start = [], None
    for index, active in enumerate(mask):
        if active and start is None:
            start = time[index]
        if start is not None and (not active or index == len(mask) - 1):
            end = time[index] if not active else time[index] + CFG.dt
            intervals.append((start, end))
            start = None
    return intervals


def plot_result(result):
    time = result["time"]
    control = result["control"]
    disturbance = result["disturbance"]
    state = result["state"]
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    axes[0].plot(time, result["V"], label=r"$V(-3,x)$")
    axes[0].plot(time, result["V0"], label=r"$V_0$")
    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[1].plot(time, result["H"], color="tab:purple", label="Hamiltonian")
    ego_acc = axes[2].twinx()
    axes[2].plot(time, np.rad2deg(control[:, 0]), color="tab:green", label="steering rate")
    ego_acc.plot(time, control[:, 1], color="tab:red", label="acceleration")
    human_acc = axes[3].twinx()
    axes[3].plot(time, np.rad2deg(disturbance[:, 0]), color="tab:brown", label="yaw rate")
    human_acc.plot(time, disturbance[:, 1], color="tab:pink", label="acceleration")
    axes[4].plot(time, state[:, 3], label=r"$v_H$")
    axes[4].plot(time, state[:, 5], label=r"$v_E$")
    axes[0].set_ylabel("Value")
    axes[1].set_ylabel("H")
    axes[2].set_ylabel("Ego steer [deg/s]")
    ego_acc.set_ylabel("Ego accel. [m/s²]")
    axes[3].set_ylabel("Human yaw [deg/s]")
    human_acc.set_ylabel("Human accel. [m/s²]")
    axes[4].set_ylabel("Speed [m/s]")
    axes[4].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True)
        for start, end in recovery_intervals(time, result["mode"]):
            axis.axvspan(start, end, color="red", alpha=0.14)
        axis.legend(loc="upper right")
    fig.tight_layout()
    return fig


def reconstruct_absolute_trajectories(result, data):
    time, state = result["time"], result["state"]
    dynamics = data["dynamics"]
    n = len(time)
    ego_x, ego_y, ego_heading = np.zeros(n), np.zeros(n), np.zeros(n)
    beta = np.arctan(dynamics.lr / (dynamics.lr + dynamics.lf) * np.tan(state[:, 4]))
    yaw_rate = state[:, 5] * np.cos(beta) / (dynamics.lr + dynamics.lf) * np.tan(state[:, 4])
    for k in range(n - 1):
        dt = time[k + 1] - time[k]
        ego_heading[k + 1] = ego_heading[k] + 0.5 * dt * (yaw_rate[k] + yaw_rate[k + 1])
        angle_0 = ego_heading[k] + beta[k]
        angle_1 = ego_heading[k + 1] + beta[k + 1]
        ego_x[k + 1] = ego_x[k] + 0.5 * dt * (
            state[k, 5] * np.cos(angle_0) + state[k + 1, 5] * np.cos(angle_1)
        )
        ego_y[k + 1] = ego_y[k] + 0.5 * dt * (
            state[k, 5] * np.sin(angle_0) + state[k + 1, 5] * np.sin(angle_1)
        )
    cosine, sine = np.cos(ego_heading), np.sin(ego_heading)
    human_x = ego_x + cosine * state[:, 0] - sine * state[:, 1]
    human_y = ego_y + sine * state[:, 0] + cosine * state[:, 1]
    return ego_x, ego_y, ego_heading, human_x, human_y, ego_heading + state[:, 2]


def vehicle_polygon(x, y, heading, length, width):
    vertices = np.array([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ])
    rotation = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)],
    ])
    return vertices @ rotation.T + np.array([x, y])


def interpolate_brt_xy_slice(data, state):
    vectors = [np.asarray(v) for v in data["grid"].coordinate_vectors]
    pairs = []
    for dimension in range(2, 6):
        coordinates, value = vectors[dimension], float(state[dimension])
        if value < coordinates[0] or value > coordinates[-1]:
            return None
        upper = int(np.clip(np.searchsorted(coordinates, value, side="right"), 1, len(coordinates) - 1))
        lower = upper - 1
        weight = (value - coordinates[lower]) / (coordinates[upper] - coordinates[lower])
        pairs.append((lower, upper, weight))
    result = np.zeros(tuple(data["shape"][:2]))
    brt = np.asarray(data["BRT"])
    for corner in range(16):
        indices, weight = [], 1.0
        for dimension, (lower, upper, fraction) in enumerate(pairs):
            use_upper = (corner >> dimension) & 1
            indices.append(upper if use_upper else lower)
            weight *= fraction if use_upper else 1.0 - fraction
        result += weight * brt[:, :, indices[0], indices[1], indices[2], indices[3]]
    return result


def create_animation(result, data, fps=20):
    time, state = result["time"], result["state"]
    control, disturbance, modes = result["control"], result["disturbance"], result["mode"]
    ego_x, ego_y, ego_heading, human_x, human_y, human_heading = (
        reconstruct_absolute_trajectories(result, data)
    )
    figure = plt.figure(figsize=(16, 11))
    layout = figure.add_gridspec(5, 2, width_ratios=(1.35, 1.0), hspace=0.28)
    road = figure.add_subplot(layout[:, 0])
    axes = [figure.add_subplot(layout[k, 1]) for k in range(5)]
    ego_acceleration_axis = axes[2].twinx()
    human_acceleration_axis = axes[3].twinx()

    margin = 7.0
    road.set_xlim(min(ego_x.min(), human_x.min()) - margin, max(ego_x.max(), human_x.max()) + margin)
    road.set_ylim(min(ego_y.min(), human_y.min()) - margin, max(ego_y.max(), human_y.max()) + margin)
    road.set_aspect("equal")
    road.grid(True)
    road.set_title("Absolute trajectories and BRT zero contour")
    ego_path, = road.plot([], [], color="tab:blue", label="Ego")
    human_path, = road.plot([], [], color="tab:orange", label="Human")
    ego_body = Polygon(vehicle_polygon(ego_x[0], ego_y[0], ego_heading[0], 4.68, 2.20), color="tab:blue", alpha=0.7)
    human_body = Polygon(vehicle_polygon(human_x[0], human_y[0], human_heading[0], 4.28, 1.80), color="tab:orange", alpha=0.7)
    road.add_patch(ego_body)
    road.add_patch(human_body)
    info = road.text(0.02, 0.98, "", transform=road.transAxes, va="top")
    road.legend(loc="lower right")

    rate_e = np.rad2deg(control[:, 0])
    rate_h = np.rad2deg(disturbance[:, 0])
    series = [
        axes[0].plot([], [], label=r"$V(-3,x)$")[0],
        axes[0].plot([], [], label=r"$V_0$")[0],
        axes[1].plot([], [], color="tab:purple", label="H")[0],
        axes[2].plot([], [], color="tab:green", label="steering rate")[0],
        ego_acceleration_axis.plot([], [], color="tab:red", label="acceleration")[0],
        axes[3].plot([], [], color="tab:brown", label="yaw rate")[0],
        human_acceleration_axis.plot([], [], color="tab:pink", label="acceleration")[0],
        axes[4].plot([], [], label=r"$v_H$")[0],
        axes[4].plot([], [], label=r"$v_E$")[0],
    ]
    all_values = [result["V"], result["V0"], result["H"], rate_e, control[:, 1],
                  rate_h, disturbance[:, 1], state[:, 3], state[:, 5]]
    axes[0].axhline(0, color="black", ls="--", lw=0.8)
    axes[0].set_ylabel("Value")
    axes[1].set_ylabel("H")
    axes[2].set_ylabel("Ego steer [deg/s]")
    ego_acceleration_axis.set_ylabel("Ego accel. [m/s²]")
    axes[3].set_ylabel("Human yaw [deg/s]")
    human_acceleration_axis.set_ylabel("Human accel. [m/s²]")
    axes[4].set_ylabel("Speed [m/s]")
    axes[4].set_xlabel("Time [s]")
    for axis in axes:
        axis.set_xlim(time[0], max(time[-1], CFG.dt))
        axis.grid(True)
        for start, end in recovery_intervals(time, modes):
            axis.axvspan(start, end, color="red", alpha=0.14)
        axis.legend(loc="upper right", fontsize=8)
    for axis, values in zip(
        [axes[0], axes[0], axes[1], axes[2], ego_acceleration_axis,
         axes[3], human_acceleration_axis, axes[4], axes[4]], all_values
    ):
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size:
            padding = max(0.05 * np.ptp(finite), 1e-6)
            current_lo, current_hi = axis.get_ylim()
            axis.set_ylim(min(current_lo, finite.min() - padding), max(current_hi, finite.max() + padding))

    x_vector = np.asarray(data["grid"].coordinate_vectors[0])
    y_vector = np.asarray(data["grid"].coordinate_vectors[1])
    relative_x, relative_y = np.meshgrid(x_vector, y_vector, indexing="ij")
    contour = [None]
    last_slice = [None]

    def update(frame):
        upto = slice(0, frame + 1)
        ego_path.set_data(ego_x[upto], ego_y[upto])
        human_path.set_data(human_x[upto], human_y[upto])
        ego_body.set_xy(vehicle_polygon(ego_x[frame], ego_y[frame], ego_heading[frame], 4.68, 2.20))
        human_body.set_xy(vehicle_polygon(human_x[frame], human_y[frame], human_heading[frame], 4.28, 1.80))
        current_slice = interpolate_brt_xy_slice(data, state[frame])
        current = current_slice is not None
        if current:
            last_slice[0] = current_slice
        else:
            current_slice = last_slice[0]
        if contour[0] is not None:
            contour[0].remove()
            contour[0] = None
        if current_slice is not None and np.nanmin(current_slice) <= 0 <= np.nanmax(current_slice):
            cosine, sine = np.cos(ego_heading[frame]), np.sin(ego_heading[frame])
            absolute_x = ego_x[frame] + cosine * relative_x - sine * relative_y
            absolute_y = ego_y[frame] + sine * relative_x + cosine * relative_y
            contour[0] = road.contour(
                absolute_x, absolute_y, current_slice, levels=[0],
                colors="tab:blue" if current else "gray", linestyles="--",
                linewidths=2, alpha=0.85 if current else 0.3,
            )
        for line, values in zip(series, all_values):
            line.set_data(time[upto], np.asarray(values)[upto])
        mode = modes[frame].upper()
        info.set_color("red" if mode == "RECOVERY" else "black")
        info.set_text(
            f"t = {time[frame]:.2f} s\nmode = {mode}\n"
            f"V = {result['V'][frame]:.4f}\nV0 = {result['V0'][frame]:.4f}\nH = {result['H'][frame]:.4f}"
        )
        return ego_path, human_path, ego_body, human_body, info, *series

    movie = animation.FuncAnimation(
        figure, update, frames=len(time), interval=1000 / fps, repeat=False, blit=False
    )
    update(0)
    figure.tight_layout()
    return figure, movie, None


def main():
    data = load_brt(CFG.brt_file)
    initial_state = select_initial_state(data)
    result = simulate(data, initial_state)
    print("Initial state:", initial_state)
    print("Final state:", result["state"][-1])
    print("Recovery samples:", int(np.sum(result["mode"] == "recovery")))
    print("Collision:", bool(np.any(result["collision_margin"] <= 0.0)))
    plot_result(result)
    plt.show()


if __name__ == "__main__":
    main()
